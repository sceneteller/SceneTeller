"""
Nerfstudio InstructGS2GS Pipeline
"""

import matplotlib.pyplot as plt
import pdb
import typing
from dataclasses import dataclass, field
from itertools import cycle
from typing import Literal, Optional, Type

import torch
import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

#eventually add the igs2gs datamanager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig,FullImageDatamanager
from igs2gs.igs2gs import InstructGS2GSModel,InstructGS2GSModelConfig
from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig

from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from igs2gs.ip2p import InstructPix2Pix

from PIL import Image
import torchvision.transforms as transforms
import json
import torchvision
import numpy as np
import os

@dataclass
class InstructGS2GSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""
    
    _target: Type = field(default_factory=lambda: InstructGS2GSPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = InstructGS2GSDataManagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = InstructGS2GSModelConfig()
    """specifies the model config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    room: str = ""
    """room type for InstructPix2Pix"""
    path_segm: str = ""
    """path for segmentation maps for InstructPix2Pix"""
    transforms_file: str = ""
    """transforms file InstructPix2Pix"""
    guidance_scale: float = 12.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    gs_steps: int = 2500
    """how many GS steps between dataset updates"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.7
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""

class InstructGS2GSPipeline(VanillaPipeline):
    """InstructGS2GS Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """
    
    def __init__(
        self,
        config: InstructGS2GSPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        
        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)

        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )

        # which image index we are editing
        self.curr_edit_idx = 0
        # whether we are doing regular GS updates or editing images
        self.makeSquentialEdits = False

        # get transforms file
        if not self.config.transforms_file == "":
            self.transforms_file = json.load(open(self.config.transforms_file))
        else:
            self.transforms_file = None

        path_id = self.config.transforms_file.split("/")[-2]
        path_segm = config.path_segm
        self.segm_map_paths = os.listdir(path_segm)
        self.segm_map_paths.sort()
        self.segm_maps = []
        for i in range(0, len(self.segm_map_paths)):
            map = np.array(Image.open(os.path.join(path_segm, self.segm_map_paths[i])))
            map = map.astype(np.int32)
            self.segm_maps.append(torch.from_numpy(map))


        if config.room == "bedroom":
            self.class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "floor_lamp", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "start", "end"]
        else:
            self.class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair",
                        "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table",
                        "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa",
                        "pendant_lamp", "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "start", "end"]

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
      
        if ((step-1) % self.config.gs_steps) == 0:
            self.makeSquentialEdits = True

        if (not self.makeSquentialEdits):
            camera, data = self.datamanager.next_train(step)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)
        else:
            # get index
            idx = self.curr_edit_idx
            camera, data = self.datamanager.next_train_idx(idx)
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, data)

            original_image = self.datamanager.original_cached_train[idx]["image"].unsqueeze(dim=0).permute(0, 3, 1, 2)
            rendered_image = model_outputs["rgb"].detach().unsqueeze(dim=0).permute(0, 3, 1, 2)

            edited_image = self.ip2p.edit_image(
                        self.text_embedding.to(self.ip2p_device),
                        rendered_image.to(self.ip2p_device),
                        original_image.to(self.ip2p_device),
                        None,
                        self.transforms_file,
                        guidance_scale=self.config.guidance_scale,
                        image_guidance_scale=self.config.image_guidance_scale,
                        diffusion_steps=self.config.diffusion_steps,
                        lower_bound=self.config.lower_bound,
                        upper_bound=self.config.upper_bound,
                    )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')


            # write edited image to dataloader
            edited_image = edited_image.to(original_image.dtype)
            if self.transforms_file != None:
                obj_name = ""
                promptcheck = self.config.prompt[1:-1]
                obj_keys = list(self.transforms_file['frames'][idx]['objects'].keys())
                for o in range(0, len(obj_keys)):
                    if obj_keys[o] in promptcheck:
                        obj_name = obj_keys[o]
                        break
                if ("bed" in promptcheck):
                    if "double_bed" in obj_keys:
                        obj_name = "double_bed"
                    elif "single_bed" in obj_keys:
                        obj_name = "single_bed"
                if ("mirror" in promptcheck):
                    obj_name = "wardrobe"
                if obj_name != "":
                    # LOAD CAM DATA & SEGM MAPS
                    K = np.eye(3)
                    K[0, 0] = float(self.transforms_file['fl_x'])
                    K[1, 1] = float(self.transforms_file['fl_y'])
                    K[0, 2] = float(self.transforms_file['cx'])
                    K[1, 2] = float(self.transforms_file['cy'])
                    pose = np.array(self.transforms_file['frames'][idx]['transform_matrix'])[:3]
                    pose_trans = pose[:, 3]
                    class_label_id = self.class_labels.index(obj_name)
                    segm_map = self.segm_maps[idx]

                    # GET TARGET BBOXES
                    bboxes = self.transforms_file['frames'][idx]['objects'][obj_name]
                    mask = torch.zeros((edited_image.shape[0], 3, edited_image.shape[2], edited_image.shape[3]))
                    for b in range(0, len(bboxes)):
                        mask_box = torch.zeros((edited_image.shape[0], 3, edited_image.shape[2], edited_image.shape[3]))
                        mask_box[:, :, int(bboxes[b][1]):int(bboxes[b][3]), int(bboxes[b][0]):int(bboxes[b][2])] = 1.0
                        mask[(segm_map == class_label_id) & (mask_box==1.0)] = 1.0
                        bbox_3d = np.array(self.transforms_file['frames'][idx]['objects_3d'][obj_name][b])
                        center = np.mean(bbox_3d.T, axis=1)
                        dist_bbox_3d = np.linalg.norm(pose_trans - center)

                        # GET OTHER BBOXES
                        for m in range(0, len(obj_keys)):
                            if obj_keys[m] != obj_name:
                                bboxes_other = np.array(self.transforms_file['frames'][idx]['objects'][obj_keys[m]])
                                bboxes_3d_other = np.array(self.transforms_file['frames'][idx]['objects_3d'][obj_keys[m]])
                                for n in range(0, len(bboxes_3d_other)):
                                    bbox_other = bboxes_other[n]
                                    bbox_3d_other = bboxes_3d_other[n]
                                    center_other = np.mean(bbox_3d_other.T, axis=1)
                                    dist_bbox_3d_other = np.linalg.norm(pose_trans - center_other)
                                    if dist_bbox_3d_other < dist_bbox_3d:
                                        class_label_id_other = self.class_labels.index(obj_keys[m])
                                        mask_box_other = torch.zeros((edited_image.shape[0], 3, edited_image.shape[2], edited_image.shape[3]))
                                        mask_box_other[:, :, int(bbox_other[1]):int(bbox_other[3]),int(bbox_other[0]):int(bbox_other[2])] = 1.0
                                        mask[(segm_map==class_label_id_other) & (mask_box_other==1.0)] = 0.0              

                    original_image = original_image.to("cuda")
                    mask = mask.to("cuda")
                    edited_image[mask != 1.0] = original_image[mask != 1.0]
                else:
                    segm_map = self.segm_maps[idx]
                    mask = torch.zeros((edited_image.shape[0], 3, edited_image.shape[2], edited_image.shape[3]))
                    obj_keys = list(self.transforms_file['frames'][idx]['objects'].keys())
                    for m in range(0, len(obj_keys)):
                        bboxes_other = np.array(self.transforms_file['frames'][idx]['objects'][obj_keys[m]])
                        for n in range(0, len(bboxes_other)):
                            bbox_other = bboxes_other[n]
                            class_label_id_other = self.class_labels.index(obj_keys[m])
                            mask_box_other = torch.zeros((edited_image.shape[0], 3, edited_image.shape[2], edited_image.shape[3]))
                            mask_box_other[:, :, int(bbox_other[1]):int(bbox_other[3]),int(bbox_other[0]):int(bbox_other[2])] = 1.0
                            mask[(segm_map == class_label_id_other) & (mask_box_other == 1.0)] = 1.0

            self.datamanager.cached_train[idx]["image"] = edited_image.squeeze().permute(1,2,0)
            data["image"] = edited_image.squeeze().permute(1,2,0)

            #increment curr edit idx
            self.curr_edit_idx += 1
            if (self.curr_edit_idx >= len(self.datamanager.cached_train)):
                self.curr_edit_idx = 0
                self.makeSquentialEdits = False


        loss_dict = self.model.get_loss_dict(model_outputs, data, metrics_dict)
        
        return model_outputs, loss_dict, metrics_dict
    
    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    