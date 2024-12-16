"""Allows rendering the content of the scene in the coco file format."""

import datetime
from itertools import groupby
import json
import os
from typing import Optional, Dict, Union, Tuple, List

import numpy as np
from skimage import measure
import cv2
import bpy

from blenderproc.python.utility.LabelIdMapping import LabelIdMapping


def write_coco_annotations(output_dir: str, instance_segmaps: List[np.ndarray],
                           instance_attribute_maps: List[dict],
                           colors: List[np.ndarray], color_file_format: str = "PNG",
                           mask_encoding_format: str = "rle", supercategory: str = "coco_annotations",
                           append_to_existing_output: bool = True,
                           jpg_quality: int = 95, label_mapping: Optional[LabelIdMapping] = None,
                           file_prefix: str = "", indent: Optional[Union[int, str]] = None):
    """ Writes coco annotations in the following steps:
    1. Locate the seg images
    2. Locate the rgb maps
    3. Locate the seg mappings
    4. Read color mappings
    5. For each frame write the coco annotation

    :param output_dir: Output directory to write the coco annotations
    :param instance_segmaps: List of instance segmentation maps
    :param instance_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
    :param colors: List of color images. Does not support stereo images, enter left and right inputs subsequently.
    :param color_file_format: Format to save color images in
    :param mask_encoding_format: Encoding format of the binary masks. Default: 'rle'. Available: 'rle', 'polygon'.
    :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset set
                          by 'bop_dataset_name' or any loaded object with specified 'cp_supercategory'
    :param append_to_existing_output: If true and if there is already a coco_annotations.json file in the output
                                      directory, the new coco annotations will be appended to the existing file.
                                      Also, the rgb images will be named such that there are no collisions.
    :param jpg_quality: The desired quality level of the jpg encoding
    :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                          If None, is given then the `name` field in the csv files is used or - if not existing -
                          the category id itself is used.
    :param file_prefix: Optional prefix for image file names
    :param indent: If indent is a non-negative integer or string, then the annotation output
                   will be pretty-printed with that indent level. An indent level of 0, negative, or "" will
                   only insert newlines. None (the default) selects the most compact representation.
                   Using a positive integer indent indents that many spaces per level.
                   If indent is a string (such as "\t"), that string is used to indent each level.
    """

    if len(colors) > 0 and len(colors[0].shape) == 4:
        raise ValueError("BlenderProc currently does not support writing coco annotations for stereo images. "
                         "However, you can enter left and right images / segmaps separately.")

    # Create output directory
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)

    coco_annotations_path = os.path.join(output_dir, "coco_annotations.json")
    # Calculate image numbering offset, if append_to_existing_output is activated and coco data exists
    if append_to_existing_output and os.path.exists(coco_annotations_path):
        with open(coco_annotations_path, 'r', encoding="utf-8") as fp:
            existing_coco_annotations = json.load(fp)
        image_offset = max(image["id"] for image in existing_coco_annotations["images"]) + 1
    else:
        image_offset = 0
        existing_coco_annotations = None

    # collect all RGB paths
    new_coco_image_paths = []

    # for each rendered frame
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):
        color_rgb = colors[frame - bpy.context.scene.frame_start]

        # Reverse channel order for opencv
        color_bgr = color_rgb.copy()
        color_bgr[..., :3] = color_bgr[..., :3][..., ::-1]

        if color_file_format == 'PNG':
            target_base_path = f'images/{file_prefix}{frame + image_offset:06d}.png'
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(target_path, color_bgr)
        elif color_file_format == 'JPEG':
            target_base_path = f'images/{file_prefix}{frame + image_offset:06d}.jpg'
            target_path = os.path.join(output_dir, target_base_path)
            cv2.imwrite(target_path, color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
        else:
            raise RuntimeError(f'Unknown color_file_format={color_file_format}. Try "PNG" or "JPEG"')


        new_coco_image_paths.append(target_base_path)

    coco_output = _CocoWriterUtility.generate_coco_annotations(instance_segmaps,
                                                               instance_attribute_maps,
                                                               new_coco_image_paths,
                                                               supercategory,
                                                               mask_encoding_format,
                                                               existing_coco_annotations,
                                                               label_mapping)

    print("Writing coco annotations to " + coco_annotations_path)
    with open(coco_annotations_path, 'w', encoding="utf-8") as fp:
        json.dump(coco_output, fp, indent=indent)


def binary_mask_to_rle(binary_mask: np.ndarray) -> Dict[str, List[int]]:
    """Converts a binary mask to COCOs run-length encoding (RLE) format. Instead of outputting
    a mask image, you give a list of start pixels and how many pixels after each of those
    starts are included in the mask.
    :param binary_mask: a 2D binary numpy array where '1's represent the object
    :return: Mask in RLE format
    """
    rle: Dict[str, List[int]] = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle_to_binary_mask(rle: Dict[str, List[int]]) -> np.ndarray:
    """Converts a COCOs run-length encoding (RLE) to binary mask.
    :param rle: Mask in RLE format
    :return: a 2D binary numpy array where '1's represent the object
    """
    binary_array = np.zeros(np.prod(rle.get('size')), dtype=bool)
    counts: List[int] = rle.get('counts')

    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        end = start + counts[i + 1]
        binary_array[start:end] = (i + 1) % 2

    binary_mask = binary_array.reshape(*rle.get('size'), order='F')

    return binary_mask


class _CocoWriterUtility:

    @staticmethod
    def generate_coco_annotations(inst_segmaps, inst_attribute_maps, image_paths, supercategory,
                                  mask_encoding_format, existing_coco_annotations=None,
                                  label_mapping: LabelIdMapping = None):
        """Generates coco annotations for images

        :param inst_segmaps: List of instance segmentation maps
        :param inst_attribute_maps: per-frame mappings with idx, class and optionally supercategory/bop_dataset_name
        :param image_paths: A list of paths which points to the rendered images.
        :param supercategory: name of the dataset/supercategory to filter for, e.g. a specific BOP dataset
        :param mask_encoding_format: Encoding format of the binary mask. Type: string.
        :param existing_coco_annotations: If given, the new coco annotations will be appended to the given
                                          coco annotations dict.
        :param label_mapping: The label mapping which should be used to label the categories based on their ids.
                              If None, is given then the `name` field in the csv files is used or - if not existing -
                              the category id itself is used.
        :return: dict containing coco annotations
        """

        categories = []
        visited_categories = []
        instance_2_category_maps = []

        for inst_attribute_map in inst_attribute_maps:
            instance_2_category_map = {}
            for inst in inst_attribute_map:
                # skip background
                if int(inst["category_id"]) != 0:
                    # take all objects or objects from specified supercategory is defined
                    inst_supercategory = "coco_annotations"
                    if "bop_dataset_name" in inst:
                        inst_supercategory = inst["bop_dataset_name"]
                    elif "supercategory" in inst:
                        inst_supercategory = inst["supercategory"]

                    if supercategory in [inst_supercategory, 'coco_annotations']:
                        if int(inst["category_id"]) not in visited_categories:
                            cat_dict: Dict[str, Union[str, int]] = {'id': int(inst["category_id"]),
                                                                    'supercategory': inst_supercategory}
                            # Determine name of category based on label_mapping, name or category_id
                            if label_mapping is not None:
                                cat_dict["name"] = label_mapping.label_from_id(cat_dict['id'])
                            elif "name" in inst:
                                cat_dict["name"] = inst["name"]
                            else:
                                cat_dict["name"] = inst["category_id"]

                            categories.append(cat_dict)
                            visited_categories.append(cat_dict['id'])
                        instance_2_category_map[int(inst["idx"])] = int(inst["category_id"])
            instance_2_category_maps.append(instance_2_category_map)

        licenses = [{
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
        }]
        info = {
            "description": supercategory,
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2020,
            "contributor": "Unknown",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        images: List[Dict[str, Union[str, int]]] = []
        annotations: List[Dict[str, Union[str, int]]] = []

        for inst_segmap, image_path, instance_2_category_map in zip(inst_segmaps, image_paths,
                                                                    instance_2_category_maps):

            # Add coco info for image
            image_id = len(images)
            images.append(_CocoWriterUtility.create_image_info(image_id, image_path, inst_segmap.shape))

            # Go through all objects visible in this image
            instances = np.unique(inst_segmap)
            # Remove background
            instances = np.delete(instances, np.where(instances == 0))
            for inst in instances:
                if inst in instance_2_category_map:
                    # Calc object mask
                    binary_inst_mask = np.where(inst_segmap == inst, 1, 0)
                    # Add coco info for object in this image
                    annotation = _CocoWriterUtility.create_annotation_info(len(annotations) + 1,
                                                                           image_id,
                                                                           instance_2_category_map[inst],
                                                                           binary_inst_mask,
                                                                           mask_encoding_format)
                    if annotation is not None:
                        annotations.append(annotation)

        new_coco_annotations = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": images,
            "annotations": annotations
        }

        if existing_coco_annotations is not None:
            new_coco_annotations = _CocoWriterUtility.merge_coco_annotations(existing_coco_annotations,
                                                                             new_coco_annotations)

        return new_coco_annotations

    @staticmethod
    def merge_coco_annotations(existing_coco_annotations, new_coco_annotations):
        """ Merges the two given coco annotation dicts into one.

        Currently, this requires both coco annotations to have the exact same categories/objects.
        The "images" and "annotations" sections are concatenated and respective ids are adjusted.

        :param existing_coco_annotations: A dict describing the first coco annotations.
        :param new_coco_annotations: A dict describing the second coco annotations.
        :return: A dict containing the merged coco annotations.
        """

        # Concatenate category sections
        for cat_dict in new_coco_annotations["categories"]:
            if cat_dict not in existing_coco_annotations["categories"]:
                existing_coco_annotations["categories"].append(cat_dict)

        # Concatenate images sections
        image_id_offset = max(image["id"] for image in existing_coco_annotations["images"]) + 1
        for image in new_coco_annotations["images"]:
            image["id"] += image_id_offset
        existing_coco_annotations["images"].extend(new_coco_annotations["images"])

        # Concatenate annotations sections
        if len(existing_coco_annotations["annotations"]) > 0:
            annotation_id_offset = max(annotation["id"] for annotation in existing_coco_annotations["annotations"])
        else:
            annotation_id_offset = 0
        for annotation in new_coco_annotations["annotations"]:
            annotation["id"] += annotation_id_offset
            annotation["image_id"] += image_id_offset
        existing_coco_annotations["annotations"].extend(new_coco_annotations["annotations"])

        return existing_coco_annotations

    @staticmethod
    def create_image_info(image_id: int, file_name: str, image_size: Tuple[int, int]) -> Dict[str, Union[str, int]]:
        """Creates image info section of coco annotation

        :param image_id: integer to uniquly identify image
        :param file_name: filename for image
        :param image_size: The size of the image, given as [W, H]
        """
        image_info: Dict[str, Union[str, int]] = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[1],
            "height": image_size[0],
            "date_captured": datetime.datetime.utcnow().isoformat(' '),
            "license": 1,
            "coco_url": "",
            "flickr_url": ""
        }

        return image_info

    @staticmethod
    def create_annotation_info(annotation_id: int, image_id: int, category_id: int, binary_mask: np.ndarray,
                               mask_encoding_format: str, tolerance: int = 2) -> Optional[Dict[str, Union[str, int]]]:
        """Creates info section of coco annotation

        :param annotation_id: integer to uniquly identify the annotation
        :param image_id: integer to uniquly identify image
        :param category_id: Id of the category
        :param binary_mask: A binary image mask of the object with the shape [H, W].
        :param mask_encoding_format: Encoding format of the mask. Type: string.
        :param tolerance: The tolerance for fitting polygons to the objects mask.
        """

        area = _CocoWriterUtility.calc_binary_mask_area(binary_mask)
        if area < 1:
            return None

        bounding_box = _CocoWriterUtility.bbox_from_binary_mask(binary_mask)

        if mask_encoding_format == 'rle':
            segmentation = binary_mask_to_rle(binary_mask)
        elif mask_encoding_format == 'polygon':
            segmentation = _CocoWriterUtility.binary_mask_to_polygon(binary_mask, tolerance)
            if not segmentation:
                return None
        else:
            raise RuntimeError(f"Unknown encoding format: {mask_encoding_format}")

        annotation_info: Dict[str, Union[str, int]] = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0,
            "area": area,
            "bbox": bounding_box,
            "segmentation": segmentation,
            "width": binary_mask.shape[1],
            "height": binary_mask.shape[0],
        }
        return annotation_info

    @staticmethod
    def bbox_from_binary_mask(binary_mask: np.ndarray) -> List[int]:
        """ Returns the smallest bounding box containing all pixels marked "1" in the given image mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The bounding box represented as [x, y, width, height]
        """
        # Find all columns and rows that contain 1s
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        # Find the min and max col/row index that contain 1s
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # Calc height and width
        h = rmax - rmin + 1
        w = cmax - cmin + 1
        return [int(cmin), int(rmin), int(w), int(h)]

    @staticmethod
    def calc_binary_mask_area(binary_mask: np.ndarray) -> int:
        """ Returns the area of the given binary mask which is defined as the number of 1s in the mask.

        :param binary_mask: A binary image mask with the shape [H, W].
        :return: The computed area
        """
        return binary_mask.sum().tolist()

    @staticmethod
    def close_contour(contour: np.ndarray) -> np.ndarray:
        """ Makes sure the given contour is closed.

        :param contour: The contour to close.
        :return: The closed contour.
        """
        # If first != last point => add first point to end of contour to close it
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    @staticmethod
    def binary_mask_to_polygon(binary_mask: np.ndarray, tolerance: int = 0) -> List[np.ndarray]:
        """Converts a binary mask to COCO polygon representation

         :param binary_mask: a 2D binary numpy array where '1's represent the object
         :param tolerance: Maximum distance from original points of polygon to approximated polygonal chain. If
                           tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = np.array(measure.find_contours(padded_binary_mask, 0.5))
        # Reverse padding
        contours -= 1
        for contour in contours:
            # Make sure contour is closed
            contour = _CocoWriterUtility.close_contour(contour)
            # Approximate contour by polygon
            polygon = measure.approximate_polygon(contour, tolerance)
            # Skip invalid polygons
            if len(polygon) < 3:
                continue
            # Flip xy to yx point representation
            polygon = np.flip(polygon, axis=1)
            # Flatten
            polygon = polygon.ravel()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            polygon[polygon < 0] = 0
            polygons.append(polygon.tolist())

        return polygons
