# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-GS2GS configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig,FullImageDatamanager
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.engine.trainer import TrainerConfig


from igs2gs.igs2gs_datamanager import InstructGS2GSDataManagerConfig
from igs2gs.igs2gs import InstructGS2GSModelConfig
from igs2gs.igs2gs_pipeline import InstructGS2GSPipelineConfig
from igs2gs.igs2gs_trainer import InstructGS2GSTrainerConfig


igs2gs_method = MethodSpecification(
    config=InstructGS2GSTrainerConfig(
        method_name="igs2gs",
        steps_per_eval_image=100,
        steps_per_eval_batch=100,
        steps_per_save=500,
        steps_per_eval_all_images=100000, 
        max_num_iterations=7500,
        mixed_precision=False,
        gradient_accumulation_steps = {'camera_opt': 100,'color':10,'shs':10},
        pipeline=InstructGS2GSPipelineConfig(
            datamanager=InstructGS2GSDataManagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=InstructGS2GSModelConfig(),
        ),
    optimizers={
        "xyz": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=30000,
            ),
        },
        "features_dc": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
        "features_rest": {
            "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
            "scheduler": None,
        },
        "opacity": {
            "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
            "scheduler": None,
        },
        "scaling": {
            "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
            "scheduler": None,
        },
        "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
        },
    },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-GS2GS primary method: uses LPIPS, IP2P at full precision",

)

