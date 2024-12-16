# SceneTeller: Language-to-3D Scene Generation (ECCV 2024)

### [**Project Page**](https://sceneteller.github.io/) | [**Paper**](https://arxiv.org/abs/2407.20727) | [**YouTube**](https://www.youtube.com/watch?v=N0578Zn_r_U)

SceneTeller generates realistic and high-quality 3D spaces from natural language prompts describing the object placement in the room.

<div align="center">
    <img width="90%" alt="teaser" src="./assets/teaser2.jpg"/>
</div>

## News
- Dec 16, 2024 🔥🔥: We released SceneTeller!
- Jul 1, 2024 🔥: SceneTeller is accepted at ECCV 2024!

## Installation & Dependencies

1. Clone our repo and create conda environment.
```
git clone https://github.com/sceneteller/SceneTeller.git && cd SceneTeller
conda create -n sceneteller python=3.8 -y
pip install -r requirements.txt
```

2. Install other dependencies.
```
# For scene assembling with ATISS
cd LayoutGen/ATISS
python setup.py build_ext --inplace
pip install -e .
cd ../..

# To render images and segmentation maps
cd BlenderProc
pip install -e .
cd ..

# For stylization
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
cd ..

cd instruct-gs2gs
pip install -e .
```

3. Optional: To create videos of the scenes, download [FFmpeg](https://www.ffmpeg.org/download.html) and add it to your $PATH.

## Dataset

Download [3D-FUTURE-model](https://tianchi.aliyun.com/dataset/65347), [3D-FRONT-texture](https://tianchi.aliyun.com/dataset/65347) and [preprocessed data](https://drive.google.com/file/d/1NV3pmRpWcehPO5iKJPmShsRp_lNbxJuK/view?usp=sharing) from LayoutGPT to ```./scene_data/```. Then unzip these files. 
```
cd scene_data
unzip 3D-FUTURE-model.zip 
unzip 3D-FRONT-texture.zip 
unzip data_output.zip
```
The 3D scene data split provided by LayoutGPT is located in ```./LayoutGen/dataset/splits-orig```. We further preprocess the data to remove scenes with overlapping objects and out-of-bounds (OOB) conditions, as well as to generate preliminary rule-based textual descriptions for the scenes. The preprocessed split is located in ```./LayoutGen/dataset/splits-preprocessed```. You can preprocess the data by yourself by running:
```
python preprocess_data.py --dataset_dir ./scene_data/data_output --room bedroom 
```

## 3D Layout Generation

First set up your OpenAI authentication in the following scripts, then run the scripts.

To enhance the rule-based scene descriptions using GPT, run the following command. Remove the ```--generate_train``` flag to generate descriptions also for the test data with ```--base_output_dir ./llm_output/bedroom-test-prompt```.
```
cd LayoutGen
python run_layoutgen_3d_generateprompt.py --dataset_dir ../scene_data/data_output/bedroom --room bedroom --gpt_type gpt4 --unit px --regular_floor_plan --generate_train --base_output_dir ./llm_output/bedroom-train-prompt
```

To generate 3D layouts, run the following command:
```
python run_layoutgen_3d.py --dataset_dir ../scene_data/data_output/bedroom --icl_type k-similar --K 8 --room bedroom --gpt_type gpt4 --unit px --regular_floor_plan --train_prompt_dir ./llm_output/bedroom-train-prompt/tmp/gpt4 --val_prompt_dir ./llm_output/bedroom-test-prompt/tmp/gpt4 --base_output_dir ./llm_output/bedroom-test/ 
```

## Scene Assembling

Run the following commands to assemble the scenes and render them. Assembled scenes will be saved in ```./LayoutGen/assemble_output/bedroom-test```. 
```
cd ATISS/scripts
python assemble_scenes.py --output_directory bedroom-test --room bedroom --model_output ../../llm_output/bedroom-test/tmp/gpt4
cd ../../..

cd BlenderProc
python ./examples/datasets/front_3d/front_3d_utils.py --assemble_output_dir ../LayoutGen/assemble_output/bedroom-test --hdf_output_dir ../output/hdfs/bedroom-test --scene_output_dir ../output/raw_scenes/bedroom-test --room bedroom
cd ..
```
## Scene Stylization

Run the following command to train the initial GS:
```
ns-train splatfacto --data ./output/raw_scenes/bedroom-test/... --max-num-iterations 20000 nerfstudio-data --train-split-fraction 1
```

To edit GS, run the command:
```
ns-train igs2gs --data ./output/raw_scenes/bedroom-test/...  --load-dir ./output/outputs-splatfacto/.../nerfstudio_models --pipeline.prompt '{"prompt"}' --pipeline.guidance-scale 7.5 --pipeline.image-guidance-scale 1.5 --pipeline.transforms_file ./output/raw_scenes/bedroom-test/.../transforms.json --pipeline.path_segm ./output/raw_scenes/bedroom-test/.../segmentation  --pipeline.room bedroom nerfstudio-data  --train-split-fraction 1
```

You can visualize the stylized scenes with:
```
ns-render interpolate --load-config ./output/outputs-splatfacto/.../config.yml --output-path ./output/renders/stylized_scene.mp4 --pose-source train
```


## 📑 Citation
If you find our work useful, please consider citing:
```
@article{ocal2024sceneteller,
  title     = {SceneTeller: Language-to-3D Scene Generation},
  author    = {{\"O}cal, Ba{\c{s}}ak Melis and Tatarchenko, Maxim and Karaoglu, Sezer and Gevers, Theo},
  journal   = {arXiv preprint arXiv:2407.20727},
  year      = {2024},
}
```
## Acknowledgements

We thank the authors of [LayoutGPT](https://github.com/weixi-feng/LayoutGPT/tree/master), [BlenderProc](https://github.com/DLR-RM/BlenderProc/tree/main), [nerfstudio](https://github.com/nerfstudio-project/nerfstudio/tree/main) and [Instruct-GS2GS](https://github.com/cvachha/instruct-gs2gs/tree/main) for making their code available. Please note that the code provided here is not the official or original version created by the respective individual or organization. Any use of the downstream generation code must comply with the terms and conditions established by the original authors or organizations. 
