# SceneTeller: Language-to-3D Scene Generation (ECCV 2024)

### [**Project Page**](https://sceneteller.github.io/) | [**Paper**](https://arxiv.org/abs/2407.20727) | [**YouTube**](https://www.youtube.com/watch?v=N0578Zn_r_U)

SceneTeller generates realistic and high-quality 3D spaces from natural language prompts describing the object placement in the room.

<div align="center">
    <img width="90%" alt="teaser" src="./assets/teaser2.jpg"/>
</div>

## News
- Jul 1, 2024 🔥: SceneTeller is accepted at ECCV 2024!
- Nov/Dec, 2024 : Code will be released.

## Installation & Dependencies

## Dataset

Download [3D-FUTURE](https://tianchi.aliyun.com/dataset/98063) and [preprocessed data](https://drive.google.com/file/d/1NV3pmRpWcehPO5iKJPmShsRp_lNbxJuK/view?usp=sharing) from LayoutGPT to ```./scene_data/```. Then unzip these files. 
```
cd scene_data
unzip 3D-FUTURE-model.zip -d 3D-FUTURE
unzip data_output.zip
```
The 3D scene data split provided by LayoutGPT is located in ```./LayoutGPT/dataset/splits-orig```. We further preprocess the data to remove scenes with overlapping objects and out-of-bounds (OOB) conditions, as well as to generate preliminary rule-based textual descriptions for the scenes. To run the preprocessing script, use the following command:
```
python preprocess_data.py --dataset_dir ./scene_data/data_output --room bedroom 
```
The preprocessed 3D scene data split will be saved in ```./LayoutGPT/dataset/splits-preprocessed```.


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