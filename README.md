# SceneTeller: Language-to-3D Scene Generation (ECCV 2024)

### [**Project Page**](https://sceneteller.github.io/) | [**Paper**](https://arxiv.org/abs/2407.20727) | [**YouTube**](https://www.youtube.com/watch?v=N0578Zn_r_U)

SceneTeller generates realistic and high-quality 3D spaces from natural language prompts describing the object placement in the room.

<div align="center">
    <img width="90%" alt="teaser" src="./assets/teaser2.jpg"/>
</div>

## News
- Jul 1, 2024 🔥: SceneTeller is accepted at ECCV 2024!
- Nov, 2024 : Code will be released.

## Installation & Dependencies

## Dataset

Download [3D-FUTURE](https://tianchi.aliyun.com/dataset/98063) and [preprocessed data](https://drive.google.com/file/d/1NV3pmRpWcehPO5iKJPmShsRp_lNbxJuK/view?usp=sharing) from LayoutGPT to ```./data/```. Then unzip these files. 
```
cd data
unzip 3D-FUTURE-model.zip -d 3D-FUTURE
unzip data_output.zip
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