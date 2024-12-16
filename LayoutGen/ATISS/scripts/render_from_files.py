#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

"""Script used for visualizing 3D-FRONT room specified by its scene_id."""
import argparse
import difflib
import json
import logging
import os
import os.path as op
import sys

import numpy as np
from tqdm import tqdm

from training_utils import load_config
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.utils import get_textured_objects
from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from utils import floor_plan_from_scene, export_scene


def load_room_boxes(prefix, id, stats):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset = min(data['floor_plan_vertices'][:, 0])
    y_offset = min(data['floor_plan_vertices'][:, 2])
    room_length = max(data['floor_plan_vertices'][:, 0]) - min(data['floor_plan_vertices'][:, 0])
    room_width = max(data['floor_plan_vertices'][:, 2]) - min(data['floor_plan_vertices'][:, 2])

    return room_length, room_width, x_c, y_c, x_offset, y_offset

dict_catmap = {
    'closet': 'wardrobe',
    'queen_size_bed': 'double_bed',
    'bedside_table': 'nightstand',
    'television_stand': 'tv_stand',
    'children_bed': 'kids_bed',
    'armoire': 'wardrobe',
    'cupboard': 'wardrobe',
    'vanity_table': 'dressing_table',
    'two_person_bed': 'double_bed',
    'twin_bed': 'single_bed',
    'queen-sized_bed': 'double_bed',
    'childrens_bed': 'kids_bed',
    'queen_bed': 'double_bed',
    "children's_cabinet": 'children_cabinet',
    "kids_cabinet": 'children_cabinet',
    "writing_desk": "desk",
    "solitary_bed": "single_bed",
    "storage_cabinet": "cabinet",
    "standing_wardrobe": "wardrobe",
    "two-person_bed": "double_bed",
    "small_table": "nightstand"
}
def denormalize(predictions, prefix, all_data, is_bedroom):

    dict = {
        "iter": 0,
        "object_list": [],
        "query_id" : all_data["id"],
        "prompt": all_data["prompt"][-1]["content"]
    }
    obj_list = []
    predictions = predictions.split("\n")
    for i in range(0, len(predictions)):
        if predictions[i] != "":
            obj = predictions[i].split(" ")[0]
            if obj in list(dict_catmap.keys()):
                obj = dict_catmap[obj]
            metadata = predictions[i].split("{")[1][:-1]
            length = float(metadata.split(";")[0].split(":")[1][1:-2])
            width = float(metadata.split(";")[1].split(":")[1][1:-2])
            height = float(metadata.split(";")[2].split(":")[1][1:-2])
            left = float(metadata.split(";")[3].split(":")[1][1:-2])
            top = float(metadata.split(";")[4].split(":")[1][1:-2])
            depth = float(metadata.split(";")[5].split(":")[1][1:-2])
            orientation = int(metadata.split(";")[6].split(":")[1][1:-8])
            dict_obj = {
                "length": length,
                "width": width,
                "height": height,
                "left": left,
                "top": top,
                "depth": depth,
                "orientation": orientation,
            }
            dict['object_list'].append([obj, dict_obj])
            obj_list.append(obj)
    predictions = [dict]

    for v in predictions:
        rl, rw, x_c, y_c, x_offset, y_offset = load_room_boxes(prefix, v["query_id"], None)

        for _, box in v['object_list']:
            for attr_name, attr_value in box.items():
                if attr_name == 'orientation':
                    box[attr_name] = (attr_value / 180.) * np.pi
                else:
                    box[attr_name] = attr_value

                if attr_name == 'left':
                    box[attr_name] += (x_offset - x_c)
                if attr_name == 'top':
                    box[attr_name] += (y_offset - y_c)

                if attr_name in ['length', 'width', 'height']:
                    box[attr_name] /= 2.
    return predictions, obj_list


def main(argv):
    parser = argparse.ArgumentParser(
        description="Visualize a 3D-FRONT room from json file"
    )
    parser.add_argument(
        "config_file",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "output_directory",
        help="Path to output directory"
    )
    parser.add_argument(
        "path_to_pickled_3d_futute_models",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "path_to_floor_plan_textures",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "model_output",
        help="Path to model output"
    )
    parser.add_argument(
        '--room', 
        type=str, 
        default='bedroom', 
        choices=['bedroom', 'livingroom'])
    parser.add_argument(
        "--without_floor_layout",
        action="store_true",
        help="Visualize also the rooom's floor"
    )
    parser.add_argument(
        "--export_scene",
        action="store_true",
        help="Export scene"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'test', 'val', 'test_regular', 'train_regular', 'val_regular'],
        default='test_regular'
    )

    args = parser.parse_args(argv)
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

        # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models, 
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    config = load_config(args.config_file)
    if '_regular' in args.split:
        config['data']['annotation_file'] = config['data']['annotation_file'].replace("_new.csv",
                                                                                      "_regular.csv")
        args.split = args.split.split("_")[0]

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=[args.split]
        ),
        split=[args.split]
    )

    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )
    classes = np.array(dataset.class_labels)

    all_data = json.load(open(args.model_output, "r"))
    predictions = all_data['choices'][0]['message']['content']

    predictions, obj_list = denormalize(predictions, config['data']['dataset_directory'], all_data,
                              args.room in args.path_to_pickled_3d_futute_models)

    all_ids = [pred['query_id'].split("_")[-1] for pred in predictions]
    predictions = {pred['query_id'].split("_")[-1]: pred for pred in predictions}

    for i in range(0, len(raw_dataset)):
        s = raw_dataset[i]
        if s.scene_id not in all_ids: continue

        bbox_params = [np.zeros(len(classes) + 3 + 3 + 1)]
        for obj in predictions[str(s.scene_id)]['object_list']:
            try:
                label_idx = np.where(classes == difflib.get_close_matches(obj[0], classes, cutoff=0.0)[0])[0]
            except:
                continue
            label = np.zeros(len(classes))
            label[label_idx] = 1
            translation = np.asarray([obj[1]['left'], obj[1]['depth'], obj[1]['top']])
            size = np.asarray([obj[1]['length'], obj[1]['height'], obj[1]['width']])
            angle = np.asarray([obj[1]['orientation']])
            param = np.concatenate([label, translation, size, angle])
            bbox_params.append(param)
        bbox_params.append(np.zeros(len(classes) + 3 + 3 + 1))
        bbox_params_t = np.stack(bbox_params)[None, ...]

        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )

        if not args.without_floor_layout:
            # Get a floor plan
            tr_floor, _ = floor_plan_from_scene(
                s, args.path_to_floor_plan_textures, without_room_mask=True
            )


        if args.export_scene:
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes, tr_floor, obj_list)


if __name__ == "__main__":
    main(sys.argv[1:])