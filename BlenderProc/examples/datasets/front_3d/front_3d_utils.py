import os
import argparse
import numpy as np
import json
from PIL import Image
from scipy.spatial import ConvexHull
from itertools import product

def project(bbox, RT, K, scale):

    RT = np.array(RT)
    K = np.array(K)
    R = RT[:, :3]
    T = RT[:, 3:]

    xyz = bbox
    xyz = np.dot(xyz, R.T) + T.T
    xyz = np.dot(xyz, K.T)
    xy = xyz[..., :2] / xyz[..., 2:]
    #xy = (xy/scale).astype(np.uint8)
    return xy


def getPixelsWithinHull(image_vertices):

    hull = ConvexHull(image_vertices)
    image_vertices_within_hull = image_vertices[hull.vertices]

    min_x, min_y = np.min(image_vertices_within_hull, axis=0)
    max_x, max_y = np.max(image_vertices_within_hull, axis=0)

    # Generate grid points within the bounding box
    x_range = np.arange(int(min_x), int(max_x) + 1)
    y_range = np.arange(int(min_y), int(max_y) + 1)
    grid_points = np.array(list(product(x_range, y_range)))

    # Check which grid points are inside the convex hull using vectorized operations
    grid_points_homogeneous = np.column_stack([grid_points, np.ones(len(grid_points))])
    is_inside_hull = hull.equations.dot(grid_points_homogeneous.T) <= 0
    points_within_convex_hull = grid_points[np.all(is_inside_hull, axis=0)]

    return points_within_convex_hull

def renderfromLLMLayout(args):

    hdf_output_dir = args.hdf_output_dir
    scene_output_dir = args.scene_output_dir
    assemble_output_dir = args.assemble_output_dir
    vis_folders = os.listdir(assemble_output_dir)
    vis_folders.sort()

    if not os.path.exists(scene_output_dir):
        os.mkdir(scene_output_dir)

    if not os.path.exists(hdf_output_dir):
        os.mkdir(hdf_output_dir)

    for i in range(0,2):
        folder = vis_folders[i]
        assemble_output_dir_folder = os.path.join(assemble_output_dir, folder)

        front_id = folder.split("_")[0]
        scene_id = folder.split("_")[1]
        scene_output_dir_folder = os.path.join(scene_output_dir, folder)
        if not os.path.exists(scene_output_dir_folder):
            os.mkdir(scene_output_dir_folder)
        command = "blenderproc run examples/datasets/front_3d/main_shell_fromLLM.py {} {} {} {} {} {}".format(front_id, hdf_output_dir, assemble_output_dir_folder, scene_output_dir_folder, scene_id, args.room)
        os.system(command)
        print("Generated hdf of {} scenes".format(i))



def hdftoPngfromLLMLayout(args):

    hdf_output_dir = args.hdf_output_dir
    scene_output_dir = args.scene_output_dir
    hdfs = os.listdir(hdf_output_dir)
    hdfs.sort()

    scenes_done=0
    for i in range(0, len(hdfs)):
        hdf = hdfs[i]
        front_id = hdf.split("_")[0]
        scene_id = hdf.split("_")[1]
        output_images_dir = os.path.join(scene_output_dir, front_id + "_" + scene_id, "images_raw")
        if not os.path.exists(output_images_dir):
            os.mkdir(output_images_dir)
        command = "blenderproc vis hdf5 {}/{} --save {}".format(hdf_output_dir, hdf, output_images_dir)
        os.system(command)


def createTransformsJsonwithSgmentation(args):
    blender_output_path = args.scene_output_dir
    blender_output_folders = os.listdir(blender_output_path)
    blender_output_folders.sort()

    image_size = 512

    for m in range(0,len(blender_output_folders)):
        scene_id = blender_output_folders[m]
        raw_scene_path = os.path.join(blender_output_path, scene_id)

        json_path = os.path.join(raw_scene_path, scene_id + ".json")
        image_path = os.path.join(raw_scene_path, "images_raw")
        segmentation_path = os.path.join(raw_scene_path, "segmentation_raw")
        data = json.load(open(json_path))
        keys = list(data.keys())

        if not os.path.exists('{}/images'.format(raw_scene_path)):
            os.mkdir('{}/images'.format(raw_scene_path))

        if not os.path.exists('{}/segmentation'.format(raw_scene_path)):
            os.mkdir('{}/segmentation'.format(raw_scene_path))

        dict_all = {
            "w": image_size,
            "h": image_size,
            "fl_x": 711.11127387,
            "fl_y": 711.11127387,
            "cx": 255.5,
            "cy": 255.5,
            "k1": 0.000,
            "k2": 0.000,
            "p1": 0.000,
            "p2": 0.000,
            "camera_model": "OPENCV",
            "frames": [],
            "applied_transform": [
                [
                    1.0,
                    0.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0
                ],
                [
                    0.0,
                    0.0,
                    1.0,
                    0.0
                ]
            ],
            "ply_file_path": "sparse_pc.ply"
        }

        for i in range(0, len(keys)):
            bbox_image_center_arr = []

            # FORMAT POSES FOR TRANSFORMS.JSON
            pose = data[keys[i]]['pose']
            dict_elem = {
                "file_path": "images/frame_{:05d}.png".format(i + 1),
                "transform_matrix": pose,
                "colmap_im_id": i + 1,
                "objects": {},
                "objects_3d": {},
                "points_within_convex_hull": {}
            }

            # FORMAT IMAGES FOR TRANSFORMS.JSON
            image_name = os.path.join(image_path, "{}_{}_colors.png".format(scene_id, i))
            original_img = Image.open(image_name)
            original_img = original_img.convert("RGBA")
            white_bg = Image.new("RGBA", original_img.size, (255, 255, 255, 255))
            original_img = np.array(Image.alpha_composite(white_bg,original_img))
            original_img = original_img[:, :, :3]
            original_img = Image.fromarray(original_img)
            original_img.save(os.path.join(raw_scene_path, "images", "frame_{:05d}.png".format(i + 1)))


            # FORMAT SEGMENTATION FOR TRANSFORMS.JSON
            segm_image_name = os.path.join(segmentation_path, "{}.png".format(i))
            segm_original_img = Image.open(segm_image_name)
            segm_original_img.save(os.path.join(raw_scene_path, "segmentation", "frame_{:05d}.png".format(i + 1)))

            # FORMAT 2D BBOXES FOR TRANSFORMS.JSON
            for j in range(0, len(data[keys[i]]['bboxes_list'])):
                K = np.array(data[keys[i]]['K'])
                RT = np.linalg.inv(np.array(pose))[:3]
                B = np.array(data[keys[i]]['bboxes_list'][j])
                vertices = np.array(data[keys[i]]['vertices_list'][j])
                image_bbox = project(B, RT, K, scale=4)
                image_vertices = project(vertices, RT, K, scale=4)
                points_within_convex_hull = getPixelsWithinHull(image_vertices)
                obj = data[keys[i]]['objname_list'][j]


                center = np.mean(image_bbox.T, axis=1)
                bbox_image_center_arr.append(center)

                lds = (image_bbox.min(axis=-2))
                rus = (image_bbox.max(axis=-2))
                lds[0] = image_size - lds[0]
                # lds[1] = 512 - lds[1]
                # lds[1] = 512 - lds[1]
                rus[0] = image_size - rus[0]
                # rus[1] = 512 - rus[1]
                original_img = np.array(original_img)
                points_within_convex_hull[:, 0] = image_size - points_within_convex_hull[:, 0]
                points_within_convex_hull[points_within_convex_hull >= image_size - 1] = image_size - 1
                points_within_convex_hull[points_within_convex_hull <= 0] = 0
          


                if obj not in list(dict_elem['objects'].keys()):
                    dict_elem['objects'][obj] = []
                    dict_elem['objects_3d'][obj] = []
                    dict_elem['points_within_convex_hull'][obj] = []
                    dict_elem['objects'][obj].append([rus[0], lds[1], lds[0], rus[1]])
                    dict_elem['objects_3d'][obj].append(B.tolist())
                    dict_elem['points_within_convex_hull'][obj].append(points_within_convex_hull.tolist())
                else:
                    dict_elem['objects'][obj].append([rus[0], lds[1], lds[0], rus[1]])
                    dict_elem['objects_3d'][obj].append(B.tolist())
                    dict_elem['points_within_convex_hull'][obj].append(points_within_convex_hull.tolist())

            dict_all['frames'].append(dict_elem)

        with open("{}/transforms.json".format(raw_scene_path), "w") as outfile:
            json.dump(dict_all, outfile)
        command = "rm -r {}".format(segmentation_path)
        os.system(command)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='SceneTeller',
                                 description='Preprocess scenes.')
    parser.add_argument('--assemble_output_dir', type=str, default='../LayoutGen/assemble_output/bedroom-test')  
    parser.add_argument('--hdf_output_dir', type=str, default='../output/hdfs/bedroom-test')  
    parser.add_argument('--scene_output_dir', type=str, default='../output/raw_scenes/bedroom-test')  
    parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom', 'livingroom'])
    args = parser.parse_args()

    renderfromLLMLayout(args)
    hdftoPngfromLLMLayout(args)
    createTransformsJsonwithSgmentation(args)




