import blenderproc as bproc
import argparse
import os
import numpy as np
import json
import bpy
import trimesh
from blenderproc.python.loader.ObjectLoader import load_obj
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("front_id", help="Path to the 3D front file")
parser.add_argument("hdf_output_dir", help="Path to where the data should be saved")
parser.add_argument("assemble_output_dir_folder", help="Path to where the data should be saved")
parser.add_argument("scene_output_dir_folder", help="Path to where the json data should be saved")
parser.add_argument("scene_id", help="Scene id")
parser.add_argument('room', type=str, default='bedroom', choices=['bedroom', 'livingroom'])
args = parser.parse_args()

bproc.init()
mapping_file = bproc.utility.resolve_resource(os.path.join("front_3D", "3D_front_mapping.csv"))
mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

def main():
    bproc.renderer.set_light_bounces(diffuse_bounces=200, glossy_bounces=200, max_bounces=200,
                                     transmission_bounces=200, transparent_max_bounces=200)
    bproc.renderer.set_output_format(enable_transparency=True)

    if not os.path.exists(os.path.join(args.scene_output_dir_folder, "scene_objs")):
        os.mkdir(os.path.join(args.scene_output_dir_folder, "scene_objs"))

    path_atiss = args.assemble_output_dir_folder
    files = os.listdir(path_atiss)
    obj_list = []
    mat_list = []
    for i in range(0, len(files)):
        if ".obj" in files[i]:
            if "floor" not in files[i]:
                obj_list.append(files[i])
        if ".mtl" in files[i]:
            mat_list.append(files[i])
    obj_list.sort()
    mat_list.sort()
    if args.room == "bedroom":
        class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "floor_lamp", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe", "start", "end"]
    else:
        class_labels = ["armchair", "bookshelf", "cabinet", "ceiling_lamp", "chaise_longue_sofa", "chinese_chair",
                    "coffee_table", "console_table", "corner_side_table", "desk", "dining_chair", "dining_table",
                    "l_shaped_sofa", "lazy_sofa", "lounge_chair", "loveseat_sofa", "multi_seat_sofa", "pendant_lamp",
                    "round_end_table", "shelf", "stool", "tv_stand", "wardrobe", "wine_cabinet", "start", "end"]

    loaded_objects = []
    objname_list = []
    bboxes_list = []
    vertices_list = []
    for i in range(0, len(obj_list)):
        # Load obj properties
        obj_file = os.path.join(path_atiss, obj_list[i])
        obj_id = obj_list[i].split("_")[1].split("-")[0]
        obj_name = obj_list[i].split("-")[1][:-4]
        # Copy obj to scene_obj dict
        new_obj_path = os.path.join(args.scene_output_dir_folder, "scene_objs", obj_name + obj_id.lstrip("0") + ".obj")
        command = "cp {} {}".format(obj_file, new_obj_path)
        os.system(command)
        obj = load_obj(filepath=obj_file)[0]
        objname_list.append(obj_name)
        bboxes_list.append(obj.get_bound_box().tolist())
        class_label_id = class_labels.index(obj_name)
        obj.set_cp("category_id", class_label_id)


        # Load material properties
        for mat in obj.get_materials():
            if mat is None:
                continue
            principled_node = mat.get_nodes_with_type("BsdfPrincipled")
            if len(principled_node) == 0:
                continue
            if len(principled_node) == 1:
                principled_node = principled_node[0]
            principled_node.inputs["Emission"].default_value[:3] = [0, 0, 0]
            principled_node.inputs["Transmission"].default_value = 0
            image_node = mat.new_node('ShaderNodeTexImage')
            base_image_path = os.path.join(path_atiss, "material_{}-{}.png".format(obj_id,obj_name))
            image_node.image = bpy.data.images.load(base_image_path, check_existing=True)
            mat.link(image_node.outputs['Color'], principled_node.inputs['Base Color'])
        loaded_objects.append(obj)

    # Load floor obj
    floor_file = os.path.join(path_atiss, "floor.obj")
    new_obj_path = os.path.join(args.scene_output_dir_folder, "scene_objs", "room.obj")
    command = "cp {} {}".format(floor_file, new_obj_path)
    os.system(command)
    floor_obj = load_obj(filepath=floor_file)[0]
    floor_obj.set_cp("category_id", 45)

    loaded_objects[0].persist_transformation_into_mesh()
    verts_obj = np.array(loaded_objects[0].mesh_as_trimesh().vertices)
    vertices_list.append(verts_obj.tolist())
    for i in range(1, len(loaded_objects)):
        loaded_objects[i].persist_transformation_into_mesh()
        verts_obj_i = np.array(loaded_objects[i].mesh_as_trimesh().vertices)
        vertices_list.append(verts_obj_i.tolist())
        verts_obj = np.concatenate((verts_obj, verts_obj_i), axis=0)
    cloud = trimesh.points.PointCloud(vertices=verts_obj)
    cloud.export(os.path.join(args.scene_output_dir_folder,"sparse_pc.ply"))

    bounding_box = floor_obj.get_bound_box()
    center = np.mean(bounding_box.T,axis=1)
    min_corner = np.min(bounding_box, axis=0)
    max_corner = np.max(bounding_box, axis=0)
    diagonal_len = np.linalg.norm(max_corner - min_corner)
    bproc.renderer.set_world_background([1.0,1.0,1.0])

    bproc.camera.set_resolution(image_width=512, image_height=512)
    poses = 0
    tries = 0
    num_poses = 250
    degree_increment = float(360.0 / num_poses)

    dict_all = {}
    while tries < 50000 and poses < num_poses:
        if poses % 10 == 0:
            print(poses)
        location = bproc.sampler.shell(center=center,
                                       radius_min=(diagonal_len * 2.8) / 2,
                                       radius_max=(diagonal_len * 2.8) / 2,
                                       elevation_min=35.0,
                                       elevation_max=35.1,
                                       azimuth_min=-180 + (degree_increment * poses),
                                       azimuth_max=-180 + (degree_increment * poses) + 0.1,
                                       uniform_volume=False)
        poi = bproc.object.compute_poi(loaded_objects) #+ np.random.uniform([-1, -1, -1], [1, 1, 1])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location) #3,3 - 3
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        if True:
            bproc.camera.add_camera_pose(cam2world_matrix)
            l_cam2world_matrix = cam2world_matrix.tolist()
            visible_objs = bproc.camera.visible_objects(cam2world_matrix)
            visible_objs_list = []

            for elem in visible_objs:
                if not ("floor" in elem.get_name().lower()):
                    dict_elem = {
                        "location": elem.get_location().tolist(),
                    }
                    visible_objs_list.append(dict_elem)
            dict_curr = {
                "pose": l_cam2world_matrix,
                "visible_objs_list": visible_objs_list,
                "center": center.tolist(),
                "K": (bproc.camera.get_intrinsics_as_K_matrix()).tolist(),
                "objname_list": objname_list,
                "bboxes_list": bboxes_list,
                "vertices_list": vertices_list,
            }

            dict_all[poses] = dict_curr
            poses += 1
        tries+=1


    # render the whole pipeline
    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
    data = bproc.renderer.render()

    # write the data to a .hdf5 container
    front_id = args.front_id + "_" + args.scene_id
    bproc.writer.write_hdf5(args.hdf_output_dir, data, front_id)

    # write the pose data to a .json file
    output_path = os.path.join(args.scene_output_dir_folder, front_id + ".json")
    with open(output_path, "w") as outfile:
        json.dump(dict_all, outfile)

    if not os.path.exists(os.path.join(args.scene_output_dir_folder, "segmentation_raw")):
        os.mkdir(os.path.join(args.scene_output_dir_folder, "segmentation_raw"))
    for i in range(0, len(data['category_id_segmaps'])):
        segmap = data['category_id_segmaps'][i].astype(np.int32)
        segmap = Image.fromarray(segmap, mode="I")
        segmap.save(os.path.join(args.scene_output_dir_folder, "segmentation_raw", "{}.png".format(i)))


main()