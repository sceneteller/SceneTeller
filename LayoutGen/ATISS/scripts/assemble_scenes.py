import os
import argparse

def renderFromFiles(args):


    llm_output_folder = args.model_output
    llm_outputs = os.listdir(llm_output_folder)
    llm_outputs.sort()

    config = args.config
    future_pkl = args.path_to_pickled_3d_futute_models
    front_texture = args.path_to_floor_plan_textures
    vis_out_folder = args.output_directory

    if not os.path.exists("../../assemble_output/{}".format(vis_out_folder)):
        os.mkdir("../../assemble_output/{}".format(vis_out_folder))

    for i in range(0, len(llm_outputs)):
        llm_output = llm_outputs[i]
        id = llm_output[:-5]

        if not os.path.exists("../../assemble_output/{}/{}".format(vis_out_folder,id)):
            os.mkdir("../../assemble_output/{}/{}".format(vis_out_folder,id))

        command = "python render_from_files.py {} ../../assemble_output/{}/{} {} {} {}/{} --room {} --split test_regular --export_scene".format(config, vis_out_folder, id, future_pkl, front_texture, llm_output_folder, llm_output, args.room)
        os.system(command)
        print("############### {} scenes assembled ###############".format(i+1))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='SceneTeller',
                                 description='Assemble 3D scenes from 3D layouts.')
    parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom', 'livingroom'])
    parser.add_argument('--config', type=str, default='../config/bedrooms_eval_config.yaml')  
    parser.add_argument('--output_directory', type=str, default='bedroom-test')  
    parser.add_argument('--path_to_pickled_3d_futute_models', type=str, default='../../../scene_data/data_output_future/threed_future_model_bedroom.pkl')  
    parser.add_argument('--path_to_floor_plan_textures', type=str, default='../../../scene_data/3D-FRONT-texture')  
    parser.add_argument('--model_output', type=str, default='../../llm_output/bedroom-test/tmp/gpt4')  

    args = parser.parse_args()
    renderFromFiles(args)
