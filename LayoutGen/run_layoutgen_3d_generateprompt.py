import os
import os.path as op
import json
import pdb
import numpy as np
from tqdm import tqdm
import time
import argparse
import openai
from utils import *

from transformers import GPT2TokenizerFast

openai.organization = ""
openai.api_key = ""
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

parser = argparse.ArgumentParser(prog='Scene description generation',
                                 description='Use GPT to enhance rule-based descriptions')
parser.add_argument('--room', type=str, default='bedroom', choices=['bedroom', 'livingroom'])
parser.add_argument('--dataset_dir', type=str)
parser.add_argument('--split', type=str, default='splits-preprocessed')
parser.add_argument('--gpt_type', type=str, default='gpt4', choices=['gpt3.5-chat', 'gpt4'])
parser.add_argument('--base_output_dir', type=str, default='./llm_output/bedroom-train/')
parser.add_argument('--gpt_input_length_limit', type=int, default=7000)
parser.add_argument('--unit', type=str, choices=['px', 'm', ''], default='px')
parser.add_argument("--n_iter", type=int, default=1)
parser.add_argument("--generate_train", action='store_true')
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument("--suffix", type=str, default="")
parser.add_argument("--normalize", action='store_true')
parser.add_argument("--regular_floor_plan", action='store_true')
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

# GPT Type
gpt_name = {
    'gpt3.5-chat': 'gpt-3.5-turbo',
    'gpt4': 'gpt-4',
}

def load_room_boxes(prefix, id, stats, unit):
    data = np.load(op.join(prefix, id, 'boxes.npz'))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset = min(data['floor_plan_vertices'][:, 0])
    y_offset = min(data['floor_plan_vertices'][:, 2])
    room_length = max(data['floor_plan_vertices'][:, 0]) - min(data['floor_plan_vertices'][:, 0])
    room_width = max(data['floor_plan_vertices'][:, 2]) - min(data['floor_plan_vertices'][:, 2])
    vertices = np.stack((data['floor_plan_vertices'][:, 0] - x_offset, data['floor_plan_vertices'][:, 2] - y_offset),axis=1)
    vertices = np.asarray([list(nxy) for nxy in set(tuple(xy) for xy in vertices)])

    # normalize
    if args.normalize:
        norm = min(room_length, room_width)
        room_length, room_width = room_length / norm, room_width / norm
        vertices /= norm
        if unit in ['px', '']:
            scale_factor = 256
            room_length, room_width = int(room_length * scale_factor), int(room_width * scale_factor)

    vertices = [f'({v[0]:.2f}, {v[1]:.2f})' for v in vertices]

    if unit in ['px', '']:
        condition = f"Condition:\n"
        if args.room == 'livingroom':
            if 'dining' in id.lower():
                condition += f"Room Type: living room & dining room\n"
            else:
                condition += f"Room Type: living room\n"
        else:
            condition += f"Room Type: {args.room}\n"
        condition += f"Room Size: max length {room_length}{unit}, max width {room_width}{unit}\n"
    else:
        condition = f"Condition:\n" \
                    f"Room Type: {args.room}\n" \
                    f"Room Size: max length {room_length:.2f}{unit}, max width {room_width:.2f}{unit}\n"

    layout = 'Layout:\n'
    for label, size, angle, loc in zip(data['class_labels'], data['sizes'], data['angles'], data['translations']):
        label_idx = np.where(label)[0][0]
        if label_idx >= len(stats['object_types']):  # NOTE:
            continue
        cat = stats['object_types'][label_idx]

        length, height, width = size  # NOTE: half the actual size
        length, height, width = length * 2, height * 2, width * 2
        orientation = round(angle[0] / 3.1415926 * 180)
        dx, dz, dy = loc  # NOTE: center point
        dx = dx + x_c - x_offset
        dy = dy + y_c - y_offset

        # normalize
        if args.normalize:
            length, width, height = length / norm, width / norm, height / norm
            dx, dy, dz = dx / norm, dy / norm, dz / norm
            if unit in ['px', '']:
                length, width, height = int(length * scale_factor), int(width * scale_factor), int(
                    height * scale_factor)
                dx, dy, dz = int(dx * scale_factor), int(dy * scale_factor), int(dz * scale_factor)

        if unit in ['px', '']:
            layout += f"{cat} {{length: {length}{unit}; " \
                      f"width: {width}{unit}; " \
                      f"height: {height}{unit}; " \
                      f"left: {dx}{unit}; " \
                      f"top: {dy}{unit}; " \
                      f"depth: {dz}{unit};" \
                      f"orientation: {orientation} degrees;}}\n"
        else:
            layout += f"{cat} {{length: {length:.2f}{unit}; " \
                      f"height: {height:.2f}{unit}; " \
                      f"width: {width:.2f}{unit}; " \
                      f"orientation: {orientation} degrees; " \
                      f"left: {dx:.2f}{unit}; " \
                      f"top: {dy:.2f}{unit}; " \
                      f"depth: {dz:.2f}{unit};}}\n"

    return condition, layout, dict(data)


def load_set(prefix, ids, stats, unit):
    id2prompt = {}
    meta_data = {}
    for id in tqdm(ids):
        condition, layout, data = load_room_boxes(prefix, id, stats, unit)
        id2prompt[id] = [condition, layout]
        meta_data[id] = data
    return id2prompt, meta_data



def create_prompt(sample):
    return sample[0] + sample[1] + "\n\n"


def form_prompt_for_chatgpt(preprompt):

    message_list = []
    rtn_prompt = 'You are a 3D indoor scene layout textual descriptor generator. \nInstruction: Given a text prompt describing the 3D layout of an indoor scene, ' \
                 'provide a variation of the given text prompt. Do not change the position, orientation and names of objects. Focus on keeping the geometry and locations the same. \n\n' \

    message_list.append({'role': 'system', 'content': rtn_prompt})

    current_messages = [
        {'role': 'user', 'content': preprompt + "\nTextual description generated:\n"},
    ]
    message_list = message_list + current_messages

    return message_list


def _main(args):

    dataset_prefix = f"{args.dataset_dir}"
    with open(f"dataset/{args.split}/{args.room}_splits.json", "r") as file:
        splits = json.load(file)

    with open(f"dataset/{args.split}/{args.room}_splits_preprompts.json", "r") as file:
        preprompts = json.load(file)

    with open(f"{args.dataset_dir}/dataset_stats.txt", "r") as file:
        stats = json.load(file)

    if args.regular_floor_plan:
        args.suffix += '_regular'

    # check if have been processed
    args.output_dir = args.base_output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # load train examples
    train_ids = splits['rect_train'] if args.regular_floor_plan else splits['train']
    train_data, meta_train_data = load_set(dataset_prefix, train_ids, stats,args.unit)  # train_data =[condition,layout], meta_train_data=?

    # load val examples
    val_ids = splits['rect_test'] if args.regular_floor_plan else splits['test']
    val_data, meta_val_data = load_set(dataset_prefix, val_ids, stats, args.unit)
    print(f"Loaded {len(train_data)} train samples and {len(val_data)} validation samples")

    # GPT prediction process
    args.gpt_name = gpt_name[args.gpt_type]

    c = 0
    if args.generate_train:
        for train_id, train_example in tqdm(train_data.items(), total=len(train_data), desc='gpt3'):

            if os.path.exists(os.path.join("{}tmp/gpt4".format(args.output_dir), "{}.json".format(train_id))):
                continue
            # predict
            while True:
                if args.gpt_type in ['gpt3.5-chat', 'gpt4']:
                    prompt_for_gpt3 = form_prompt_for_chatgpt(
                        preprompt=preprompts[train_id])
                else:
                    raise NotImplementedError

                if args.verbose:
                    print(train_id)
                    print(prompt_for_gpt3)
                    print('\n' + '-' * 30)
                    pdb.set_trace()

                try:
                    if args.gpt_type in ['gpt3.5', 'gpt4']:
                        response = openai.ChatCompletion.create(
                            model=args.gpt_name,
                            messages=prompt_for_gpt3,
                            temperature=0.7,
                            max_tokens=1024 if args.room == 'livingroom' else 1024,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            stop="Condition:",
                            n=args.n_iter,
                        )
                    else:
                        raise NotImplementedError
                    break
                except openai.error.ServiceUnavailableError:
                    print('OpenAI ServiceUnavailableError.\tWill try again in 5 seconds.')
                    time.sleep(5)
                except openai.error.RateLimitError:
                    print('OpenAI RateLimitError.\tWill try again in 5 seconds.')
                    time.sleep(5)
                except openai.error.InvalidRequestError as e:
                    print(e)
                except openai.error.APIError as e:
                    print('OpenAI Bad Gateway Error.\tWill try again in 5 seconds.')
                    time.sleep(5)

            os.makedirs(op.join(args.output_dir, 'tmp', args.gpt_type), exist_ok=True)
            response['prompt'] = prompt_for_gpt3
            write_json(op.join(args.output_dir, 'tmp', args.gpt_type, f"{train_id}.json"), response)

            for i_iter in range(args.n_iter):
                # parse output
                if args.verbose:
                    try:
                        print(response['choices'][i_iter]['text'])
                    except:
                        print(response['choices'][i_iter]['message']['content'])

                if args.gpt_type in ['gpt4']:
                    time.sleep(5)
                print("{}:{}".format(c, train_id))
            c += 1

    else:
        for val_id, val_example in tqdm(val_data.items(), total=len(val_data), desc='gpt3'):

            if os.path.exists(os.path.join("{}tmp/gpt4".format(args.output_dir), "{}.json".format(val_id))):
                continue
            # predict
            while True:
                if args.gpt_type in ['gpt3.5-chat', 'gpt4']:
                    prompt_for_gpt3 = form_prompt_for_chatgpt(
                        preprompt=preprompts[val_id])
                else:
                    raise NotImplementedError

                if args.verbose:
                    print(val_id)
                    print(prompt_for_gpt3)
                    print('\n' + '-' * 30)
                    pdb.set_trace()

                try:
                    if args.gpt_type in ['gpt3.5', 'gpt4']:
                        response = openai.ChatCompletion.create(
                            model=args.gpt_name,
                            messages=prompt_for_gpt3,
                            temperature=0.7,
                            max_tokens=1024 if args.room == 'livingroom' else 1024,
                            top_p=1.0,
                            frequency_penalty=0.0,
                            presence_penalty=0.0,
                            stop="Condition:",
                            n=args.n_iter,
                        )
                    else:
                        raise NotImplementedError
                    break
                except openai.error.ServiceUnavailableError:
                    print('OpenAI ServiceUnavailableError.\tWill try again in 5 seconds.')
                    time.sleep(5)
                except openai.error.RateLimitError:
                    print('OpenAI RateLimitError.\tWill try again in 5 seconds.')
                    time.sleep(5)
                except openai.error.InvalidRequestError as e:
                    print(e)
                except openai.error.APIError as e:
                    print('OpenAI Bad Gateway Error.\tWill try again in 5 seconds.')
                    time.sleep(5)

            os.makedirs(op.join(args.output_dir, 'tmp', args.gpt_type), exist_ok=True)
            response['prompt'] = prompt_for_gpt3
            write_json(op.join(args.output_dir, 'tmp', args.gpt_type, f"{val_id}.json"), response)

            for i_iter in range(args.n_iter):
                # parse output
                if args.verbose:
                    try:
                        print(response['choices'][i_iter]['text'])
                    except:
                        print(response['choices'][i_iter]['message']['content'])

                if args.gpt_type in ['gpt4']:
                    time.sleep(5)
                print("{}:{}".format(c, val_id))
            c += 1


if __name__ == '__main__':
    _main(args)
