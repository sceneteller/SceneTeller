import numpy as np
import os
import json
import argparse


# PREPROCESS

def checkOverlapandOOB(args, id):

    root = args.dataset_dir
    render_type = args.room

    path_stats = os.path.join(root, render_type, "dataset_stats.txt")
    with open(path_stats, "r") as file:
        stats = json.load(file)
    path = os.path.join(root, render_type, id, "boxes.npz")
    data = (np.load(path))

    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset = min(data['floor_plan_vertices'][:, 0])
    y_offset = min(data['floor_plan_vertices'][:, 2])
    room_length = max(data['floor_plan_vertices'][:, 0]) - min(data['floor_plan_vertices'][:, 0])
    room_width = max(data['floor_plan_vertices'][:, 2]) - min(data['floor_plan_vertices'][:, 2])

    objects = []

    angles = data['angles']
    flag_OOB = False
    flag_overlap = False
    for i in range(0, len(angles)):
        labels = data['class_labels'][i]
        label_idx = np.where(labels)[0][0]
        if label_idx >= len(stats['object_types']):  # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        length, height, width = data['sizes'][i]
        length, height, width = length * 2, height * 2, width * 2
        orientation = round(angles[i][0] / 3.1415926 * 180)
        dx, dz, dy = data['translations'][i]
        dx = dx + x_c - x_offset
        dy = dy + y_c - y_offset

        if (orientation != 180) and (orientation != -180) and (orientation != 0):
            dx1 = dx - (width / 2)
            dy1 = dy - (length / 2)
            dx2 = dx + (width / 2)
            dy2 = dy + (length / 2)
        else:
            dx1 = dx - (length / 2)
            dy1 = dy - (width / 2)
            dx2 = dx + (length / 2)
            dy2 = dy + (width / 2)

        if (dx1 < 0) or (dy1 < 0) or (dx2 > float(room_length)) or (dy2 > float(room_width)):
            flag_OOB = True
            # Find overlapping area
            area_a = float(room_length) * float(room_width)

            x_dist_b = np.abs(dx1 - dx2)
            y_dist_b = np.abs(dy1 - dy2)
            area_b = x_dist_b * y_dist_b

            x_distance = min(dx2, room_length) - max(dx1, 0)
            y_distance = min(dy2, room_width) - max(dy1, 0)
            intersection = x_distance * y_distance
            excess = area_a + area_b - intersection - area_a
            union = area_a + area_b - intersection
            percent_int = (excess * 100) / union
            if percent_int < 3:
                flag_OOB = False

        if (dx < 0) or (dy < 0) or (dx > float(room_length)) or (dy > float(room_width)):
            flag_OOB = True

        if flag_OOB == True:
            return flag_overlap, flag_OOB

        obj = [dx1,dy1,dx2,dy2,cat]
        objects.append(obj)

    for i in range(0, len(objects)):
        for j in range(0, len(objects)):
            if i == j:
                continue
            a0 = objects[i][0]
            a1 = objects[i][1]
            a2 = objects[i][2]
            a3 = objects[i][3]
            a_cat = objects[i][4]

            b0 = objects[j][0]
            b1 = objects[j][1]
            b2 = objects[j][2]
            b3 = objects[j][3]
            b_cat = objects[j][4]

            if ("pendant" in a_cat) or ("pendant" in b_cat) or ("ceiling" in a_cat) or ("ceiling" in b_cat):
                pass
            else:
                if (a0 >= b2) or (a2 <= b0) or (a3 <= b1) or (a1 >= b3):
                    flag_overlap = False
                else:
                    flag_overlap = True
                    # Find overlapping area
                    x_dist_a = np.abs(a0-a2)
                    y_dist_a = np.abs(a1-a3)
                    area_a = x_dist_a * y_dist_a

                    x_dist_b = np.abs(b0 - b2)
                    y_dist_b = np.abs(b1 - b3)
                    area_b = x_dist_b * y_dist_b

                    x_distance = min(a2,b2) - max(a0,b0)
                    y_distance = min(a3,b3) - max(a1,b1)
                    intersection = x_distance * y_distance
                    union = area_a + area_b - intersection
                    percent_int = (intersection*100)/union
                    if percent_int < 10:
                        flag_overlap = False
                    return flag_overlap, flag_OOB


    return flag_overlap, flag_OOB

def generateSplits(args): 

    root = os.path.join(".", "LayoutGen/dataset")
    split_name = "splits-orig"
    data = json.load(open(os.path.join(root, split_name, args.room + "_splits.json")))

    rect_train = data['rect_train']
    rect_val = data['rect_val']
    rect_test = data['rect_test']
    train = data['train']
    val = data['val']
    test = data['test']

    rect_train_new = []
    rect_val_new = []
    rect_test_new = []
    train_new = []
    val_new = []
    test_new = []

    count = 0
    for i in range(0, len(rect_train)):
        id = rect_train[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True):
            count+=1
        else:
            rect_train_new.append(id)
    print("{}/{} overlaps in rect_train".format(count,len(rect_train)))

    count = 0
    for i in range(0, len(rect_val)):
        id = rect_val[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True) :
            count += 1
        else:
            rect_val_new.append(id)
    print("{}/{} overlaps in rect_val".format(count,len(rect_val)))

    count = 0
    for i in range(0, len(rect_test)):
        id = rect_test[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True):
            count += 1
        else:
            rect_test_new.append(id)
    print("{}/{} overlaps in rect_test".format(count,len(rect_test)))

    count = 0
    for i in range(0, len(train)):
        id = train[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True):
            count += 1
        else:
            train_new.append(id)
    print("{}/{} overlaps in train".format(count,len(train)))

    count = 0
    for i in range(0, len(val)):
        id = val[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True):
            count += 1
        else:
            val_new.append(id)
    print("{}/{} overlaps in val".format(count,len(val)))

    count = 0
    for i in range(0, len(test)):
        id = test[i]
        overlap, OOB = checkOverlapandOOB(args, id)
        if (overlap==True) or (OOB == True) :
            count += 1
        else:
            test_new.append(id)
    print("{}/{} overlaps in test".format(count,len(test)))

    dict = {
        "train": train_new,
        "val": val_new,
        "test": test_new,
        "rect_train": rect_train_new,
        "rect_val": rect_val_new,
        "rect_test": rect_test_new
    }
    new_split_name = "splits-preprocessed"
    if not os.path.exists(os.path.join(root,new_split_name)):
        os.mkdir(os.path.join(root,new_split_name))
    output_path = os.path.join(root,new_split_name, args.room + "_splits.json")
    with open(output_path, "w") as outfile:
        json.dump(dict, outfile)

def describeOrigBoxes(args, id):

    root = args.dataset_dir
    render_type = args.room

    path_stats = os.path.join(root,render_type, "dataset_stats.txt")
    with open(path_stats, "r") as file:
        stats = json.load(file)

    path = os.path.join(root,render_type,id,"boxes.npz")
    data = (np.load(path))
    x_c, y_c = data['floor_plan_centroid'][0], data['floor_plan_centroid'][2]
    x_offset = min(data['floor_plan_vertices'][:, 0])
    y_offset = min(data['floor_plan_vertices'][:, 2])
    room_length = max(data['floor_plan_vertices'][:, 0]) - min(data['floor_plan_vertices'][:, 0])
    room_width = max(data['floor_plan_vertices'][:, 2]) - min(data['floor_plan_vertices'][:, 2])

    r0 = [0, 0, float(room_length / 3),float(room_width / 3)]
    r1 = [0, float(room_width / 3), float(room_length / 3),2*float(room_width / 3)]
    r2 = [0, 2 * float(room_width / 3), float(room_length / 3), float(room_width)]
    r3 = [float(room_length / 3), 0, 2 * float(room_length / 3), float(room_width / 3)]
    r4 = [float(room_length / 3), float(room_width / 3), 2 * float(room_length / 3), 2 * float(room_width / 3)]
    r5 = [float(room_length / 3), 2 * float(room_width / 3), 2 * float(room_length / 3), float(room_width)]
    r6 = [2 * float(room_length / 3), 0, float(room_length), float(room_width / 3)]
    r7 = [2 * float(room_length / 3), float(room_width / 3) , float(room_length), 2 * float(room_width / 3)]
    r8 = [2 * float(room_length / 3), 2 * float(room_width / 3) , float(room_length), float(room_width)]
    rects = [r0,r1,r2,r3,r4,r5,r6,r7,r8]
  
    angles = data['angles']
    rule = ""

    for i in range(0, len(angles)):
        labels = data['class_labels'][i]
        label_idx = np.where(labels)[0][0]
        if label_idx >= len(stats['object_types']):  # NOTE:
            continue
        cat = stats['object_types'][label_idx]
        if ("pendant" in cat) or ("ceiling" in cat):
            continue
        length, height, width = data['sizes'][i]
        length, height, width = length*2, height*2, width*2
        orientation = round(angles[i][0] / 3.1415926 * 180)
        dx, dz, dy = data['translations'][i]
        dx = dx + x_c - x_offset
        dy = dy + y_c - y_offset

        rect_id = -1
        addition = ""
        preposition = "at"
        for r in range(0, len(rects)):
            if (dx > rects[r][0]) and (dx <= rects[r][2]) and (dy > rects[r][1]) and (dy <= rects[r][3]):
                rect_id = r
                if rect_id == 1:
                    if np.abs(rects[2][1] - dy) < np.abs(rects[0][3] - dy):
                        addition = ", slightly closer to the upper wall"
                    elif np.abs(rects[2][1] - dy) > np.abs(rects[0][3] - dy):
                        addition = ", slightly closer to the lower wall"
                elif rect_id == 7:
                    if np.abs(rects[8][1] - dy) < np.abs(rects[6][3] - dy):
                        addition = ", slightly closer to the upper wall"
                    elif np.abs(rects[8][1] - dy) > np.abs(rects[6][3] - dy):
                        addition = ", slightly closer to the lower wall"
                elif rect_id == 3:
                    if np.abs(rects[0][2] - dx) < np.abs(rects[6][0] - dx):
                        addition = ", slightly closer to the left wall"
                    elif np.abs(rects[0][2] - dx) > np.abs(rects[6][0] - dx):
                        addition = ", slightly closer to the right wall"
                elif rect_id == 5:
                    if np.abs(rects[2][2] - dx) < np.abs(rects[8][0] - dx):
                        addition = ", slightly closer to the left wall"
                    elif np.abs(rects[2][2] - dx) > np.abs(rects[8][0] - dx):
                        addition = ", slightly closer to the right wall"
                elif rect_id == 4:
                    if np.abs(rects[1][2] - dx) < np.abs(rects[7][0] - dx):
                        addition = ", slightly closer to the left wall"
                        if np.abs(rects[5][1] - dy) < np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the upper left wall"
                        elif np.abs(rects[5][1] - dy) > np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the lower left wall"
                    elif np.abs(rects[1][2] - dx) > np.abs(rects[7][0] - dx):
                        addition = ", slightly closer to the right wall"
                        if np.abs(rects[5][1] - dy) < np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the upper right wall"
                        elif np.abs(rects[5][1] - dy) > np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the lower right wall"
                    else:
                        if np.abs(rects[5][1] - dy) < np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the upper wall"
                        elif np.abs(rects[5][1] - dy) > np.abs(rects[3][3] - dy):
                            addition = ", slightly closer to the lower wall"


                if (rect_id == 0):
                    if (orientation != 180) and (orientation != -180) and (orientation != 0):
                        lx = dx - (width / 2)
                        ly = dy - (length / 2)
                    else:
                        lx = dx - (length / 2)
                        ly = dy - (width / 2)
                    if (np.abs(rects[rect_id][0] - lx) < float(room_length / 3) * 0.2) and (np.abs(rects[rect_id][1] - ly) < float(room_width / 3) * 0.2):
                        preposition = "at"
                    else:
                        preposition = "near"
                elif (rect_id == 2):
                    if (orientation != 180) and (orientation != -180) and (orientation != 0):
                        lx = dx - (width / 2)
                        ry = dy + (length / 2)
                    else:
                        lx = dx - (length / 2)
                        ry = dy + (width / 2)
                    if (np.abs(rects[rect_id][0] - lx) < float(room_length / 3) * 0.2) and (np.abs(rects[rect_id][3] - ry) < float(room_width / 3) * 0.2):
                        preposition = "at"
                    else:
                        preposition = "near"
                elif (rect_id == 6):
                    if (orientation != 180) and (orientation != -180) and (orientation != 0):
                        rx = dx + (width / 2)
                        ly = dy - (length / 2)
                    else:
                        rx = dx + (length / 2)
                        ly = dy - (width / 2)
                    if (np.abs(rects[rect_id][2] - rx) < float(room_length / 3) * 0.2) and (np.abs(rects[rect_id][1] - ly) < float(room_width / 3) * 0.2):
                        preposition = "at"
                    else:
                        preposition = "near"
                elif (rect_id == 8):
                    if (orientation != 180) and (orientation != -180) and (orientation != 0):
                        rx = dx + (width / 2)
                        ry = dy + (length / 2)
                    else:
                        rx = dx + (length / 2)
                        ry = dy + (width / 2)
                    if (np.abs(rects[rect_id][2] - rx) < float(room_length / 3) * 0.2) and (np.abs(rects[rect_id][3] - ry) < float(room_width / 3) * 0.2):
                        preposition = "at"
                    else:
                        preposition = "near"
                break

        obj_rule = None
        if rect_id == 0:
            obj_rule = "A {} is located {} the lower left corner".format(cat,preposition) + addition
        elif rect_id == 1:
            obj_rule = "A {} is located {} the center of the left wall".format(cat,preposition) + addition
        elif rect_id == 2:
            obj_rule = "A {} is located {} the upper left corner".format(cat,preposition) + addition
        elif rect_id == 3:
            obj_rule = "A {} is located {} the center of the lower wall".format(cat,preposition) + addition
        elif rect_id == 4:
            obj_rule = "A {} is located {} the center of the room".format(cat,preposition) + addition
        elif rect_id == 5:
            obj_rule = "A {} is located {} the center of the upper wall".format(cat,preposition) + addition
        elif rect_id == 6:
            obj_rule = "A {} is located {} the lower right corner".format(cat,preposition) + addition
        elif rect_id == 7:
            obj_rule = "A {} is located {} the center of the right wall".format(cat,preposition) + addition
        elif rect_id == 8:
            obj_rule = "A {} is located {} the upper right corner".format(cat,preposition) + addition

        if obj_rule == None:
            print("{}  {}  {}  {}".format(rect_id, dx, dy, id))
            

        if ((orientation <= 10) and (orientation >= -10)):
            obj_rule = obj_rule + ", oriented with no rotation. "
        elif ((orientation >= 170) and (orientation <= 190)) or ((orientation <= -170) and (orientation >= -190)):
            obj_rule = obj_rule + ", oriented backwards. "
        elif ((orientation >= 80) and (orientation <= 100)):
            obj_rule = obj_rule + ", oriented perpendicularly. "
        elif ((orientation <= -80) and (orientation >= -100)):
            obj_rule = obj_rule + ", oriented perpendicularly in the opposite direction. "


        elif ((orientation > 10) and (orientation < 80)):
            obj_rule = obj_rule + ", tilted at an angle almost halfway between horizontal and vertical. "
        elif ((orientation < -10) and (orientation > -80)):
            obj_rule = obj_rule + ", tilted at an angle almost halfway between horizontal and vertical in the opposite direction. "
        elif ((orientation > 100) and (orientation < 170)):
            obj_rule = obj_rule + ", tilted at an angle that lies between a right angle and a full rotation. "
        elif ((orientation < -100) and (orientation > -170)):
            obj_rule = obj_rule + ", tilted at an angle that lies between a right angle and a full rotation in the opposite direction. "
        else:
            print(orientation)

        rule = rule + obj_rule

    return rule

def generateSplitPreprompts(args):

    root = os.path.join(".", "LayoutGen/dataset")
    split_name = "splits-preprocessed"
    data = json.load(open(os.path.join(root, split_name, args.room + "_splits.json")))

    rect_train = data['rect_train']
    rect_val = data['rect_val']
    rect_test = data['rect_test']

    dict = {}
    for i in range(0, len(rect_train)):
        id = rect_train[i]
        dict[id] = describeOrigBoxes(args, id)
    for i in range(0, len(rect_val)):
        id = rect_val[i]
        dict[id] = describeOrigBoxes(args, id)
    for i in range(0, len(rect_test)):
        id = rect_test[i]
        dict[id] = describeOrigBoxes(args, id)

    output_path = os.path.join(root, split_name, args.room + "_splits_preprompts.json")
    with open(output_path, "w") as outfile:
        json.dump(dict, outfile)




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog='SceneTeller',
                                 description='Data preprocessing arguments')
    parser.add_argument('--room', type=str, default='livingroom', choices=['bedroom', 'livingroom'])
    parser.add_argument('--dataset_dir', type=str, default='./scene_data/data_output')  
    args = parser.parse_args()

    generateSplits(args)
    generateSplitPreprompts(args)


   











