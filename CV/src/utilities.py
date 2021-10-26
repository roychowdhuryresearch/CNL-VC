import cv2
import random
import os
import shutil
import pickle
import numpy as np
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        os.mkdir(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        print(fn)
    return lookup

def reverse_key_list_dict(list_dict):
    res = {}
    for k in list_dict.keys():
        values = list_dict[k]
        for v in values:
            res[v] = k
    return res



def sort_filename(frame_list: list):
    ## sort the filename by \u201cframeXX\u201d
    frame_dir = {}
    for f in frame_list:
        frame_number = int(f.split("frame_")[1].split(".")[0])
        frame_dir[frame_number] = f
    frame_dir_key = list(frame_dir.keys())
    frame_dir_key.sort()
    sorted_filename = []
    for frame_n in frame_dir_key:
        sorted_filename.append(frame_dir[frame_n])
    return sorted_filename
    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] +20, c1[1] - t_size[1] - 3 +20
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + 20), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def read_scene_list(scene_list_fn):
    res = {}
    with open(scene_list_fn) as f:
        lines = f.readlines()
    for line in lines:
        line_s = line.strip().split("|")
        if len(line_s) < 5:
            continue
        start = line_s[2].strip()
        end = line_s[4].strip()
        if start.isdigit() and end.isdigit():
            res[int(start)] = int(end)
    return res 

def join( a, b):
    return os.path.join(a,b)