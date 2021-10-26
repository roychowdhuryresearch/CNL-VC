import sys 
sys.path.append(".")
import shutil
import os
import pickle
import cv2
sys.path.insert(0, "/media/yipeng/data/movie/Movie_Analysis")
def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        #print(fn)
    return lookup

patch_dir = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/resnet_result_all_shift_1_3"
res = {}
for i in range(4):
    character_size_frame = {} #key frame value size
    one_label_folder = os.path.join(patch_dir, str(i)) 
    one_label_patches_fns = os.listdir(one_label_folder)
    for patch_fn in one_label_patches_fns:
        frame_number = int(patch_fn.split("_")[1])
        patch_fn_full = os.path.join(one_label_folder,patch_fn)
        im = cv2.imread(patch_fn_full)
        h, w, c = im.shape
        character_size_frame[frame_number] = h*w
    res[i] = character_size_frame
dump_pickle("draft_result/episode1_character_size.pkl", res)
