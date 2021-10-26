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
## episode 2
Jack = [36, 55,58,76,114, 193, 326, 535, 558,561,578, 580, 594,595,599, 600, 620, 616, 694, 716, 774, 765, 766, 773,1169, 1187, 1207, 1231, 1258, 1342, 1345, 1346, 1348, 1355, 1383, 1423, 1716, 1729, 2201, 2351, 2354, 2477, 2528, 2534, 2585, 2463, 2927, 3153, 3234, 3813,3849, 3887, 3899, 3909, 3911, 3917, 3956, 4194,4364,4785,4807,4827,4956,4976,5347,5508,5608,5821,6005,6030,6673,6739,6770,6773,6814,6815,6858,7240,7313,7824,8634,10974,6673, 10974, 12085, 12091, 14594, 15836, 19325, 21153, 21756, 25573, 40117, 50977, 61837]
Bill = [216, 229,231, 266, 309,407, 551, 782, 992, 1322, 3387,3415, 3420, 3429, 3446, 3583, 3620, 3656, 3851, 3853, 4107,4880,4922,4942,7023, 7049,7074,7717,10627, 12096, 12120, 12748,  13859, 13913, 13948, 14524, 14743, 18136, 19104, 19306, 21953, 28339, 43737  ]
Chloe = [167, 170, 178,233, 1260, 1310, 1312, 3463, 3466,3495, 3786, 3837,7050, 7085 ,8500,]
Terro = [54, 69, 79, 347, 784, 1040, 1042, 1084, 2675, 2731, 3706,4740, 14200, 25376 ]

group_list_all = [Jack, Bill,Chloe, Terro]
patch_dir = "/media/yipeng/Sandisk/movie_tracking/tracking/pipeline/group_result_network"
patch_dir_2 = "/media/yipeng/Sandisk/movie_tracking/tracking/pipeline/filter_tracking_result"
vis_folder = "/media/yipeng/data/movie/Movie_Analysis/draft_result/visualize_episode2"
res = {}
for i in range(4):
    character_size_frame = {} #key frame value size
    for character_label in group_list_all[i]:
        one_label_folder = os.path.join(patch_dir, str(character_label))
        if not os.path.exists(one_label_folder):
            print(i,character_label)
            one_label_folder = os.path.join(patch_dir_2, str(character_label))
            if not os.path.exists(one_label_folder):
                continue
        one_label_patches_fns = os.listdir(one_label_folder)
        for patch_fn in one_label_patches_fns:
            frame_number = int(patch_fn.split("_")[1])
            patch_fn_full = os.path.join(one_label_folder,patch_fn)
            im = cv2.imread(patch_fn_full)
            h, w, c = im.shape
            character_size_frame[frame_number] = h*w
            shutil.copy(patch_fn_full, os.path.join(vis_folder, str(i),patch_fn))
    res[i] = character_size_frame
dump_pickle("draft_result/episode2_character_size.pkl", res)
