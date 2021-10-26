import os
from utilities import *
class ConfusedGroupFilter:
    def __init__(self, confused_group_dir):
        self.confused_group_dir = confused_group_dir
        self.group_dir = os.listdir(confused_group_dir)
        self.good_set = set()
        self.scene_dict = {}
    def filter_by_size(self):
        for g in self.group_dir:
            fn_list = os.listdir(os.path.join(self.confused_group_dir , g))
            if len (fn_list) < 5:
                continue 
            self.good_set.add(g)
    
    def filter_tracking(self,scene_list_fn):
        self.scene_dict:{} = read_scene_list("./scene_list.txt")
        saved_dir = "/media/yipeng/Sandisk/yolov3/yolov3/confused_tracking_subgroup/"
        clean_folder(saved_dir)
        for g in self.good_set:
            fn_list = os.listdir(os.path.join(self.confused_group_dir , g)) 
            sorted_fn = self.sort_filename(fn_list) 
            start = int(sorted_fn[0].split("_")[1])
            end = int(sorted_fn[-1].split("_")[1])
            res = self.find_divider(start, end)
            for s in sorted_fn:
                idx = int(s[0].split("_")[1])
                key = list(sorted(res.keys()))
                for k in range(len(key)):
                    if idx >= key[k] and idx <= res[key[k]]:
                        sub_folder = os.path.join(saved_dir, g+"_"+str(k))
                        os.mkdir(sub_folder)
                        filename = os.path.join(sub_folder, s)
                        original_fn = os.path.join(self.confused_group_dir,g, s)
                        shutil.copy(original_fn, filename)




    def find_divider(self, start, end):
        res = {}
        key_list = sorted(self.scene_dict.keys())
        flag = False
        for i in range(0, len(key_list)-1):
            if key_list[i] <= start and key_list[i+1] > start:
                flag = True
            if flag: 
                res[i] = key_list[i]
            if key_list[i] <= end and key_list[i+1] > end:
                res[i] = key_list[i]
                return res

            
    
    def sort_filename(self, frame_list: list):
    ## sort the filename by \u201cframeXX\u201d
        frame_dir = {}
        for f in frame_list:
            frame_number = int(f.split("_")[1])
            frame_dir[frame_number] = f
        frame_dir_key = list(frame_dir.keys())
        frame_dir_key.sort()
        sorted_filename = []
        for frame_n in frame_dir_key:
            sorted_filename.append(frame_dir[frame_n])
        return sorted_filename