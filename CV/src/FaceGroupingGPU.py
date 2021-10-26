import face_recognition
import os
import pickle
import numpy as np
import cv2
from ColorLayoutComputer import *
from utilities import *
import itertools
#import skcuda
#import pycuda.autoinit
#import pycuda.gpuarray as gpuarray
#import skcuda.linalg as linalg
from project_setting import feature_dir,filter_tracking_result_dir, stats_dir
from scipy.spatial import distance
class FaceGroupingGPU:
    
    def __init__(self, face_folder_dir):
        self.face_folder_dir = face_folder_dir
        self.sub_folder = os.listdir(face_folder_dir) 
        self.found_group:{str,[]} = {}
        self.encodings_lookup = {}
        self.table_list = []
        self.fn_list = []
        self.distance_map = {}
        self.find_face()
        dump_pickle(join(stats_dir,"found_group.pkl"), self.found_group)
            
    def construct(self, talble_fns, group_file=None):
        for i in talble_fns:
            self.table_list.append(self.load_encodings_lookup(i))
        self.fn_list = list(self.table_list[0].keys())
        if group_file == None: 
            self.group_face()
        else:
            self.update_group_file(talble_fns)
            self.group_class_based_on_cluster(group_file)
        self.export_face_group()
        self.export_group_mapping()
    
    def update_group_file(self, talble_fns):
        target_list = list(self.load_pickle(talble_fns[0]).keys())
        for f in target_list:
            group_num = f.split("_")[-1].split(".")[0]
            if group_num in self.found_group:
                self.found_group[group_num].append(f)
            else:
                self.found_group[group_num] = []
                self.found_group[group_num].append(f)    

        
    def find_face(self):
        self.found_group = self.load_encodings_lookup(join(stats_dir,"found_group.pkl"))
        if self.found_group == None:
            self.found_group = {}
        else:
            return
        counter = 0
        for f in self.sub_folder:
            group_folder = join(self.face_folder_dir,f)
            imf = sorted(os.listdir(group_folder))
            if len(imf) <= 3:
                continue
            print("current progress ", f)
            for i in imf:
                counter = counter + 1
                image = face_recognition.load_image_file(join(group_folder,i))
                face_locations =  face_recognition.face_locations(image,number_of_times_to_upsample=1,model="cnn")
                if len(face_locations) != 0:
                    group_num = f
                    if group_num in self.found_group:
                        self.found_group[group_num].append(i)
                    else:
                        self.found_group[group_num] = []
                        self.found_group[group_num].append(i)
    
    def group_face(self):
        face_group = sorted(list(self.found_group.keys()))
        for i in range(len(face_group)):
            group_num = face_group[i]
            for j in range(i+1, len(face_group)):
                group_num_cmp =face_group[j]
                if group_num == group_num_cmp:
                    continue 
                self.distance_map[str(group_num)+"##"+str(group_num_cmp)] = self.distance(group_num, group_num_cmp)
    
    def group_class_based_on_cluster(self,group_file):
        if len(group_file) != 2:
            return
        for g in group_file:
            if "good" in g:
                good_group = self.load_pickle(g)
            else:
                bad_group = self.load_pickle(g)
        bad_classes = list(bad_group.values())
        bad_classes = list(itertools.chain(*bad_classes))
        good_group_strkey = self.change_dict_key_with_string(good_group)

        ## self loop 
        for k in good_group_strkey.keys():
            for kk in good_group_strkey.keys():
                if k == kk:
                    continue
                self.distance_map[k+"##"+kk] = self.distance(list(good_group_strkey[k]), list(good_group_strkey[kk]))
        #other loop
        for k in good_group_strkey.keys():
            for g in bad_classes:
                self.distance_map[str(g)+"##"+k] = self.distance(list(good_group_strkey[k]), g)
        
        ## self badloop
        for g in bad_classes:
            for gg in bad_classes:
                self.distance_map[str(g)+ "##" + str(gg)] = self.distance(g, gg)
        
    
    def change_dict_key_with_string(self, mydict):
        dlim = "_" 
        res = {}
        for k in mydict:
            if len(mydict[k]) == 1:
                new_key = str(mydict[k][0])
            else:
                new_key = dlim.join(str(i) for i in mydict[k])
            res[new_key] = mydict[k]
        return res
    
    def export_face_group(self):
        found = self.found_group.keys()
        with open(join(stats_dir,"face.txt"), "w") as f:
            for ff in found:
                f.write(ff +  "\n")

    def export_group_mapping(self):
        with open(join(stats_dir,"group_matching.txt"), "w") as f:
            for k in self.distance_map.keys():
                k_s = k.split("##")
                line = str(k_s[0]) + "\t" + str(k_s[1]) + "\t" + str(self.distance_map[k][0])+ "\t" + str(self.distance_map[k][1]) + "\t" + str(self.distance_map[k][2]) + "\n"
                f.write(line)

    def distance(self, group_num, group_num_cmp):
        if isinstance(group_num, list):
            my_encodings = self.merge_list_of_group_feature(group_num)
        else:
            encoding_list = self.get_encoding_list(group_num_cmp)
            my_encodings = self.get_encoding_list(group_num)
        if isinstance(group_num_cmp,list):
            unknown_face_encodings = self.merge_list_of_group_feature(group_num_cmp)
        else:    
            encoding_list = self.get_encoding_list(group_num_cmp)
            unknown_face_encodings = encoding_list
        results = distance.cdist(my_encodings, unknown_face_encodings)
        dis = np.median(results.flatten())
        var = results.var()
        return dis, var, dis*1000*var

    def merge_list_of_group_feature(self, list_group):
        my_encodings = []
        for g in list_group:
            my_encodings.append(self.get_encoding_list(g))
        my_encodings = np.squeeze(np.concatenate(my_encodings))
        return my_encodings


    def standardize(self, a):
        a = (a - np.mean(a)) / np.std(a)
        return a
    
    def get_encoding_list(self, group_num:str):
        if isinstance(group_num, int):
            group_num = str(group_num)
        encodings = []
        for image_fn in self.found_group[group_num]:
            ecd = self.get_all_feature(image_fn, self.table_list)
            if ecd is not None:
                encodings.append(ecd)
        return np.concatenate(encodings)
            
    def load_encodings_lookup(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
            print(fn, "is", len(lookup.keys()))
        return lookup
    
    def get_all_feature(self, fn, table_list):
        res_list = []
        for t in table_list:
            if fn not in t:
                return None
            if isinstance(t[fn], list) and len(t[fn][0]) >= 128:
                res_list.append(t[fn][0][0:128].reshape(1, -1)*2)
            else:
                res_list.append(t[fn])
        return np.concatenate(res_list, axis= None).reshape(1, -1) 
    
    def load_pickle(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
            print(fn,"is" ,len(lookup.keys()))
        return lookup