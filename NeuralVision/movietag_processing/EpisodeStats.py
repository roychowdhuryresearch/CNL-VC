import sys
sys.path.append(".")
import scipy as sp
import scipy.io as sio
import numpy as np
import os
import shutil
from Clip import Clip
from Scene import Scene
import pickle
#from neural_correlation.utilities import clean_folder
class EpisodeStates:
    def __init__(self, stats_fn, episode_number, cvlabel_dir):
        self.stats_fn = stats_fn
        self.episode_number = -1
        self.total_frames = 74251
        self.cvlabel_dir = cvlabel_dir
        
        self.clips_overall:[Clip] = []
        self.init_frame_all = None
        self.cvlabel_dict = {}
        self.scene_dict = {}
        
        self.construct()
    
    def __str__(self):
        for i in sorted(self.clips_overall):
            print(i)

    def construct(self):
        stats = sio.loadmat(self.stats_fn)
        scenechange_mode = stats["scenechange_mode"].flatten()
        init_frame_all = stats["init_frame_all"].flatten()
        #init_frame_all = self.parse_to_30(init_frame_all)
        scenechange_indices_in = stats["scenechange_indices_in"].flatten()
        character_checkboxes = stats["char_checkbox_mode"]
        char_emotions = stats["char_emotions_mode"]
        char_movement = stats["char_movement_mode"]
        char_poses = stats["char_poses_mode"]
        char_talking = stats["char_talking_mode"]
        character_checkboxes = character_checkboxes+char_emotions+char_poses+char_talking+char_movement
        self.construct_clips(init_frame_all, character_checkboxes)
        self.assign_scene(scenechange_indices_in, scenechange_mode)
        self.assign_cvlabel()
        #self.cvlabel_filtering()
        self.create_scene_dict()
    def parse_to_30(self, init_frame_all):
        res = []
        for i in init_frame_all:
            res.append(int(i/24*30))
        return np.array(res)
    def benchmark_cv_labels(self):
        res = np.zeros((4, 2, 2))
        for c in self.clips_overall:
            res += c.cv_accuracy()
        return res

    def cvlabel_filtering(self):
        for i in self.clips_overall:
            i.cvlabel_filtering() 

    def construct_clips(self, init_frame_all, character_checkboxes = None):
        self.init_frame_all = init_frame_all - 1 ## we start as frame 0 
        num_clip = len(self.init_frame_all)
        for i in range(num_clip):
            if i == 0:
                clip = Clip(0, init_frame_all[i]-1)
            elif i == num_clip -1:
                clip = Clip(self.init_frame_all[i], self.total_frames)
            else:    
                clip = Clip(self.init_frame_all[i], self.init_frame_all[i+1]-1)
            if character_checkboxes is not None:
                clip.update_character_checkbox(character_checkboxes[i])
            self.clips_overall.append(clip)

    def get_all_frame_character(self, character_index):
        res_dir = os.path.join("/media/yipeng/data/movie/Movie_Analysis/movietag_processing/result/contains/", str(character_index))
        self.clean_folder(res_dir)
        frame_dir = "/home/yipeng/Desktop/video_process/frames/"
        self.clips_overall.sort()
        for idx, i in enumerate(self.clips_overall):
            if i.contains_character(character_index):
                for i in range(i.frame_start, i.frame_end, 4):
                    image_fn = "frame_" + str(i) + ".jpg"
                    image_fn_save = "frame_" + str(i) + "_"+ str(idx+1)+ ".jpg"
                    shutil.copy(os.path.join(frame_dir,image_fn), os.path.join(res_dir, image_fn_save))
        return

    def export_annotation_label(self):
        res = {}
        for i in range(4):
            res[i] = set()
        self.clips_overall.sort()
        for c in self.clips_overall:
            temp = c.get_character_checkbox()
            for character_index in temp:
                res[character_index] = res[character_index].union(temp[character_index])
        return res


    def get_all_fn_character(self, character_index):
        res_dir = os.path.join("/media/yipeng/data/movie/Movie_Analysis/movietag_processing/result/contains_fn/", str(character_index))
        self.clean_folder(res_dir)
        frame_dir = "/home/yipeng/Desktop/video_process/frames/"
        self.clips_overall.sort()
        for idx, i in enumerate(self.clips_overall):
            if i.cv_accuracy()[character_index,1,0] == 1:
                for i in range(i.frame_start, i.frame_end, 8):
                    image_fn = "frame_" + str(i) + ".jpg"
                    image_fn_save = "frame_" + str(i) + "_"+ str(idx+1)+ ".jpg"
                    shutil.copy(os.path.join(frame_dir,image_fn), os.path.join(res_dir, image_fn_save))
        return

    def get_all_frame_fp(self, character_index):
        res_dir = os.path.join("/media/yipeng/data/movie/Movie_Analysis/movietag_processing/result/FPs_cv/", str(character_index))
        self.clean_folder(res_dir)
        frame_dir = "/home/yipeng/Desktop/video_process/frames/"
        self.clips_overall.sort()
        for idx, i in enumerate(self.clips_overall):
            if i.cv_accuracy()[character_index,0,1] == 1:
                '''
                for i in range(i.frame_start, i.frame_end, 4):
                    image_fn = "frame_" + str(i) + ".jpg"
                    image_fn_save = "frame_" + str(i) + "_"+ str(idx)+ ".jpg"
                    shutil.copy(os.path.join(frame_dir,image_fn), os.path.join(res_dir, image_fn_save))
                '''
                for i in i.cvlabel_checkbox[character_index]:
                    image_fn = "frame_" + str(i) + ".jpg"
                    image_fn_save = "frame_" + str(i) + "_"+ str(idx)+ ".jpg"
                    shutil.copy(os.path.join(frame_dir,image_fn), os.path.join(res_dir, image_fn_save))
        return


    def assign_scene(self, scenechange_indices_in, scenechange_mode):
        start_index = np.where(scenechange_mode == -1)[0][-1]
        all_start = self.init_frame_all[start_index]
        start_frames_middle = self.init_frame_all[np.where(np.multiply(scenechange_mode, scenechange_indices_in)==1)] 
        start_frames = np.insert(start_frames_middle, 0, all_start)
        end_frames_middle = start_frames_middle -1
        end_frames = np.append(end_frames_middle,self.init_frame_all[-1])
        for i in range(len(start_frames)):
            start = start_frames[i]
            end = end_frames[i]
            for c in self.clips_overall:
                if c.within(start, end):
                    c.set_scene(i)
        
    def create_scene_dict(self):
        for c in self.clips_overall:
            scene = c.scene
            if scene in self.scene_dict:
                self.scene_dict[scene].append(c)
            else:
                self.scene_dict[scene] = Scene(scene)
                self.scene_dict[scene].append(c)
        for scene in self.scene_dict.keys():
            self.scene_dict[scene].stats_cal()
    
    def character_association(self, threshold=0.1):
        res = {}
        character_set = self.cvlabel_dict.keys()
        for character_conditional in character_set:        
            condition_scenes, total_length, _ = self.find_relative_scenes(list(self.scene_dict.values()),character_conditional, threshold=threshold)
            for character_index in character_set:
                key = str(character_index) + "|" + str(character_conditional)
                if total_length == 0:
                    res[key] = 0
                    continue
                _ , length_obser, _ = self.find_relative_scenes(condition_scenes,character_index, threshold=threshold)
                res[key] = length_obser *1.0/total_length
        return res

    def character_conditional_prob_overall(self):
        #P(Jack|Terro) == \sum_{I=1}^{K}P(I) * {min(Jack(I), Terro(I))}/{Terro(I)}
        res = {}
        character_set = self.cvlabel_dict.keys()
        for s_index in self.scene_dict.keys():
            scene = self.scene_dict[s_index] 
            scene_length = scene.length 
            Pscene = scene_length*1.0/self.total_frames
            for character_conditional in sorted(character_set):
                for character_index in sorted(character_set):
                    key = str(character_index) + "|" + str(character_conditional)
                    probalibity = 0
                    if character_conditional in scene.character_occurance and character_index in scene.character_occurance:
                        scene_character_conditonal = scene.character_occurance[character_conditional]
                        scene_character = scene.character_occurance[character_index]
                        if scene_character_conditonal != 0:
                            probalibity = Pscene * min(scene_character,scene_character_conditonal)*1.0/scene_character_conditonal
                        else:
                            probalibity = 0
                        if key == str(2) + "|" + str(0):
                            print(scene_character, scene_character_conditonal, probalibity)
                    else:
                        probalibity = 0 
                    if key in res:
                        res[key] += probalibity
                    else:
                        res[key] = probalibity
        return res
    
    def character_conditional_prob(self, mode="sum_first"):
        #P(Jack|Terro) == \sum_{I=1}^{K}P(I) * {min(Jack(I), Terro(I))}/{Terro(I)}
        res = {}
        character_set = sorted(self.cvlabel_dict.keys())
        for character_conditional in character_set:        
            condition_scenes, total_length, _ = self.find_relative_scenes(list(self.scene_dict.values()),character_conditional)
            for character_index in character_set:
                key = str(character_index) + "|" + str(character_conditional)
                if key not in res:
                    res[key] = 0
                if mode is "sum_first":
                    for scene in condition_scenes:
                        occurance_condition = scene.character_occurance[character_conditional]
                        scene_length = scene.length
                        if character_index not in scene.character_occurance: 
                            prob = 0 
                        else:
                            occurance_obser = scene.character_occurance[character_index]
                            prob = scene_length*min(occurance_condition, occurance_obser)*1.0/(total_length*occurance_condition)
                        res[key] += prob
                        if key == str(2) + "|" + str(0):
                            print(scene_length, total_length, occurance_condition , occurance_obser, prob)
                else:    
                    _ , length_obser, occurance_obser = self.find_relative_scenes(condition_scenes,character_index)
                    res[key] = length_obser * min(occurance_obser, occurance_condition) *1.0/(total_length * occurance_condition)
        return res

    def find_relative_scenes(self, scene_list ,character_index,threshold = 0):
        res = []
        total_length = 0
        total_condition_occurance = 0
        for scene in scene_list:
            if scene.appear_enough(character_index, threshold):
                res.append(scene)
                total_length += scene.length
                total_condition_occurance += scene.character_occurance[character_index]
        return res, total_length, total_condition_occurance

    def parse_cvlabel(self):
        for fn in os.listdir(self.cvlabel_dir):
            character_index = int(fn.strip().split(".")[0])
            self.cvlabel_dict[character_index] = self.load_pickle(os.path.join(self.cvlabel_dir, fn))

    def assign_cvlabel(self):
        self.parse_cvlabel()
        for character_index in self.cvlabel_dict.keys():
            frame_occurance = self.cvlabel_dict[character_index]
            for frame_num in frame_occurance:
                for c in self.clips_overall:
                    if c.contains(frame_num):
                        c.update_cvlabel(character_index, frame_num)
                        break
    def print_scene(self):
        for i in self.scene_dict:
            print(i, "~~",self.scene_dict[i].length, "~",self.scene_dict[i].character_occurance)

    def plot_scene(self, frame_dir, result_dir):
        output_dir = os.path.join(result_dir,"scene_split") 
        self.clean_folder(output_dir)
        for scene in self.scene_dict.keys():
            one_scene_folder = os.path.join(output_dir, str(scene))
            os.mkdir(one_scene_folder)
            for clip in self.scene_dict[scene]:
                for frame in sorted(clip.frame_set):
                    file_name = "frame_"+str(frame)+".jpg"
                    from_file = os.path.join(frame_dir,file_name)
                    to_file = os.path.join(one_scene_folder, file_name)
                    shutil.copyfile(from_file, to_file)

    def clean_folder(self, saved_fn):
        if not os.path.exists(saved_fn):
            os.mkdir(saved_fn)
        else:
            shutil.rmtree(saved_fn)
            os.mkdir(saved_fn)

    def load_pickle(self, fn):
        if not os.path.exists(fn):
            print(fn , " notexist")
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
            print(fn)
        return lookup

    def get_character_distribution(self):
        self.clips_overall.sort()
        res_cv = np.zeros((len(self.clips_overall),4))
        res_annotation = np.zeros((len(self.clips_overall),4))
        for c_idx, c in enumerate(self.clips_overall):
            res_cv[c_idx,:] = c.get_overall_cvlabel()
            res_annotation[c_idx,:] = c.get_overall_character()
        return res_cv.T, res_annotation.T