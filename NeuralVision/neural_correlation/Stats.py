import numpy as np 
from Interval import Interval
import more_itertools as mit
from project_setting import episode_number
from Interval import Interval
if episode_number == 1:
    from data import character_dict, frame_dir, sr_final_movie_data,num_characters
elif episode_number == 2:
    from data2 import character_dict, frame_dir, sr_final_movie_data,num_characters
from statistics import mode
import matplotlib.pyplot as plt
import pandas as pd
import os
class Stats:
    def __init__(self, patient ,interval_list,result_board, label_board, predicton, frame_board, patient_response_board, k = -1 ):
        self.patient = patient
        self.interval_list:[Interval] = interval_list
        self.result_board = result_board
        self.label_board = label_board
        self.patient_prob = predicton
        self.patient_response_board = patient_response_board
        self.k = k
        self.frame_board = frame_board
        self.prob_cond_predict = {}#str(character_index) + "#" + str(character_index_conditon) + "#" + str(episode_number)
        self.prob_cond_label = {}
        self.feed_prediction_interval()
        self.prob_cond_predict = self.prob_hist_cond_predict()
        self.prob_cond_label = self.prob_hist_cond_label()

    def __str__(self):
        line = str(self.patient) + " "
        for i in self.interval_list:
            line += str(i) + "\n"
        return line
    
    def feed_prediction_interval(self):
        self.interval_list.sort()
        for idx, item in enumerate(self.interval_list):
            start, end = item.get_start_end()
            item.set_prediction(self.patient_prob[:,start:item.response_sample_end])
            #break
    def get_clip_cvlabel(self):
        res = []
        self.interval_list.sort()
        for item in self.interval_list:
            res.append(np.max(item.cv_label,axis=1))
        return np.array(res)

    def get_clip_episode(self):
        res = []
        self.interval_list.sort()
        for item in self.interval_list:
            res.append(item.episode)
        return np.array(res)

    def stats_accuracy(self, threshold=0.5, mode="size"):
        matrix = np.zeros((2, 4,2,2))
        for i in self.interval_list:
            cv_label = i.get_cvlabel(threshold=threshold, mode=mode) 
            prediction = i.get_predictions()
            for character_idx in range(num_characters):
                matrix[int(i.episode -1),character_idx, int(not prediction[character_idx]), int(not cv_label[character_idx])] += 1
        return matrix

    def get_prob_cond_prob_all(self, mode="prob", clip_mode="all"):
        res = {}
        self.interval_list.sort()
        for character_index_conditon in range(num_characters):
            related_interval = []
            for i in self.interval_list:
                if mode is "prob":
                    check = i.check_character_prob(character_index_conditon, mode = clip_mode)
                elif mode is "label":
                    check = i.check_character_label(character_index_conditon)
                if check:
                    related_interval.append(i)
            for character_index in range(num_characters):
                for i in related_interval:
                    key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(i.get_episode_number()) + "#" + str(i.get_patient_response_ans())
                    if key in res:
                        res[key].append(i.get_character_prob(character_index, mode = clip_mode))
                    else:
                        res[key] = []
                        res[key].append(i.get_character_prob(character_index, mode = clip_mode)) 
        return res

    def prob_hist_cond_predict(self, clip_mode="all"):   
        res = {}
        for character_index_conditon in range(num_characters):
            related_interval = []
            for i in self.interval_list:
                if i.check_character_prob(character_index_conditon, mode=clip_mode):
                    related_interval.append(i)
            for character_index in range(num_characters):
                for episode_number in [1,2]:
                    relative_character_prob = []
                    for i in related_interval:
                        if i.episode == episode_number: 
                            relative_character_prob.append(i.get_character_prob(character_index, mode=clip_mode))
                    relative_character_prob = np.array(relative_character_prob).flatten()
                    key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(episode_number)
                    res[key] = relative_character_prob
        #print(relative_character_prob)
        return res
    
    def prob_hist_cond_label(self, clip_mode="all"):
        res = {}
        for character_index_conditon in range(num_characters):
            related_interval = []
            for i in self.interval_list:
                if i.check_character_label(character_index_conditon):
                    related_interval.append(i)
            for character_index in range(num_characters):
                for episode_number in [1,2]:
                    relative_character_prob = []
                    for i in related_interval:
                        if i.episode == episode_number: 
                            relative_character_prob.append(i.get_character_prob(character_index, mode=clip_mode))
                    relative_character_prob = np.array(relative_character_prob).flatten()
                    key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(episode_number)
                    res[key] = relative_character_prob
        return res


    def find_ranges(self, iterable):
        """Yield range of consecutive numbers."""
        for group in mit.consecutive_groups(iterable):
            group = list(group)
            if len(group) == 1:
                yield group[0]
            else:
                yield group[0], group[-1]
    
    def prediction_percentage(self, threshold = 0.5):
        res = np.zeros(len(self.patient_prob.keys()))
        for character_index in sorted(self.patient_prob.keys()):
            res[character_index] = len(np.where(self.patient_prob[character_index] > threshold)[0]) *1.0 / len(self.patient_prob[character_index] )
        return res
    #def deploy_test_info(self):

    def get_edge_prediction(self, width = 2):
        width_sample = int(width * sr_final_movie_data)
        res = {}
        res["start"] =  []
        res["end"] = []
        res["response_end"] = []
        for i in self.interval_list:
            start, end = i.get_start_end()
            response_sample_end = i.response_sample_end
            res["start"].append(self.patient_prob[:,start-width_sample: start+width_sample])
            res["end"].append(self.patient_prob[:,end-width_sample: end+width_sample])
            res["response_end"].append(self.patient_prob[:,response_sample_end-width_sample: response_sample_end+width_sample])
        return res


    def exclusive_coactivation(self, mode="prob"):
        res = np.zeros((num_characters, num_characters))
        cvlabels = self.get_clip_cvlabel()
        #print(cvlabels.shape)
        clips = np.array(self.interval_list)
        for i in range(num_characters): # condition
            for j in range(num_characters): # checked
                #extracted_condition:
                conditioned_index = np.where(cvlabels[:,i] == 1)[0]
                #print(i, j, conditioned_index)
                exclusive_index = np.where(cvlabels[conditioned_index,j] == 0)[0]
                #print(i, j,exclusive_index)
                if len(exclusive_index) == 0:
                    res[i, j] = 0
                    continue
                clip_jcond_i = clips[exclusive_index] 
                res[i, j] = self._cal_clips_cooccurance(clip_jcond_i, i, j, mode)
        return res

    def _cal_clips_cooccurance(self, clip_list, character_cond, character_check, mode="prob"):
        clip_condi = self._get_occurance_clip(clip_list, character_cond)
        if len(clip_condi) == 0:
            return 0 
        clip_check = self._get_occurance_clip(clip_condi, character_check)
        if mode == "prob":
            return len(clip_check)*1.0/len(clip_condi)
        elif mode == "raw":
            return len(clip_list)
        elif mode == "prob2":
            clip_check1 = self._get_occurance_clip(clip_list, character_check)
            return len(clip_check1)/len(clip_list)
        else:
            print("raw or prob all no")
        return None

    def _get_occurance_clip(self, clip_list, char):
        res = []
        for c in clip_list:
            if c.check_character_prob(char, threshold = 0.5, mode="clip"):
                res.append(c)
        return res


    def draft_function_get_yes_distribution(self, char_id):
        character_prob = []
        for i in self.interval_list:
            if i.check_character_label(char_id):
                #print(np.array(i.prediction).shape)
                character_prob.append(np.array(i.prediction)[char_id,:])
        character_prob = np.hstack(character_prob)
        return character_prob

    
    '''
    def parse_interval_appearance(self):
        interval_list = []
        for ep_number in [1, 2]:
            ep_location = np.where(self.label_board == ep_number)[0]
            ranges = list(self.find_ranges(ep_location))    
            for r in ranges:
                interval = Interval(r[0], r[1])
                interval.set_episode(ep_number)
                label = self.result_board[:,r[0]:r[1]]
                interval.set_label(label)
                prediction = []
                for character_index in self.patient_prob.keys():
                    prediction.append(self.patient_prob[character_index][r[0]:r[1]])
                prediction = np.vstack(prediction)
                interval.set_prediction(prediction)
                interval.set_frame_number(self.frame_board[r[0]:r[1]])
                interval.set_patient_response(mode(self.patient_response_board[r[0]:r[1]]))
                interval_list.append(interval)
        self.interval_list = interval_list
    '''

    ''' 
    MTL region experiment_scripts
    '''
    def prob_hist_cond_predict_MTL(self, threshold, threshold_num,clip_mode="all"):   
        res = {}
        for character_index_conditon in range(num_characters):
            related_interval = []
            for i in self.interval_list:
                if i.check_character_prob_MTL(character_index_conditon,threshold, threshold_num, mode=clip_mode):
                    related_interval.append(i)
            for character_index in range(num_characters):
                for episode_number in [1,2]:
                    key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(episode_number)
                    if character_index == character_index_conditon:
                        res[key] = 1
                        continue
                    count = 0
                    relative_character_prob = []
                    for i in related_interval:
                        if i.episode == episode_number: 
                            relative_character_prob.append(i.check_character_prob_MTL(character_index,threshold, threshold_num, mode=clip_mode))
                            count += 1
                    if count == 0:
                        res[key] = 0
                    else:
                        relative_character_prob = np.sum(relative_character_prob)
                        res[key] = relative_character_prob*1.0/count
        return res
    
    def create_dict(self, ls1, ls2):
        res = {}
        for idx in range(len(ls1)):
            res[ls1[idx]] = ls2[idx]
        return res
    def parse_region(self, mapping, region_name):
        res = []
        for r in region_name:
            res.append(mapping[r])
        return np.array(res)
    
    def parse_squeeze(self, x):
        res = []
        for xx in x:
            res.append(xx[0])
        return res

    def cal_activation(self):
        df = pd.read_csv("/media/yipeng/data/movie_2021/Movie_Analysis/neural_correlation/region_name.csv")
        mapping = self.create_dict(df["region_name"].values, df["MTL"].values)
        surfix = "features_mat_regions_clean.npy"
        data_folder = "/media/yipeng/data/movie_2021/Movie_Analysis/data"
        region_fn = os.path.join(data_folder,  self.patient , surfix)
        region = self.parse_squeeze(np.load(region_fn, allow_pickle=True))
        MTL_mask = self.parse_region(mapping, region)
        no_mtl_index = np.where(MTL_mask ==0)[0]
        mtl_index = np.where(MTL_mask == 1)[0]
        print("MTL", len(mtl_index), "nonMTL", len(no_mtl_index))
        mtl_data_list = []
        non_mtl_data_list = []
        for i in self.interval_list:
            data = i.memtest_neural_signal
            #print(data.shape)
            mtl_data = data[mtl_index,:]
            no_mtl_data =data[no_mtl_index,:]
            mtl_data_list.append(mtl_data)
            non_mtl_data_list.append(no_mtl_data)
        return mtl_data_list, non_mtl_data_list       