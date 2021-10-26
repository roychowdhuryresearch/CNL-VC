import sys
sys.path.append(".")
import os 
from data import patientNums, path_to_matlab_generated_movie_data, num_characters
from memtest_setting import *
import scipy as sp
import scipy.io as sio
import numpy as np 
from utilities import *
from Stats import Stats
import copy
import argparse
import shutil
episode1_patch_size = load_pickle("/media/yipeng/data/movie/Movie_Analysis/draft_result/episode1_character_size.pkl")
episode2_patch_size = load_pickle("/media/yipeng/data/movie/Movie_Analysis/draft_result/episode2_character_size.pkl")
class KfoldStats:
    def __init__(self, patient, result_dir, k):
        self.patient = patient
        self.stats_list: Stats = []
        self.result_dir = result_dir
        self.k = k
        self.interval_list = []
        self.episode1_cv_label_fn = episode1_cv_label_fn
        self.episode2_cv_label_fn = episode1_cv_label_fn
        self.offset = 2520
        self.process()
        

    def process(self):
        patient = self.patient
        patient_folder = os.path.join(path_to_matlab_generated_movie_data, patient)
        file_list = os.listdir(patient_folder)
        ts_file = os.path.join(path_to_matlab_generated_movie_data, patient+"/triggers_"+patient+"series1_ht.mat")
        if not os.path.exists(ts_file):
            print("ts_file doesn't exist!", patient, ts_file)
            return
        ts_info = sio.loadmat(ts_file)
        ts_offset = ts_info["br_triggers"].squeeze()[0] - ts_info["movie_times"].squeeze()[0]
        #print("br_triggers", ts_info["br_triggers"].squeeze()[0])
        #print("movie_times", ts_info["movie_times"].squeeze()[0])
        for folder in file_list:
            if "memtest1" in folder:
                men_test_folder = os.path.join(patient_folder, folder)
                exp_fn = "psychophysics/" + "EXP_" +  patient+ "memtest1.mat"
                trigger_fn =  "triggers/" + "triggers_" +  patient+ "memtest1.mat"
                exp_file = sio.loadmat(os.path.join(men_test_folder, exp_fn))
                trigger_file = sio.loadmat(os.path.join(men_test_folder, trigger_fn))
                triggle_file_start = np.array(trigger_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
                triggle_file_end = np.array(trigger_file["triggers"]["trigger_clip_end_ttl"][0][0][0])
                start_time_stamp, end_time_stamp = triggle_file_start[0], triggle_file_end[-1] 
                #print("start_time_stamp", start_time_stamp)
                #print("ts_offset", ts_offset)
                #print("self.offset", self.offset)
                start_sample_idx = int((start_time_stamp - ts_offset - self.offset)*7.5 + 0.9999/7.5)
                end_sample_idx = int((end_time_stamp - ts_offset - self.offset)*7.5) +1
                #print("start, end_sample idx:", (start_time_stamp- ts_offset - offset)*7.5, (end_time_stamp - ts_offset - offset)*7.5)
                result_board, label_board, frame_board, patient_response_board= self.get_result_board(exp_file, trigger_file)
                self.create_interval_list(exp_file, trigger_file, ts_offset)
                #print(len(self.interval_list))
                #self.plot_response_time_vs_options()
        '''
        ep1=0
        ep2=0
        boom =0
        for i in self.interval_list:
            if i.episode == 1:
                ep1 +=1
            elif i.episode == 2:
                ep2 += 1
            else:
                boom +=1
        #print(ep1, ep2, boom, len(self.interval_list))
        '''
        for k in range(self.k):
            if self.k == 1:
                patient_result_folder = os.path.join(self.result_dir, patient)
                direct = os.path.join(self.result_dir, self.patient)
            else:
                patient_result_folder = os.path.join(self.result_dir, patient, str(k))
                direct = os.path.join(self.result_dir, self.patient ,str(k))
            patient_prob = load_pickle(os.path.join(patient_result_folder, "memory_test_prob.pkl"))
            patient_prob_np = self.convert_dict2np(patient_prob)
            patient_prob_cut = self.cut_prob_result(patient_prob, start_sample_idx, end_sample_idx)
            stats = Stats(self.patient, copy.deepcopy(self.interval_list) ,result_board, label_board, patient_prob_np, frame_board, patient_response_board, k = k)
            self.stats_list.append(stats)
            
            self.plot_overall_probabilities(patient_prob_cut, result_board, label_board, direct)
            #self.plot_prob_vs_options(stats, k, mode = "label")
            #self.plot_prob_vs_options(stats, k, mode = "prob")
        #self.memtest_vis()
        #self.memtest_benchmark()
    
    def memtest_vis(self):
        res_folder = os.path.join("/media/yipeng/data/movie/Movie_Analysis/memtest_vis", self.patient)
        frame_folder = "/media/yipeng/Sandisk/yolov3/yolov3/frame_sample"
        frame_folder_2 = "/media/yipeng/data/movie/frame_2"
        clean_folder(res_folder)
        for i in range(len(self.interval_list)):
            interval = self.interval_list[i]
            kfold_prediction = []
            for s in self.stats_list:
                kfold_prediction.append(s.interval_list[i].prediction)
            prediction = np.sum(np.concatenate(kfold_prediction, axis=1) >0.5, axis=1) > 1
            #start_frame = int(interval.time_start*24/4)*4
            #end_frame = int(interval.time_end*24/4)*4
            for character_id in range(4):
                if interval.check_character_label(character_id):
                    for frame_num in range(interval.frame_number[0], interval.frame_number[-1], 1):
                        if interval.episode == 1:
                            frame_size_check = episode1_patch_size
                        else:
                            frame_size_check = episode2_patch_size
                        if frame_num not in frame_size_check[character_id]:
                            continue
                        if interval.episode == 1:
                            frame_fn = os.path.join(frame_folder, "frame_"+str(frame_num)+".jpg")
                            #dis_fn = os.path.join(res_folder, str(i) + "_frame_"+str(frame_num)+"_"+str(prediction.astype(int))+str(int(interval.episode))+"_.jpg" )
                            #shutil.copy(frame_fn, dis_fn)
                        else:
                            frame_fn = os.path.join(frame_folder_2, "frame_"+str(frame_num)+".jpg")
                        size = frame_size_check[character_id][frame_num]
                        dist_folder = os.path.join(res_folder, str(character_id))
                        if not os.path.exists(dist_folder):
                            os.mkdir(dist_folder)
                        dis_fn = os.path.join(dist_folder,str(i) + "_frame_"+str(frame_num)+"_"+str(prediction.astype(int))+str(int(interval.episode))+"_"+str(int(size))+"_.jpg" )
                        shutil.copy(frame_fn, dis_fn)
    def memtest_vis_all(self):
        res_folder = os.path.join("/media/yipeng/data/movie/Movie_Analysis/memtest_vis_all", self.patient)
        frame_folder = "/media/yipeng/Sandisk/yolov3/yolov3/frame_sample"
        frame_folder_2 = "/media/yipeng/data/movie/frame_2"
        clean_folder(res_folder)
        for i in range(len(self.interval_list)):
            interval = self.interval_list[i]
            kfold_prediction = []
            for s in self.stats_list:
                kfold_prediction.append(s.interval_list[i].prediction)
            prediction = np.sum(np.concatenate(kfold_prediction, axis=1) >0.5, axis=1) > 1
            #start_frame = int(interval.time_start*24/4)*4
            #end_frame = int(interval.time_end*24/4)*4
            for frame_num in range(interval.frame_number[0], interval.frame_number[-1], 4):
                if interval.episode == 1:
                    frame_fn = os.path.join(frame_folder, "frame_"+str(frame_num)+".jpg")
                else:
                    frame_fn = os.path.join(frame_folder_2, "frame_"+str(frame_num)+".jpg")
                dist_folder = os.path.join(res_folder, self.patient)
                if not os.path.exists(dist_folder):
                    os.mkdir(dist_folder)
                dis_fn = os.path.join(dist_folder,str(i) + "_frame_"+str(frame_num)+"_"+str(prediction.astype(int))+str(int(interval.episode))+"_"+str(interval.get_cvlabel())+"_.jpg" )
                shutil.copy(frame_fn, dis_fn)
    
    def exclusive_coactivation(self):
        res = []
        for s in self.stats_list:
            res.append(s.exclusive_coactivation(mode = "prob")) 
        res = np.mean(np.array(res), axis = 0)
        return res

    def exclusive_coactivation_count(self):
        res = []
        for s in self.stats_list:
            res.append(s.exclusive_coactivation(mode= "raw")) 
        res = np.mean(np.array(res), axis = 0)
        return res

    def clip4_div_clip1(self):
        res = []
        for s in self.stats_list:
            res.append(s.exclusive_coactivation(mode= "prob2")) 
        res = np.mean(np.array(res), axis = 0)
        return res



    def memtest_benchmark(self):
        for threshold in [0]:
            matrix = np.zeros((2,4,2,2))
            threshold = threshold*1.0/10
            for s in self.stats_list:
                one_matrix = s.stats_accuracy(threshold)
                matrix_temp = np.sum(one_matrix, axis=0)
                #print(matrix_temp[:,0,0] + matrix_temp[:,1,0])
                matrix += one_matrix
            for i in range(4):
                matrix_temp = np.sum(s.stats_accuracy(), axis=0)
                #print(i,int(matrix_temp[i,0,0]),int(matrix_temp[i,0,1]),int(matrix_temp[i,1,0]), int(matrix_temp[i,1,1]))
            #print("_______________________________")
        #print(np.sum(matrix, axis=0).shape)
            matrix = np.mean(matrix, axis=0)
            #print("ch", "TP", "FP", "FN", "TN")
            matrix_stats = np.mean(matrix, axis=0)
            for i in range(4):
                tp = int(matrix[i,0,0])
                fp = int(matrix[i,0,1])
                fn = int(matrix[i,1,0])
                tn = int(matrix[i,1,1])
                recall = tp*1.0/(tp+fp)
                precision = tp*1.0/(tp+fn)
                print(recall)
                #print(i,int(matrix[i,0,0]),int(matrix[i,0,1]),int(matrix[i,1,0]), int(matrix[i,1,1]))
            #print("recall",matrix_stats[0,0]/(matrix_stats[0,0]+matrix_stats[1,0]), "threshold", threshold, "acc", (matrix_stats[0,0]+matrix_stats[1,1])/matrix_stats.sum())
            #print(np.sum(matrix[0], axis = 0))
        return matrix_stats[0,0]/(matrix_stats[0,0]+matrix_stats[1,0])

    def memtest_benchmark_search(self):
        res = []
        for threshold in range(10):
            matrix = np.zeros((2,4,2,2))
            threshold = threshold*1.0/10
            for s in self.stats_list:
                one_matrix = s.stats_accuracy(threshold)
                #matrix_temp = np.sum(one_matrix, axis=0)
                #print(matrix_temp[:,0,0] + matrix_temp[:,1,0])
                matrix += one_matrix
            #for i in range(4):
            #    matrix_temp = np.sum(s.stats_accuracy(), axis=0)
                #print(i,int(matrix_temp[i,0,0]),int(matrix_temp[i,0,1]),int(matrix_temp[i,1,0]), int(matrix_temp[i,1,1]))
            #print("_______________________________")
        #print(np.sum(matrix, axis=0).shape)
            matrix = np.sum(matrix, axis=0)
            #print("ch", "TP", "FP", "FN", "TN")
            #matrix_stats = np.sum(matrix, axis=0)
            matrix_stats = matrix[0,:,:]
            for i in range(4):
                print(i,int(matrix[i,0,0]),int(matrix[i,0,1]),int(matrix[i,1,0]), int(matrix[i,1,1]))
            #print("recall",matrix_stats[0,0]/(matrix_stats[0,0]+matrix_stats[1,0]), "threshold", threshold, "acc", (matrix_stats[0,0]+matrix_stats[1,1])/matrix_stats.sum())
            #print(np.sum(matrix[0], axis = 0))
            res.append([threshold, matrix_stats[0,0]/(matrix_stats[0,0]+matrix_stats[1,0]), (matrix_stats[0,0]+matrix_stats[1,1])/matrix_stats.sum()])
        #print(matrix_stats[0,0] + matrix_stats[1,0])
        return res



    def plot_prob_on_response_edge(self):
        patient_folder = os.path.join(self.result_dir, self.patient)
        fig_all, ax_all = plt.subplots(1, 4, figsize=(16, 4))
        for k in range(self.k):
            kfold_folder = os.path.join(patient_folder, str(k))
            onefoldStats = self.stats_list[k]
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            for i in onefoldStats.interval_list:
                time = i.get_shifted_time()
                for character_index in range(num_characters):
                    plot_y = int(character_index)
                    ax[plot_y].plot(time, i.prediction[character_index,:], alpha=0.06, c="orange")
                    ax_all[plot_y].plot(time, i.prediction[character_index,:], alpha= 0.04, c="orange")
                    ax[plot_y].set_xlim(-20, 20)
                    ax_all[plot_y].set_xlim(-20, 20)
            fig.savefig(os.path.join(kfold_folder,"response_edge.jpg"))
        fig_all.savefig(os.path.join(patient_folder,"response_edge.jpg"))

    def avg_prob_on_response_edge(self):
        """
        get prob on edge
            
        Return
        ----------
        res : {kfold_num:{mode:[prob]}}
        """

        patient_folder = os.path.join(self.result_dir, self.patient)
        res = {}
        for k in range(self.k):
            res[k] = self.stats_list[k].get_edge_prediction()
        dump_pickle(os.path.join(patient_folder,"prob_on_response_edge_final_4sec.pkl"), res)
        
        character_each_clip = self.stats_list[0].get_clip_cvlabel()
        dump_pickle(os.path.join(patient_folder,"character_each_clip.pkl"), character_each_clip)

        episode_each_clip = self.stats_list[0].get_clip_episode()
        dump_pickle(os.path.join(patient_folder,"episode_each_clip.pkl"), episode_each_clip)
    
    def plot_response_time_vs_options(self):
        plt.close("all")
        res = {}
        for i in self.interval_list:
            response_ans = i.get_patient_response_ans()
            episode_number = i.get_episode_number()
            key = str(int(response_ans)) + "#" + str(int(episode_number))
            if key in res:
                res[key].append(i.get_patient_response_time())
            else:
                res[key] = []
                res[key].append(i.get_patient_response_time())
        
        direct = os.path.join(self.result_dir, str(self.patient))
        fig, ax = plt.subplots(1, 4, figsize=(12, 5))
        for response_ans in [1,2,3,4]:
            for episode_number in [1, 2]:
                key = str(int(response_ans)) + "#" + str(int(episode_number))
                plot_y = int(response_ans - 1)
                if key not in res:
                    continue
                count_ep, bins = np.histogram(res[key])
                ax[plot_y].plot(bins[:-1], count_ep, label = "ep"+str(episode_number) + " sum: " +str(sum(count_ep)))
            ax[plot_y].set_title("response_time of "+ str(response_ans))
            ax[plot_y].set_ylim(0, 30)
            ax[plot_y].set_xlim(0, 11)
            ax[plot_y].set_xlabel("response_time")
            ax[plot_y].legend()
        fig.suptitle("patient "+str(self.patient)+" response_time_vs_options ", fontsize="x-large")
        plt.savefig(os.path.join(direct, "response_time_vs_options_"+self.patient+".jpg"))

    def plot_prob_vs_options(self, one_states,k,mode = "prob"):
        patient_dir = os.path.join(self.result_dir, str(self.patient))
        direct = os.path.join(patient_dir, str(k))
        cond_result = one_states.get_prob_cond_prob_all(mode=mode)
        plt.close("all")
        for response_ans in [1,2,3,4]:
            fig, ax = plt.subplots(num_characters, num_characters, figsize=(20, 10))
            for character_index_conditon in range(num_characters):
                for character_index in range(num_characters): 
                    plot_x = int(character_index_conditon)
                    plot_y = int(character_index*1.0%num_characters)
                    for episode_number in episodes:
                        stats_key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(int(episode_number)) + "#" + str(int(response_ans))
                        if stats_key not in cond_result:
                            continue
                        relative_character_prob = cond_result[stats_key]  
                        self.plot_by_episode(relative_character_prob,episode_number, ax[plot_x, plot_y], mode ="keep")
                    ax[plot_x, plot_y].set_title(str(character_index) + " condition on " + str(character_index_conditon))
                    ax[plot_x, plot_y].legend()
            fig.suptitle("patient"+str(self.patient)+"interval conditon on " + mode + " resp: " + str(response_ans), fontsize="x-large")
            plt.savefig(os.path.join(direct, "mentest_cond_"+mode+"_" + str(response_ans) +".jpg"))

    def convert_dict2np(self,dict_in):
        res_np = np.zeros((len(dict_in), len(dict_in[0])))
        for key in range(len(dict_in)):
            res_np[key,:] = dict_in[key]
        return res_np

    def create_interval_list(self, exp_file, trigger_file, ts_offset):
        ep1_result = self.get_cv_label(episode1_cv_label_fn)
        ep2_result = self.get_cv_label(episode2_cv_label_fn)
        response_arr = exp_file["trial_struct"]["resp_time"][0]
        response_ans = exp_file["trial_struct"]["resp_answer"][0]
        response_time = np.array([response_arr[i][0][0] for i in range(len(response_arr))])
        response_ans = np.array([response_ans[i][0][0] for i in range(len(response_ans))])
        exp_file = exp_file['TRIAL_ID']

        trigger_file_start = np.array(trigger_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
        trigger_file_end = np.array(trigger_file["triggers"]["trigger_clip_end_ttl"][0][0][0])
        patient_folder = os.path.join(path_to_matlab_generated_movie_data, self.patient)
        original_neural_signal = np.load(os.path.join(patient_folder, "features_mat_clean.npy"))
        original_memtest_signal = np.load(os.path.join(patient_folder, "features_mat_clean2.npy"))
        offset = self.offset + ts_offset
        for i in range(exp_file.shape[0]):
            clip_start = trigger_file_start[i] - offset
            clip_end = trigger_file_end[i] - offset
            interval = Interval(clip_start, clip_end)
            one_line = exp_file[i,:]
            episode = one_line[1]
            movie_frame_start = int(one_line[5]*7.5)*frame_sample_frequency
            patient_response = response_ans[i]
            patient_response_time = response_time[i]
            sample_diff = interval.num_samples
            cv_label_board = np.zeros((num_characters, sample_diff))
            movie_frame_numbers = []
            #current_frames = []
            for sample_stp in range(sample_diff):
                current_frame = movie_frame_start + sample_stp * frame_sample_frequency
                #current_frames.append(current_frame)
                if episode == 1: ### first movie
                    check_res = self.check(current_frame, ep1_result)
                else:
                    check_res = self.check(current_frame, ep2_result)
                cv_label_board[:,sample_stp] = check_res
                #print(cv_label_board)
                movie_frame_numbers.append(current_frame)
            #print(current_frames)
            interval.set_cv_label(cv_label_board)
            interval.set_episode(episode)
            interval.set_frame_number(np.array(movie_frame_numbers))
            interval.set_patient_response(patient_response)
            interval.set_response_time(patient_response_time)
            interval.set_orignal_neural_signal(original_neural_signal[:,int(movie_frame_start/2):int(movie_frame_start/2+sample_diff)])
            interval.set_memtest_neural_signal(original_memtest_signal[:,int(interval.start*2):int(interval.start+sample_diff)*2])
            self.interval_list.append(interval)

    def get_result_board(self, exp_file, trigger_file):
        ep1_result = self.get_cv_label("/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_annotation")
        ep2_result = self.get_cv_label("/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_annotation_2")
        
        response_arr = exp_file["trial_struct"]["resp_time"][0]
        patient_response_ans = exp_file["trial_struct"]["resp_answer"][0]
        response_time = np.array([response_arr[i][0][0] for i in range(len(response_arr))])
        patient_response_ans = np.array([patient_response_ans[i][0][0] for i in range(len(patient_response_ans))])
        exp_file = exp_file['TRIAL_ID']

        trigger_file_start = np.array(trigger_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
        trigger_file_end = np.array(trigger_file["triggers"]["trigger_clip_end_ttl"][0][0][0])

        start_time_stamp = trigger_file_start[0]
        end_time_stamp = trigger_file_end[-1]
        start_sample_idx = int((start_time_stamp - time_offset)*7.5)
        end_sample_idx = int((end_time_stamp- time_offset)*7.5)
        total_sample = end_sample_idx - start_sample_idx 
        result_board = np.zeros((num_characters, total_sample)) -1
        label_board = np.zeros(total_sample)
        frame_board = np.zeros(total_sample)
        patient_response_board = np.zeros(total_sample)
        for i in range(exp_file.shape[0]):
            one_line = exp_file[i,:]
            #time_interval = ((np.array([triggle_file_start[i], triggle_file_end[i]]) - triggle_file_start[0])*7.5).astype(int)
            time_interval = (np.array([trigger_file_start[i], trigger_file_end[i]])*7.5 - time_offset * 7.5 -start_sample_idx).astype(int) 
            time_res = (np.array([trigger_file_start[i], trigger_file_end[i]+response_time[i]])*7.5 - time_offset * 7.5 -start_sample_idx).astype(int) 

            label = one_line[1] ## episode
            label_board[time_res[0]:time_res[1]] = label 
            sample_diff = time_interval[1]-time_interval[0]
            #print("at: ", i , time_interval)
            #print("at : ", i ,time_res,response_time[i] ,label)
            result_interval_board = np.zeros((num_characters, sample_diff))
            frame_interval_board = np.zeros(sample_diff)
            movie_frame_start = one_line[5] * fps_movie 
            #print(movie_frame_start)
            movie_frame_start = self.find_closed_4_factor(movie_frame_start)
            for sample_stp in range(sample_diff):
                current_frame = movie_frame_start + sample_stp * frame_sample_frequency
                if label == 1: ### first movie
                    check_res = self.check(current_frame, ep1_result)
                else:
                    check_res = self.check(current_frame, ep2_result)
                result_interval_board[:,sample_stp] = check_res
                frame_interval_board[sample_stp] = current_frame
            result_board[:,time_interval[0]:time_interval[1]] = result_interval_board 
            frame_board[time_interval[0]:time_interval[1]] = frame_interval_board
            patient_response_board[time_interval[0]:time_interval[1]] = patient_response_ans[i]
        return result_board, label_board, frame_board, patient_response_board

    def find_closed_4_factor(self, num):
        num_4 = round(num/4)
        num = num_4*4
        return num
    
    def cut_prob_result(self, patient_prob, start_sample_idx, end_sample_idx):
        res = {}
        for character_index in patient_prob.keys():
            res[character_index] = patient_prob[character_index][start_sample_idx:end_sample_idx]
        return res
    
    def check(self, current_frame, ep_result):
        #print(current_frame)
        res = np.zeros(num_characters)
        for character_idx in range(num_characters):
            res[character_idx] = int(current_frame in ep_result[character_idx])
        #print(res)
        return res
    
    def get_cv_label(self, ep1_dir):
        res = {}
        for character_index in range(num_characters):
            res[character_index]  = set(load_pickle(os.path.join(ep1_dir , str(character_index)+".pkl")))
        return res


    def average_over_kfold(self, mode="cv_label", mode2 ="normalize", clip_mode="all"):
        """
        get 16 plots kfolds stats 
        
        Parameters
        ----------
        mode : str, optional
            the condition on cv_label or CNN_prediction (default is cv_label)
        
        Returns
        -------
        dict
            a dict {character_idx#conditional_character_idx#episode_number: np.array(prob)} 
            a dict key as the character_information and value as nparray of prob
        """
        res = {}
        for s in self.stats_list:
            if mode == "cv_label":
                ss = s.prob_hist_cond_label(clip_mode=clip_mode)
            else:
                ss = s.prob_hist_cond_predict(clip_mode=clip_mode)
            for key in ss.keys():
                if key in res:
                    #print(res[key].shape, ss[key].shape)
                    res[key] = np.concatenate([res[key],ss[key]])
                    #print(res[key].shape)
                else:
                    res[key] = ss[key]
        
        min_max, std_res = self.kfold_stats_min_max(mode = mode, mode2=mode2, clip_mode = clip_mode)
        #print(min_max)
        self.__plot_avg_stats(res,min_max, std_res, mode, clip_mode)
        #add sample counts 
        return res

    
    def __plot_avg_stats(self, overall_stats, min_max, std_res, mode, clip_mode):
        """
        (private) The actual plotting function of 16 plots kfolds stats 
        
        Parameters
        ----------
        overall_stats : dict
            the condition stats
        
        
        """

        direct = os.path.join(self.result_dir, str(self.patient))
        plt.close("all")
        fig, ax = plt.subplots(num_characters, num_characters, figsize=(20, 10))
        patient_acc_stats = np.zeros((2, num_characters*num_characters))
        index = 0
        for character_index_conditon in range(num_characters):
            plot_x = int(character_index_conditon)
            for character_index in range(num_characters):
                plot_y = int(character_index*1.0%num_characters)
                for episode_number in episodes:
                    stats_key = str(character_index) + "#" + str(character_index_conditon) + "#" + str(episode_number)
                    relative_character_prob = overall_stats[stats_key]
                    min_max_temp = min_max[stats_key]
                    std_res_temp = std_res[stats_key]
                    acc = self.plot_by_episode(relative_character_prob ,episode_number ,ax[plot_x, plot_y],  min_max_temp,std_res_temp, mode ="normalize")
                    patient_acc_stats[episode_number-1, index] = acc
                index = index + 1
                ax[plot_x, plot_y].set_title(str(character_index) + " condition on " + str(character_index_conditon))
                ax[plot_x, plot_y].legend()
                ax[plot_x, plot_y].set_ylim([0,1])
        fig.suptitle("patient"+str(self.patient)+"interval conditon on " + mode + " ("+clip_mode+")", fontsize="x-large")
        plt.savefig(os.path.join(direct, "mentest_cond_"+mode+"_"+ clip_mode +"_annotation"+ ".jpg"))
        stats = {"overall_stats":overall_stats,"min_max":min_max_temp, "std_res":std_res_temp, "patient_acc_stats":patient_acc_stats}
        dump_pickle(os.path.join(direct,"mentest_cond_"+mode+"_"+ clip_mode +"_annotation"+ ".pkl"), stats)

    def print_prediction_percentage(self):
        line = ""
        for s in self.stats_list:
            line += str(s.patient) + "," + str(s.k) + ", "+ np.array2string(s.prediction_percentage(), formatter={'float_kind':lambda x: "%.3f" % x})[1:-1]
            line += "\n"
        print(line,  file=open(os.path.join(self.result_dir, self.patient,'chatacter_precentage.txt'), 'w'))
        
    def kfold_stats_min_max(self, mode = "cv_label", mode2 = "normalize", clip_mode = "all"):
        res = {}
        std_res = {}
        x_axis = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for s in self.stats_list:
            if mode == "cv_label":
                ss = s.prob_hist_cond_label(clip_mode=clip_mode)
            else:
                ss = s.prob_hist_cond_predict(clip_mode=clip_mode)
            for key in ss.keys():
                count_ep, bins = np.histogram(ss[key], bins= x_axis)
                if mode2 is "normalize":
                    count_ep = count_ep/len(ss[key])
                acc = give_overall_acc(ss[key])
                if key in res:
                    min_value = count_ep
                    max_value = count_ep
                    min_value = np.minimum(min_value, res[key][0])
                    max_value = np.maximum(max_value, res[key][1])
                    res[key] = [min_value, max_value]
                    #print(res[key].shape)
                else:        
                    min_value = count_ep
                    max_value = count_ep
                    res[key] = [min_value, max_value]
                if key in std_res:
                    std_res[key].append(acc)
                else:
                    std_res[key] = []
                    std_res[key].append(acc)
        for key in std_res.keys():
            #print(key, std_res[key])
            std_res[key] = np.std(std_res[key])
        return res, std_res
    
    def plot_by_episode(self, prob_list,episode_number, ax, min_max=None, std_res=None, mode="normalize"):
        x_axis = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        count_ep, bins = np.histogram(prob_list, bins= x_axis)
        color = ["blue","orange"]
        if mode is "normalize":
            count_ep = count_ep/len(prob_list)
        if min_max is not None:
            min_value = count_ep - min_max[0] 
            max_value = min_max[1] - count_ep
            acc = round(give_overall_acc(prob_list), 2)
            ax.errorbar(np.array(x_axis[:-1]) + 0.01*(episode_number-1), count_ep, yerr=[min_value, max_value], fmt='o')
            ax.plot(x_axis[:-1],count_ep, color = color[episode_number-1],label = "ep"+str(episode_number) + " : %0.2f, %0.2f, %d" % (acc, round(std_res, 2), len(prob_list)))
            return acc
        else:
            ax.plot(x_axis[:-1],count_ep, color = color[episode_number-1],label = "ep"+str(episode_number) + " : sum " + str(sum(count_ep)))
            return 0

    def load_neural_signals(self, mode="all clips"):
        patient = self.patient
        patient_folder = os.path.join(path_to_matlab_generated_movie_data, patient)
        print("patient_folder", patient_folder)
        inmovie_signal = np.load(os.path.join(patient_folder, "features_mat_clean.npy"))
        testclip_signal = np.load(os.path.join(patient_folder, "features_mat_clean2.npy"))
        ## get intervals from stats
        assert len(self.stats_list) >0
        Stats_info = self.stats_list[0]
        assert len(Stats_info.interval_list) > 0 
        interval_list = Stats_info.interval_list
        number_neuron = inmovie_signal.shape[0]
        #max_len_list = {"response time only": 0, "watch time": 0, }
        max_len = 0
        if mode == "all clips":
            for interval in interval_list:
                start1, end1 = interval.get_mentest_datapoint()
                max_len += end1 - start1
            selected_inmovie_signal = np.zeros((number_neuron, max_len))
            selected_testclip_signal = np.zeros((number_neuron, max_len))
            pointer = 0
            for interval in interval_list:
                start1, end1 = interval.get_mentest_datapoint()
                data_len = end1 - start1
                selected_inmovie_signal[:, pointer: (pointer + data_len)] = inmovie_signal[:, start1: end1]
                start2, end2 = interval.get_movie_datapoint()
                selected_testclip_signal[:, pointer: (pointer + data_len)] = testclip_signal[:, start1: end1]
                pointer += data_len
            print("loaded selected in movie and test clip signal", selected_inmovie_signal.shape, selected_testclip_signal.shape)
        if mode == "select jack":
            ### pick all jack for now:
            character_label_board = Stats_info.result_board
            ## which one is jack??? 
            bool_select = (character_label_board == 0)
            new_selected_list = [] ### only select clips jack show up over 50% 
            for interval in interval_list:
                start1, end1 = interval.get_mentest_datapoint()
                if np.sum(bool_select[start1: end1]) > 0.5*(end1 - start1):
                    new_selected_list.append((start1, end1))
                    max_len += end1 - start1

            selected_inmovie_signal = np.zeros((number_neuron, max_len))
            selected_testclip_signal = np.zeros((number_neuron, max_len))
            pointer = 0
            for start1, end1 in new_selected_list:
                data_len = end1 - start1
                selected_inmovie_signal[:, pointer: (pointer + data_len)] = inmovie_signal[:, start1: end1]
                start2, end2 = interval.get_movie_datapoint()
                selected_testclip_signal[:, pointer: (pointer + data_len)] = testclip_signal[:, start1: end1]
                pointer += data_len
            print("loaded selected in movie and test clip signal", selected_inmovie_signal.shape, selected_testclip_signal.shape)

        return selected_inmovie_signal, selected_testclip_signal

    def plot_correlation_mat(self, mode="all clips"):
        selected_inmovie_signal, selected_testclip_signal = self.load_neural_signals()
        inmovie_cor = pearson_corr_coef(selected_inmovie_signal)
        testclip_cor = pearson_corr_coef(selected_testclip_signal)
        cross_cov, cross_corr_coef = cross_coef(selected_inmovie_signal, selected_testclip_signal)
        correlation_maps = {"in movie pearson_corr": inmovie_cor , "test clip pearson_corr": testclip_cor, "cross_cov": cross_cov, "cross_corrcoef": cross_corr_coef}
        self.corr_stats = correlation_maps

        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        cnt = 0
        direct = os.path.join(self.result_dir, str(self.patient))
        for key, val in correlation_maps.items():
            print(val.shape)
            plot_x = cnt%2
            plot_y = int(cnt/2)
            ax1 = ax[plot_x, plot_y]
            im1 = ax1.imshow(val)
            plt.colorbar(im1, ax=ax1)
            ax1.set_title(f"patient {self.patient} {key}")
            cnt +=1
        fig.suptitle("patient"+str(self.patient)+ mode +"correlation map", fontsize="x-large")
        plt.savefig(os.path.join(direct, f"{mode}_memtest_corr.jpg"))

    def plot_correlation_mat2(self, mode="all clips"):
        interval_list = self.interval_list
        num_valid_clips = 0
        valid_interval_list = []
        for interval in interval_list:
            if interval.episode == 1:
                num_valid_clips +=1
                valid_interval_list.append(interval)
        interval_corr_val= np.zeros((num_valid_clips, num_valid_clips))
        #interval_corr_time_lag = np.zero((num_valid_clips, num_valid_clips))
        interval_corr_time_lag = np.zeros(num_valid_clips)
        conv_shift = 300
        self_conv_mat = np.zeros((num_valid_clips, conv_shift))
        for i, interval in enumerate(valid_interval_list):
            time_lag, r_xy, r_xy_arr = interval.self_correlation()
            #print(interval.original_neural_signal)
            interval_corr_time_lag[i] = time_lag
            #self_conv_mat[i, :] = np.pad(r_xy_arr, (0, conv_shift - len(r_xy_arr)), 'constant')
            #self_conv_mat[i, :min(conv_shift, len(r_xy_arr))] = r_xy_arr[:min(conv_shift, len(r_xy_arr))]
            interval_corr_val[i,i] = r_xy
            for j, interval_next in enumerate(valid_interval_list):
                if j != i:
                        _, interval_corr_val[i, j], _ = interval.neural_correlation(interval_next)
        #correlation_maps = {"self conv": self_conv_mat, "cross_cov": interval_corr_val, "corr_time_lag": interval_corr_time_lag}
        correlation_maps = {"cross_cov": interval_corr_val, "corr_time_lag": interval_corr_time_lag}

        fig, ax = plt.subplots(2, 1, figsize=(20, 20))
        cnt = 0
        direct = os.path.join(self.result_dir, str(self.patient))
        for key, val in correlation_maps.items():
            print(val.shape)
            plot_x = cnt%2
            plot_y = int(cnt/2)
            ax1 = ax[plot_x]
            if key == "corr_time_lag":
                plt.plot(val, "*-")
            else:
                im1 = ax1.imshow(val)
                plt.colorbar(im1, ax=ax1)
            ax1.set_title(f"patient {self.patient} {key}")
            cnt +=1
        fig.suptitle("patient"+str(self.patient)+ mode +"correlation map", fontsize="x-large")
        plt.savefig(os.path.join(direct, f"{mode}_memtest_corr.jpg"))

    
    def plot_overall_probabilities(self, person_prob,result_board,label_board, direct, fn='mentest_overall_annotation.jpg'):
        plt.figure(figsize=(40, 10))
        for person_index in sorted(person_prob.keys()):
            one_person_prob = person_prob[person_index]
            plt.subplot(len(person_prob),1,person_index + 1)
            one_person_label = result_board[person_index,:]
            start = 0 
            x_axis = np.divide(range(start,start+len(one_person_prob)), sr_final_movie_data)
            x_axis_2 = np.divide(range(start,start+len(one_person_label)), sr_final_movie_data)
            appearance = one_person_label>0
            response = one_person_label==-1

            plt.plot(x_axis_2, response,color= "black")
            plt.plot(x_axis_2, appearance,color= "r")
            plt.plot(x_axis_2, label_board+2,color= "g")
            plt.plot(x_axis, one_person_prob)
            plt.title("person index: " + str(person_index) +  " mentest plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('Time (s)')
        plt.savefig(os.path.join(direct, fn))
        plt.close('all')

    def draft_function_get_yes_distribution(self):
        res = "/media/yipeng/data/movie/Movie_Analysis/draft_result"
        all_prob = []
        for char_id in range(4):
            character_prob = []
            for s in self.stats_list:
                character_prob.append(s.draft_function_get_yes_distribution(char_id))
            character_prob = np.hstack(character_prob)
            all_prob.append(character_prob)
            fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
        for character in range(4):
            x = int(character/2)
            y = int(character%2)
            #print(x,"**",y)
            axs[x,y].hist(all_prob[character], range=(0,1))
        plt.suptitle(self.patient+"_memtest_distribution.png")
        plt.savefig(os.path.join(res, self.patient+"_memtest_distribution.png"))