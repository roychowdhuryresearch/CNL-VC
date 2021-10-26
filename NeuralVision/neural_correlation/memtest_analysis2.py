
import sys
sys.path.append(".")
from data import patientNums, path_to_matlab_generated_movie_data, num_characters
import os 
from utilities import *
import scipy as sp
import scipy.io as sio
import math
result_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD"


def get_cv_label(ep1_dir):
    res = {}
    res[1] = set(load_pickle(os.path.join(ep1_dir , "1.pkl")))
    res[0]  = set(load_pickle(os.path.join(ep1_dir , "0.pkl")))
    res[2]  = set(load_pickle(os.path.join(ep1_dir , "2.pkl" )))
    res[3]  = set(load_pickle(os.path.join(ep1_dir , "3.pkl")))
    return res


def check(current_frame, ep_result):
    #print(current_frame)
    res = np.zeros(num_characters)
    for character_idx in range(num_characters):
        res[character_idx] = int(current_frame in ep_result[character_idx])
    #print(res)
    return res

def find_closed_4_factor(num):
    num_4 = round(num/4)
    num = num_4*4
    return num

def get_result_board(exp_file, triggle_file):
    ep1_result = get_cv_label("/media/yipeng/toshiba/movie/Movie_Analysis/Character_TimeStamp_resnet")
    ep2_result = get_cv_label("/media/yipeng/toshiba/movie/Movie_Analysis/time_stamp_2")
    
    triggle_file_start = np.array(triggle_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
    triggle_file_end = np.array(triggle_file["triggers"]["trigger_clip_end_ttl"][0][0][0])
    start_time_stamp = triggle_file_start[0]
    end_time_stamp = triggle_file_end[-1]
    offset = 2520
    start_sample_idx = int((start_time_stamp - offset)*7.5)
    end_sample_idx = int((end_time_stamp- offset)*7.5)
    total_sample = end_sample_idx - start_sample_idx 
    result_board = np.zeros((num_characters, total_sample)) -1
    label_board = np.zeros(total_sample)

    for i in range(exp_file.shape[0]):
        one_line = exp_file[i,:]
        #time_interval = ((np.array([triggle_file_start[i], triggle_file_end[i]]) - triggle_file_start[0])*7.5).astype(int)
        time_interval = (np.array([triggle_file_start[i], triggle_file_end[i]])*7.5 - offset * 7.5 -start_sample_idx).astype(int) 
        #time_interval = ((np.array([int(triggle_file_start[i]*7.5)- 24 - start_sample_idx, triggle_file_end[i]]) - triggle_file_start[0])*7.5).astype(int)
        label = one_line[1]
        label_board[time_interval[0]:time_interval[1]] = label 
        sample_diff = time_interval[1]-time_interval[0]
        print(time_interval[1],time_interval[0])
        result_interval_board = np.zeros((num_characters, sample_diff))
        movie_frame_start = one_line[5] * 30 
        #print(movie_frame_start)
        movie_frame_start = find_closed_4_factor(movie_frame_start)
        for sample_stp in range(sample_diff):
            current_frame = movie_frame_start + sample_stp * 4
            if label == 1: ### first movie
                #print("first")
                check_res = check(current_frame, ep1_result)
            else:
                #print("second")
                check_res = check(current_frame, ep2_result)
            #print
            result_interval_board[:,sample_stp] = check_res
        result_board[:,time_interval[0]:time_interval[1]] = result_interval_board   
        #if i == 2:
        #    break
    return result_board, label_board   


def get_result_board2(exp_file0, triggle_file):
    ### Note here in the label_board, the label are extened from 1, 2 to (1,3),(2,4). Where 1,2 means in movie clip time, 3,4 means repsonse time
    ep1_result = get_cv_label("/media/yipeng/toshiba/movie/Movie_Analysis/Character_TimeStamp_resnet")
    ep2_result = get_cv_label("/media/yipeng/toshiba/movie/Movie_Analysis/time_stamp_2")
    
    exp_file = exp_file0['TRIAL_ID']
    triggle_file_start = np.array(triggle_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
    triggle_file_end = np.array(triggle_file["triggers"]["trigger_clip_end_ttl"][0][0][0])
    response_arr = exp_file0["trial_struct"]["resp_time"][0]
    response_time = np.array([response_arr[i][0][0] for i in range(len(response_arr))])
    print(response_time)

    start_time_stamp = triggle_file_start[0]
    ### note change of board size too, add the last clip response window
    end_time_stamp = triggle_file_end[-1] + response_arr[-1]
    offset = 2520
    start_sample_idx = int((start_time_stamp - offset)*7.5)
    end_sample_idx = int((end_time_stamp- offset)*7.5)
    total_sample = end_sample_idx - start_sample_idx 
    result_board = np.zeros((num_characters, total_sample)) - 1  ## this is character occurance board
    label_board = np.zeros(total_sample)

    for i in range(exp_file.shape[0]):
        one_line = exp_file[i,:]
        #time_interval = ((np.array([triggle_file_start[i], triggle_file_end[i]]) - triggle_file_start[0])*7.5).astype(int)
        time_interval_clip = (np.array([triggle_file_start[i], triggle_file_end[i]])*7.5 - offset * 7.5 -start_sample_idx).astype(int) 
        time_interval_resp = (np.array([triggle_file_end[i], triggle_file_end[i] + response_time[i]])*7.5 - offset * 7.5 -start_sample_idx).astype(int) 
        label = one_line[1]
        label_board[time_interval_clip[0]:time_interval_clip[1]] = label 
        label_board[time_interval_resp[0]:time_interval_resp[1]] = label + 2
        ### add response tinme
        sample_diff = time_interval_resp[1] - time_interval_clip[0]
        result_interval_board = np.zeros((num_characters, sample_diff))
        movie_frame_start = one_line[5] * 30 
        #print(movie_frame_start)
        movie_frame_start = find_closed_4_factor(movie_frame_start)
        for sample_stp in range(sample_diff):
            current_frame = movie_frame_start + sample_stp * 4
            if label %2 == 1: ### first movie, in clip or response time
                #print("first")
                check_res = check(current_frame, ep1_result)
            else:
                #print("second")
                check_res = check(current_frame, ep2_result)
            #print
            result_interval_board[:,sample_stp] = check_res
        result_board[:,time_interval_clip[0]:time_interval_resp[1]] = result_interval_board   
        #if i == 2:
        #    break
    return result_board, label_board 

def cut_prob_result(patient_prob, start_sample_idx, end_sample_idx):
    res = {}
    for character_index in patient_prob.keys():
        res[character_index] = patient_prob[character_index][start_sample_idx:end_sample_idx]
    return res


def main():
    for patient in patientNums:
        patient_folder = os.path.join(path_to_matlab_generated_movie_data, patient)
        file_list = os.listdir(patient_folder)
        ts_file = "/media/yipeng/toshiba/movie/Movie_Analysis/data/"+patient+"/triggers_"+patient+"series1_ht.mat"
        if not os.path.exists(ts_file):
            continue
        ts_info = sio.loadmat(ts_file)
        ts_offset = ts_info["br_triggers"].squeeze()[0] - ts_info["movie_times"].squeeze()[0]
        print(patient,ts_offset)
        #continue
        for folder in file_list:
            if "memtest1" in folder:
                men_test_folder = os.path.join(patient_folder, folder)
                exp_fn = "psychophysics/" + "EXP_" +  patient+ "memtest1.mat"
                triggle_fn =  "triggers/" + "triggers_" +  patient+ "memtest1.mat"
                #exp_file = sio.loadmat(os.path.join(men_test_folder, exp_fn))['TRIAL_ID']
                ### note changes of defintion of exp_file
                exp_file = sio.loadmat(os.path.join(men_test_folder, exp_fn))
                triggle_file = sio.loadmat(os.path.join(men_test_folder, triggle_fn))
                triggle_file_start = np.array(triggle_file["triggers"]["trigger_clip_start_ttl"][0][0][0])
                triggle_file_end = np.array(triggle_file["triggers"]["trigger_clip_end_ttl"][0][0][0])
                start_time_stamp = triggle_file_start[0]

                response_arr = exp_file["trial_struct"]["resp_time"][0]
                response_time = np.array([response_arr[i][0][0] for i in range(len(response_arr))])
                end_time_stamp = triggle_file_end[-1] + response_time[-1]
                offset = 2520
                ### modified for problematic case: pateint 436, start, end_sample idx: 4132.927217525292 12142.016406411345
                start_sample_idx = int((start_time_stamp - ts_offset - offset)*7.5 + 0.9999/7.5)
                end_sample_idx = int((end_time_stamp - ts_offset - offset)*7.5) 
                print("start, end_sample idx:", (start_time_stamp- ts_offset - offset)*7.5, (end_time_stamp - ts_offset - offset)*7.5)
                result_board, label_board = get_result_board2(exp_file, triggle_file)
                #print("trigger clip, start, ent ttl", triggle_file_start, triggle_file_end)
        print("shape of result board, label board", result_board.shape, label_board.shape)
        for k in range(5):
            patient_result_folder = os.path.join(result_dir, patient, str(k))
            patient_prob = load_pickle(os.path.join(patient_result_folder, "memory_test_prob.pkl"))
            print(start_sample_idx, end_sample_idx )
            patient_prob_cut = cut_prob_result(patient_prob,start_sample_idx, end_sample_idx)
            print("prob_cut_shape", patient_prob_cut[0].shape)
            #plot_prob(patient_prob_cut,patient_result_folder)
            plot_overall_prob(patient_prob_cut,result_board,label_board,patient_result_folder)
            
            plot_prob_hist(result_board, label_board, patient_prob_cut,patient_result_folder, fn=f'mentest_proboverall.jpg')
        
if __name__ == '__main__':

    main()
