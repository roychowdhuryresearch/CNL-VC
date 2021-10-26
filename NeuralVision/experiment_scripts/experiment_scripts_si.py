import sys
sys.path.append(".")
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
sys.path.append("./movietag_processing")
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np 
import sys
from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
    character_dict, path_to_matlab_generated_movie_data, num_characters, feature_time_width, model_option
from neural_correlation.KnockoutPerformance import KnockoutPerformance, KnockoutKfoldPerformance
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
from neural_correlation.utilities import load_pickle, dump_pickle
from neural_correlation.MemtestKfoldStats import KfoldStats
from movietag_processing.EpisodeStats import EpisodeStates

'''
utilities
'''
def removeDuplicates(arr): 
    res = []
    res_set = set()
    for a in arr: 
        if a not in res_set:
            res_set.add(a)
            res.append(a)
    return res

def get_trim_index(tag1):
    jump_diff = np.diff(tag1)
    jump_index = np.where(jump_diff)[0]
    jump_index = np.append(jump_index, len(tag1)-1) 
    return jump_index

def get_sum(reg,reg_single,region_loss,mircowire_loss):
    sum_mircowire = region_loss * 0
    for k in range(len(reg_single)):
        loc = np.where(reg==reg_single[k])[0]
        if len(loc) == 1:
            sum_mircowire[:,loc] = mircowire_loss[:,loc]    
            continue
        sum_mircowire[:,loc] = np.tile(np.expand_dims(np.sum(mircowire_loss[:,loc], axis = 1), axis = 1), (1, len(loc)))
    return sum_mircowire

def parse_key(char, cond_char):
    return str(char) + "|"+ str(cond_char)

def normalize(prob, num_character):
    res = np.zeros((num_character, num_character))
    for cond_char in range(num_character):
        for char in range(num_character):
            if cond_char == char:
                continue
            res[char, cond_char] = prob[parse_key(char,cond_char)]
    res_normalize = np.divide(res, np.sum(res,axis=0))
    return res_normalize 

def two_step(prob):
    res = np.zeros((len(prob), len(prob)))
    for cond_char in range(len(prob)):
        for char in range(len(prob)):
            if char == cond_char:
                res[char, cond_char] = 1 
                continue  
            walk_prob = prob[char,cond_char]
            for walk_char in range(len(prob)):      
                if walk_char == char or walk_char == cond_char :
                    continue
                walk_prob = walk_prob + 1.0*prob[char, walk_char] * prob[walk_char,cond_char]
            res[char,cond_char] = walk_prob 
    return res

def character_confusion_all_participates(path, output_folder):
    patientNums = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
    stats = np.zeros((len(patientNums), 4, 4))
    for p_idx , p in enumerate(patientNums):
        path_p = os.path.join(path, p )
        kfoldPerformance = KfoldPerformance.generate_bypath(path_p)
        character_confusion_mat = kfoldPerformance.get_character_confusion_mat()
        character_f1_score = kfoldPerformance.get_character_f1_score()
        character_acc= kfoldPerformance.get_character_accuracy()
        character_recall = character_confusion_mat[:,0,0] /(character_confusion_mat[:,0,0] + character_confusion_mat[:,1,0])
        character_precision = character_confusion_mat[:,0,0] /(character_confusion_mat[:,0,0] + character_confusion_mat[:,0,1])
        stats[p_idx, 0,:] = character_recall
        stats[p_idx, 1,:] = character_precision
        stats[p_idx, 2,:] = character_acc
        stats[p_idx, 3,:] = character_f1_score
    fn = os.path.join(output_folder,"character_confusion_all_participates.mat")
    #res = {'stats':stats}
    #savemat(fn,res)
    print(np.round(np.std(stats, axis= 0)*100, 2))
    print(np.round(np.mean(stats, axis= 0)*100, 2))
    
def main():
    
    """
    Usage:
        you basically need 4 external folders, 
        "LSTM_multi_2_KLD, CNN_multi_2_KLD, knockout_test_LSTM_KLD, knockout_test_CNN_KLD"
        First two is for Figure2 and Figure4
        Second one is for Figure3 
        How to run this file:
        1. you need to change the project_dir path in data/__init__.py to the current top level
        2. you change the path in the following lines.
        3. all the necessary inputs of running file are in the input_folder/
        4. memtest will be done soon
    """
    
    project_folder = "/Users/yipengzhang/Documents/GitHub/Movie_Analysis/"
    project_folder = "/media/yipeng/data/movie/Movie_Analysis"
    #path = "/Users/yipengzhang/Desktop/movie_analysis_data/CNN_multi_2_KLD"
    #path = "/media/yipeng/data/movie/data_movie_analysis_final/CNN_multi_2_KLD"
    path = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
    path = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
    result_folder = os.path.join(project_folder,"final_result_outputs")
    if "LSTM" in path:
        model_option = "LSTM"
    else:
        model_option = "CNN"
    output_folder = os.path.join(result_folder,model_option)
    character_confusion_all_participates(path, output_folder)



if __name__ == "__main__":
    main()