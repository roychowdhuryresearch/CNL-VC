import sys
sys.path.append(".")
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np 
import sys
from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
    character_dict, path_to_matlab_generated_movie_data, num_characters, feature_time_width, model_option
from neural_correlation.KnockoutPerformance import KnockoutPerformance, KnockoutKfoldPerformance
from neural_correlation.ModelPerformance import ModelPerformance
import pickle


options = ["reg", "mircowire"]

def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

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

def pipline(patient):
    #folder = "/media/yipeng/data/movie/data_movie_analysis_final/knockout_test_LSTM_KLD"
    #res_folder = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
    folder = "/media/yipeng/data/movie/data_movie_analysis_final/knockout_test_CNN_KLD"
    res_folder = "/media/yipeng/data/movie/data_movie_analysis_final/CNN_multi_2_KLD"
    kfold = 5
    
    region_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn, allow_pickle=True)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_np = np.array(reg)

    microarr_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"channel_data.mat")
    microarr_temp = loadmat(microarr_fn)["channel_reg_info"][0]
    microarr = []
    for m in microarr_temp:
        microarr.append(m[0][0][0])
    microarr_np = np.array(microarr)
    print(microarr_np)
    knockoutKfold_region = KnockoutKfoldPerformance(kfold, patient)
    knockoutKfold_mircowire = KnockoutKfoldPerformance(kfold, patient)
    for k in range(kfold):
        knockout_region = KnockoutPerformance.construct_analysis(folder,patient,k,"reg")
        knockout_region.set_tag(reg_np)
        knockoutKfold_region.add(knockout_region)

        knockout_mircowire = KnockoutPerformance.construct_analysis(folder,patient,k,"mircowire")
        knockout_mircowire.set_tag(microarr_np)
        knockoutKfold_mircowire.add(knockout_mircowire)
    
    knockoutKfold_region_f1, region_label = knockoutKfold_region.get_f1_stats()
    knockoutKfold_region_acc, _ = knockoutKfold_region.get_acc_stats()
    knockoutKfold_region_loss, _ = knockoutKfold_region.get_loss_stats()
    knockoutKfold_region_loss_normalized, _ = knockoutKfold_region.get_loss_stats_normalized()

    knockoutKfold_mircowire_f1, mircowire_label = knockoutKfold_mircowire.get_f1_stats()
    knockoutKfold_mircowire_acc, _ = knockoutKfold_mircowire.get_acc_stats()
    knockoutKfold_mircowire_loss, _ = knockoutKfold_mircowire.get_loss_stats()
    knockoutKfold_mircowire_loss_normalized, _ = knockoutKfold_mircowire.get_loss_stats_normalized()

    trim_idx = get_trim_index(microarr_np)
    reg_trim = reg_np[trim_idx]

    knockoutKfold_region_loss_exp = knockoutKfold_region.get_loss_tag_expand()[trim_idx].T
    knockoutKfold_mircowire_loss_exp = knockoutKfold_mircowire.get_loss_tag_expand()[trim_idx].T

    reg_single = removeDuplicates(reg_np)    
    knockoutKfold_region_mircowire_exp = get_sum(reg_trim, reg_single,knockoutKfold_region_loss_exp,knockoutKfold_mircowire_loss_exp)

    res = {
        "region/neuron":reg_np,
        "region_label" :region_label,
        "region_f1": knockoutKfold_region_f1,
        "region_acc":knockoutKfold_region_acc,
        "region_loss":knockoutKfold_region_loss,
        "region_loss_normalized":knockoutKfold_region_loss_normalized,

        "mircowire/region":microarr_np,
        "mircowire_label":mircowire_label,
        "mircowire_f1":knockoutKfold_mircowire_f1,
        "mircowire_loss":knockoutKfold_mircowire_loss,
        "mircowire_acc":knockoutKfold_mircowire_acc,
        "mircowire_loss_normalized":knockoutKfold_mircowire_loss_normalized,

        "reg_trim":reg_trim,
        "region_loss_exp":knockoutKfold_region_loss_exp,
        "mircowire_loss_exp":knockoutKfold_mircowire_loss_exp,
        "sum_mircowire_loss_region":knockoutKfold_region_mircowire_exp
    }
    fn = os.path.join(res_folder, patient, "knockout_final.pkl")
    dump_pickle(fn, res)

def main(patientNums = patientNums):
    for patient in patientNums:
        pipline(patient)

if __name__ == '__main__':
    main()