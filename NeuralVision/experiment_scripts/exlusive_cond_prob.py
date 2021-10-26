import sys
import numpy as np
import os
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pylab as plt
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
sys.path.append(".")
sys.path.insert(0, "/media/yipeng/data/movie/Movie_Analysis")
from  neural_correlation.utilities import *
from scipy.io import loadmat, savemat
from data import patientNums, path_to_matlab_generated_movie_data, num_characters
#result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
#result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_70-30"
from  neural_correlation.KfoldStats import KfoldStats
k = 5


def main():
    path = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
    output_path = "/media/yipeng/data/movie/Movie_Analysis/final_result_outputs"
    stats_num1 = {}
    stats_num3_num2 = {}
    stats_num4_num1 = {}
    for patient in patientNums:
        kfoldStats = KfoldStats(patient, path, 5)
        if len(kfoldStats.stats_list) > 0:
            patient_name = "p_"+patient
            stats_num3_num2[patient_name] = kfoldStats.exclusive_coactivation()
            stats_num1[patient_name] = kfoldStats.exclusive_coactivation_count()
            stats_num4_num1[patient_name] = kfoldStats.clip4_div_clip1()
    if "LSTM" in path:
        model = "LSTM"
    if "CNN" in path:
        model = "CNN"
    res = {"exclusive_prob":stats_num3_num2, "count":stats_num1, "stats_num4_num1":stats_num4_num1 }
    output_folder = os.path.join(output_path, model)
    fn = os.path.join(output_folder,"figure4_si_exclusive_association.mat")
    savemat(fn,res)    

if __name__ == '__main__':
    main()
