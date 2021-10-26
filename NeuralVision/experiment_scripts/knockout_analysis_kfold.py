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

folder = "/media/yipeng/data/movie/Movie_Analysis/knockout_test_LSTM_KLD"
patient = "431"
res = "/media/yipeng/data/movie/Movie_Analysis/draft_result/LSTM_important_regions"
for patient in patientNums:
    kfold = 5
    options = ["reg", "mircowire"]

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


    knockoutKfoldPerformance = KnockoutKfoldPerformance(kfold, patient)
    for k in range(kfold):
        knockoutPerformance = KnockoutPerformance.construct_analysis(folder,patient,k,"reg")
        knockoutPerformance.set_tag(reg_np)
        knockoutKfoldPerformance.add(knockoutPerformance)

    #print(knockoutKfoldPerformance.get_f1_stats())
    #print(knockoutKfoldPerformance.get_acc_stats())
    #print(knockoutKfoldPerformance.get_loss_stats())
    stats, tag = knockoutKfoldPerformance.get_loss_stats_normalized()
    stats_f1, _ = knockoutKfoldPerformance.get_f1_stats()
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    df0 = pd.DataFrame(stats, index= tag, columns=["1","2","3","4"])
    df1 = pd.DataFrame(stats_f1, index= tag, columns=["1","2","3","4"])
    ax = sn.heatmap(df0, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[0])
    ax = sn.heatmap(df1, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[1])
    plt.savefig(os.path.join(res,patient+".jpg"))