import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
from data import patientNums
import os 
from neural_correlation.utilities import *
#stats_folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
stats_folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
#retrain_stats_folder = "/media/yipeng/data/movie/Movie_Analysis/important_retrain/LSTM_retrain"
retrain_stats_folder = "/media/yipeng/data/movie/Movie_Analysis/important_retrain/CNN_important_retrain_0.5"

for this_patient in patientNums:
#for this_patient in ["441"]:
    knockout_stats = load_pickle(os.path.join(stats_folder, this_patient , "knockout_final.pkl"))
    knockout_region = np.array(knockout_stats["region_label"])
    knockout_f1_means = np.mean(knockout_stats["region_f1"], axis=1)
    retrain_patient_folder = os.path.join(retrain_stats_folder, this_patient)
    retrain_regions = os.listdir(retrain_patient_folder)
    retrain_stats_list = []
    region_label = []
    for idx, erased_regions in enumerate(retrain_regions):
        one_retrain_folder = os.path.join(retrain_patient_folder, erased_regions)
        if not os.path.isdir(one_retrain_folder):
            continue
        region_label.append(erased_regions)
        region_stats = KfoldPerformance.generate_bypath(one_retrain_folder)
        one_retrain_stats = np.zeros((3,4))
        one_retrain_stats[0, :]= region_stats.get_character_f1_score()
        total_knockout_f1 = 0 
        for erased_region in erased_regions.split("_"):
            if erased_region.isdigit():
                one_retrain_stats[2,:] = int(erased_region)
            else:
                one_region_knockout = knockout_f1_means[np.where(knockout_region == str(erased_region.strip()))[0]]
                total_knockout_f1 += one_region_knockout
        if total_knockout_f1*1.0/(len(erased_regions.split("_"))-1) >= np.mean(knockout_f1_means):
             one_retrain_stats[1,:] = 1 #nonimportant
        else:
            one_retrain_stats[1,:] = 2 #important
        retrain_stats_list.append(one_retrain_stats)
    one_retrain_stats = one_retrain_stats*0
    knockout_f1 = knockout_stats["region_f1"]
    one_retrain_stats[0,:] = knockout_f1[0,:]
    one_retrain_stats[2,:] = len(knockout_stats["region/neuron"])
    retrain_stats_list.append(one_retrain_stats)
    retrain_stats = np.array(retrain_stats_list)
    print(retrain_stats)
    hori_label = [0,1,2,3]
    # Create a dataset (fake)
    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    df0 = pd.DataFrame(knockout_f1, index= knockout_region, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    ax = sn.heatmap(df0, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[0])
    ax.set(title = "knockout training set") 
    region_label.append("baseline")
    res = {}
    for idx in range(len(retrain_stats)):
        one_retrain_stats = np.zeros((3,4))
        if idx == 1:
            train_with_idx = 0   
        elif idx == 0:
            train_with_idx = 1
        else:
            train_with_idx = idx
        important = int(retrain_stats[train_with_idx,1,0])
        if important == 2:
            axs[1].plot(retrain_stats[idx,0,:], label= region_label[train_with_idx] +"_" +"important",color='blue')
        elif important == 1:
            axs[1].plot(retrain_stats[idx,0,:], label= region_label[train_with_idx] +"_" +"unimportant",color='orange')
        else:
            axs[1].plot(retrain_stats[idx,0,:], label= "baseline",color='green')
        one_retrain_stats[0,:] = retrain_stats[idx,0,:]
        one_retrain_stats[1,:] = retrain_stats[train_with_idx,1,:]
        one_retrain_stats[2,:] = retrain_stats[train_with_idx,2,:]
        res[region_label[train_with_idx]] = one_retrain_stats
    axs[1].set_title("character F1")
    title = this_patient
    plt.suptitle(title)
    plt.legend()
    #plt.show()
    fn_save = os.path.join(retrain_patient_folder, "stats.jpg")
    plt.savefig(fn_save)
    dump_pickle(os.path.join(retrain_patient_folder, "stats.pkl"), res)
    