import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
from data import patientNums
import os 
from neural_correlation.utilities import *
stats_folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
#stats_folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
retrain_stats_folder = "/media/yipeng/data/movie/Movie_Analysis/important_retrain/LSTM_retrain"
#retrain_stats_folder = "/media/yipeng/data/movie/Movie_Analysis/important_retrain/CNN_important_retrain"
important_stats = []
non_important_stats = []
for this_patient in patientNums:
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
    #
    # print(retrain_stats.shape)
    hori_label = [0,1,2,3]
    important_stats.append(retrain_stats[0,0,:] - retrain_stats[1,0,:])
    non_important_stats.append(retrain_stats[2,0,:] - retrain_stats[1,0,:])
    #print(retrain_stats[2,0,:])
    #print(retrain_stats[1,0,:])
    #break
   
x = [1, 2, 3, 4]
fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(8, 4))
for i in range(len(important_stats)):
    ax0.plot(x, non_important_stats[i], 'ro')
    ax1.plot(x, important_stats[i], 'bo')
ax0.set_ylim([0,1])
ax1.set_ylim([0,1])
ax0.set_title("non_important")
ax1.set_title("important")
plt.suptitle("LSTM")
plt.show()

'''
def softmax_numpy(scores):
  return np.exp(scores)/sum(np.exp(scores))

x = [1, 2, 3, 4]
for i in range(len(important_stats)):
    print(important_stats[i])
    plt.scatter(x, important_stats[i], alpha=0.9, s=4)
plt.suptitle("CNN important - nonimportant")
plt.show()

plt.hist(np.concatenate(important_stats))
plt.show()
'''