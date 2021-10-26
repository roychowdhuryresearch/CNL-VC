import os
import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
from neural_correlation.utilities import load_pickle
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_erasing_multi_check_2/LSTM_retrain"
training_folder = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
patient = "435"

patient_training_folder = os.path.join(training_folder, patient)

patient_training_folder = os.path.join(training_folder, patient)
knockout_stats = load_pickle(os.path.join(patient_training_folder, "knockout_final.pkl"))
region_label = knockout_stats["region_label"]
knockout_stats_f1 = knockout_stats["region_f1"]
retrain_stats_f1 = knockout_stats_f1*0

patient_folder = os.path.join(folder,patient)
for r_idx, r in enumerate(region_label):
    if r == "baseline":
        region_stats = KfoldPerformance.generate_bypath(patient_training_folder)
    else:
        region_folder = os.path.join(patient_folder, r)
        region_stats = KfoldPerformance.generate_bypath(region_folder)
    retrain_stats_f1[r_idx,:] = region_stats.get_character_f1_score()

def plot_comparision(data, hori_label, verti_label, fn, loc, title=None):
    print(data[0])
    print(data[1])
    print(hori_label)
    print(verti_label)
    # Create a dataset (fake)
    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    df0 = pd.DataFrame(data[0], index= verti_label, columns=hori_label)
    df1 = pd.DataFrame(data[1], index= verti_label, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    ax = sn.heatmap(df0, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[0])
    ax.set(title = "knockout training set") 
    ax = sn.heatmap(df1, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[1])
    ax.set(title = "retrain testing set")
    if title is not None:
        plt.suptitle(title)
    plt.show()
    #fn_save = os.path.join(loc, fn+".jpg")
    #plt.savefig(fn_save)
hori_label = [0,1,2,3]
verti_label = region_label
plot_comparision([knockout_stats_f1, retrain_stats_f1], hori_label, verti_label, None, None)