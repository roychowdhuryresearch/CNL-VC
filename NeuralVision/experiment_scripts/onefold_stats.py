import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance
import matplotlib
import matplotlib.pyplot as plt
import os
from data import sr_final_movie_data, num_characters
import dill

path = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD/431/1/model_results.npz"
path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_70-30/431/model_results.npz"
path_s = path.split("/")
patient_name = path_s[-3]
model_option = path_s[-4].split("_")[0]
k = path_s[-2]
model_info = np.load(path, allow_pickle=True)
onefold_model = ModelPerformance.generate(k, patient_name, model_option, model_info['outputs'],model_info['labels'], model_info['frame_names'])
print(onefold_model.get_character_correlation())
'''
#stats outputs 
overall_confusion_mat = onefold_model.get_confusion_mat()
character_confusion_mat = onefold_model.get_character_confusion_mat()
overall_acc = onefold_model.get_overall_accuracy()
character_acc = onefold_model.get_character_accuracy()
stats_board, label_board = onefold_model.get_tpfp() 


print(overall_confusion_mat)
print(character_confusion_mat)
print(overall_acc)
print(character_acc)

def plot_tpfp(tpfp, label ,direct = None, fn= 'fp_tp.jpg'):
    plt.figure(figsize=(12, 10))
    for character_index in range(num_characters):
        one_character_tpfp = tpfp[character_index]
        one_character_label = label[character_index]
        plt.subplot(len(tpfp),1,character_index + 1)
        #one_person_tpfp = one_person_tpfp[:2000]
        start = 0
        plt.scatter(np.divide(range(start,start+len(one_character_tpfp)), sr_final_movie_data), one_character_tpfp==1, s=5, c='tab:blue', label="TP",alpha=1)
        plt.scatter(np.divide(range(start,start+len(one_character_tpfp)), sr_final_movie_data), -.1 + np.asarray(one_character_tpfp==2, dtype = float) , s=5, c='tab:orange',label="FP",alpha=1)
        plt.scatter(np.divide(range(start,start+len(one_character_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_character_label==0, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
        plt.title("person index: " + str(character_index) +  " TP FP plot")
        plt.legend()
        plt.ylim(0.5, 2)
        if character_index is not 3:
            plt.xticks([], [])
    plt.xlabel('Time (s)')
    plt.show()
    #plt.savefig(os.path.join(direct, fn))
    #plt.close('all')

plot_tpfp(stats_board, label_board) 
'''