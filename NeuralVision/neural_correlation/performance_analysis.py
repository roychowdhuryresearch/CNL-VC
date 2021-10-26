import sys
sys.path.append(".")
import os
# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
from torch.nn import Parameter
#from torchvis import util
# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
import seaborn as sn
import pandas as pd
from neural_correlation.utilities import *
import matplotlib.pyplot as plt
import numpy as np
from project_setting import episode_number
if episode_number == 1:
    from data import project_dir, num_characters,character_dict,len_dict, video_analysis_result, yolo_result_dir, character_label_dict
elif episode_number == 2:
    from data2 import project_dir, num_characters,character_dict,len_dict, video_analysis_result, yolo_result_dir, character_label_dict


from sklearn import metrics
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def KLD_loss(pre, label):
    if label == 0:
        y = np.array([1, 0])
    else:
        y = np.array([0, 1])
    yHat = np.log(np.array([pre, 1-pre])+0.000001)
    if y[1] == 0:
        if y[0] == 0 :
            return 0
        return y[0] * (np.log(y[0]) - yHat[0])
    if y[0] == 0:
        return y[1] * (np.log(y[1]) - yHat[1])
    
    res = np.sum(y * (np.log(y) - yHat))
    return res

def get_KLD_loss(pres, labels):
    sum_loss = 0
    for idx in range(len(pres)):
        sum_loss += KLD_loss(pres[idx], labels[idx])
    return sum_loss * 1.0 / len(pres)

def cal_confusion_matrix_and_distribution(pred, label, prob):
    pred = pred.astype(int)
    matrix = np.zeros((2,2))
    tp = []
    fp = []
    tn = []
    fn = []
    for i in range(len(pred)):
        matrix[label[i],pred[i]] = matrix[label[i],pred[i]] + 1
        diff = 2*prob[i]-1
        if label[i] == 1 and pred[i] == 1:
            tn.append(diff)
        elif label[i] == 1 and pred[i] == 0:
            fp.append(diff)
        elif label[i] == 0 and pred[i] == 1:
            fn.append(diff)
        else:
            tp.append(diff)
    matrix = matrix/np.sum(matrix, axis=1)[:,None]
    return matrix, [tp,fp, tn, fn]

def plot_roc(fpr, tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')"452"
    plt.title('ROC curve, (area = %0.2f)', metrics.auc(fpr, tpr))
    plt.legend(loc="lower right")
    plt.show()

def plot_roc_axe(axs,fpr, tpr, threshold):
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    lw = 1
    axs.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve')
    axs.set_xlim([0.0, 1.0])
    axs.set_ylim([0.0, 1.05])
    axs.set_xlabel('False Positive Rate')
    axs.set_ylabel('True Positive Rate')
    auc = metrics.auc(fpr, tpr)
    axs.set_title('a = %0.2f , th = %0.3f' % (auc, threshold))
    axs.legend(loc="lower right")

def plot_hist_axe(axs, hist, title):
    axs.hist(hist,bins=10, range=(-1, 1))
    axs.set_title(title)
    #axs.set_xlim([-1.1, 1.1])
    #axs.set_ylim([0.0, 1.05])

def plot_one_run_stats(fpr, tpr, confusion_matrix, distribution, accuracy, thresholds, percentage ,saved_dir):
    fig, axs = plt.subplots(num_characters, 6,figsize=(24, 20))
    for character_index in confusion_matrix.keys():
        one_character_confusion = confusion_matrix[character_index]
        one_character_tpr = tpr[character_index]
        one_character_fpr = fpr[character_index]
        one_character_dis = distribution[character_index]
        df_cm = pd.DataFrame(one_character_confusion, index = [i for i in ["yes", "no"]],columns = [i for i in ["yes_p", "no_p"]])
        ax = sn.heatmap(df_cm, annot=True, ax=axs[character_index, 0 ])
        ax.set(title = character_dict[character_index] + " acc =  %0.2f " % accuracy[character_index])
        plot_roc_axe(axs[character_index,1], one_character_fpr, one_character_tpr, thresholds[character_index])
        plot_hist_axe(axs[character_index, 2],one_character_"452"dis[2], "tn")
        plot_hist_axe(axs[character_index, 5],one_character_dis[3], "fn")
    plt.savefig(os.path.join(saved_dir, "stats.jpg"))
    plt.close("all")

def get_precentage(character_label,len_dict,character_index):
    length = len_dict[character_index]
    test_label = (np.array(character_label) == 0).sum()
    return test_label *1.0 / length

def flatten_batch_array(arr):
    return np.concatenate(arr, axis=0)

def get_confidence_size(fn, rect):
    with open(fn, "r") as f:
        lines = f.readlines()
    rect_list = []
    confidence_list = []
    for l in lines:
        line_s = l.strip().split(" ")
        if int(line_s[-2]) == 0:
            x0 = int(line_s[0])
            y0 = int(line_s[1])
            x1 = int(line_s[2])
            y1 = int(line_s[3])
            confidence  = float(line_s[5])
            w = x1 - x0
            h = y1 - y0
            rect_list.append(np.array([x0, y0, w, h]))
            confidence_list.append(confidence)
    rect_list = np.array(rect_list)
    dist = distance.cdist(rect_list, rect)
    index = np.argmin(dist)
    size = rect_list[index, 2] * rect_list[index, 3]
    res_conf = confidence_list[index]
    return size, res_conf

# here we need to change the tracking file if we use resnet
def confidence_vs_size(probs, frame_numbers, label, character_label,frame_file_dir = yolo_result_dir, video_analysis_result_fn = video_analysis_result ):
    frame_result = load_pickle(video_analysis_result_fn)
    res_list = []
    for i in range(len(probs)):
        frame_num = frame_numbers[i]
        if label[i] == 0:
            prob = probs[i]
            frame_key = "frame_"+ str(frame_num) + ".jpg.txt"
            for t in frame_result[frame_key]:
                if t[-1] == character_label:
                    x, y, w, h =t[0], t[1], t[2], t[3]
                    size, conf = get_confidence_size(join(yolo_result_dir,frame_key),np.array([[x,y,w,h]]))
                    res_list.append([frame_num, prob, conf ,w*h, size])
    return res_list


def one_result(folder_name,  use_threshold = False):
    fn = os.path.join(folder_name, "model_results.npz")
    print(fn)
    stats = np.load(fn, allow_pickle=True)
    labels = stats["labels"]
    outputs = stats["outputs"]
    frame_name = stats["frame_names"]
    frame_list = flatten_batch_array(frame_name)
    print(frame_list)
    character_stats = {}
    label_record = {}
    accuracy = {}
    confusion_matrix_overall = {}
    distribution_overall = {}
    tpr_overall = {}
    fpr_overall = {}
    thresholds_overall = {}
    prediction_label_overall = {}
    percentage_overall = {}
    model_confidence_stats = {}
    for character_index in range(num_characters):
        one_character_prediction = [] # prob
        one_character_label = [] # label
        one_character_predicted = [] # predicted label _arg _max
        for i in range(len(outputs)):
            batch_label = labels[i]
            batch_output = outputs[i]
            res_label =np.argmax(np.squeeze(batch_label, axis=1), axis=2)
            res_pre = batch_output
            for batch_index in range(len(res_label)):
                one_character_prediction.append(np.exp(res_pre[batch_index,character_index,0]))
                label_temp = res_label[batch_index, character_index]
                character_predicted_temp = np.argmax(res_pre[batch_index,character_index])
                if character_predicted_temp == 2:
                    character_predicted_temp = 1
                if label_temp == 2:
                    label_temp = 1
                one_character_label.append(label_temp)
                one_character_predicted.append(character_predicted_temp)
        fpr, tpr, thresholds = metrics.roc_curve(np.array(one_character_label), np.array(one_character_prediction), pos_label=0)
        one_character_label = np.array(one_character_label)
        one_character_prediction = np.array(one_character_prediction)    
        max_diff_index = np.argmax(tpr - fpr)
        prediction_label = one_character_prediction < thresholds[max_diff_index]
        one_character_predicted = np.array(one_character_predicted)
        if use_threshold:
            one_character_predicted = prediction_label
        else:
            one_character_predicted = one_character_predicted
        confusion_matrix_per_character, histogram = cal_confusion_matrix_and_distribution(one_character_predicted, one_character_label, one_character_prediction)
        
        tpr_overall[character_index] = tpr
        fpr_overall[character_index] = fpr
        thresholds_overall[character_index] = thresholds[max_diff_index]
        accuracy[character_index] = accuracy_score(prediction_label, one_character_label)
        character_stats[character_index] = one_character_prediction
        label_record[character_index] = one_character_label
        confusion_matrix_overall[character_index] = confusion_matrix_per_character
        prediction_label_overall[character_index] = one_character_predicted 
        distribution_overall[character_index] = histogram
        percentage_overall[character_index] = get_precentage(one_character_label, len_dict, character_index)
        #model_confidence_stats[character_index] =  confidence_vs_size(one_character_prediction, frame_list, one_character_label, character_label_dict[character_index])
    saved_dir = folder_name
    plot_one_run_stats(fpr_overall, tpr_overall, confusion_matrix_overall, distribution_overall, accuracy, thresholds_overall, percentage_overall ,saved_dir)
    dump_pickle(join(saved_dir,"confidence.pkl"),model_confidence_stats)
    return character_stats, label_record, prediction_label_overall, confusion_matrix_overall
            

def add_up(base, added):
    for k in added.keys():
        if k in base:
            base[k] = np.concatenate((base[k], added[k]))
        else:
            base[k] = added[k]
def stack_up(base, added):
    for k in added.keys():
        if k in base:
            base[k] = np.vstack((base[k], np.expand_dims(added[k], 0)))
        else:
            base[k] = np.expand_dims(added[k], 0)

def kfold_stats(prediction, label, prob ,confusion_matrice, saved_dir ,num_characters = num_characters, plot_fig = True):
    if plot_fig:
        fig, axs = plt.subplots(2, num_characters,figsize=(4*num_characters, 5))
    character_avg_accuracy = {}
    character_avg_cn_matrix = {}
    character_var_cn_matrix = {}
    character_avg_loss = {}
    for character_index in label.keys():
        character_prob = prob[character_index]
        character_label = label[character_index]
        character_pred = prediction[character_index]
        character_cm = confusion_matrice[character_index]
        avg_confusion_matrix, _ = cal_confusion_matrix_and_distribution(character_pred, character_label, character_prob)
        accuracy = accuracy_score(character_pred, character_label)
        var_character_cm = np.var(character_cm, axis=0)
        character_avg_accuracy[character_index] = accuracy
        character_var_cn_matrix[character_index] = var_character_cm
        character_avg_cn_matrix[character_index] = avg_confusion_matrix

        character_avg_loss[character_index] = get_KLD_loss(character_prob,character_label)

        if plot_fig:
            df_cm = pd.DataFrame(avg_confusion_matrix, index = [i for i in ["yes", "no"]],columns = [i for i in ["yes_p", "no_p"]])
            ax = sn.heatmap(df_cm, annot=True, ax=axs[0, character_index])
            ax.set(title = character_dict[character_index] + " acc =  %0.2f " % accuracy)       
            df_cm = pd.DataFrame(var_character_cm, index = [i for i in ["yes", "no"]],columns = [i for i in ["yes_p", "no_p"]])
            ax = sn.heatmap(df_cm, annot=True, ax=axs[1, character_index])
            ax.set(title = "variance")
    if plot_fig:
        image_dir = os.path.join(saved_dir, "patient_stats.jpg")
        plt.savefig(image_dir)
        plt.close("all")
    return character_avg_accuracy, character_avg_loss ,character_avg_cn_matrix, character_var_cn_matrix


def one_patient_result(folder_name):
    all_folders = os.listdir(folder_name)
    all_character_prediction = {}
    all_character_label = {}
    all_character_prob = {}
    all_confusion_matrix = {}
    for one_folder in all_folders:  
        if one_folder.endswith(".jpg"):
            continue
        folder = os.path.join(folder_name, one_folder)
        character_prob , character_label, character_pred, confusion_matrix = one_result(folder) 
        add_up(all_character_prediction, character_pred)
        add_up(all_character_label, character_label)
        add_up(all_character_prob, character_prob)
        stack_up(all_confusion_matrix, confusion_matrix)
    avg_acc, avg_loss ,avg_cn_matrix, var_cn_matrix = kfold_stats(all_character_prediction, all_character_label, all_character_prob ,all_confusion_matrix, folder_name ,num_characters = num_characters)
    return avg_acc,avg_loss , avg_cn_matrix, var_cn_matrix


def sigle_run_patient_result(folder_name):
    patients = os.listdir(folder_name)
    for one_patient in sorted(patients):
        if one_patient.endswith(".csv"):
            continue
        patient_folder = os.path.join(folder_name, one_patient)
        one_result(patient_folder)


def multi_patient_result(folder_name):
    patients = os.listdir(folder_name)
    prediction_stats = {}
    for one_patient in patients:
        if one_patient.endswith(".csv"):
            continue
        patient_folder = os.path.join(folder_name, one_patient)
        prediction_stats[one_patient] = list(one_patient_result(patient_folder))
    dump_patient_stats(prediction_stats, folder_name)

def multi_patient_erasing(folder_name):
    patients = os.listdir(folder_name)
    for one_patient in patients:
        prediction_stats = {}
        patient_folder = os.path.join(folder_name, one_patient)
        region = os.listdir(patient_folder)
        for r in region:
            region_folder = os.path.join(patient_folder, r)
            if r.endswith(".csv"):
                continue
            prediction_stats[r] = list(one_patient_result(region_folder))
        dump_region_stats(prediction_stats, patient_folder)

def main():
    #folder_name = os.path.join(project_dir, "CNN_result", "CNN_multi_4")
    #multi_patient_result(folder_name)
    #fn = os.path.join(project_dir, "CNN_result", "CNN_multi_70_4","431")
    #one_result(fn, "0")    
    folder_name = os.path.join(project_dir, "CNN_result", "LSTM_erasing_multi_2", "LSTM_retrain")
    multi_patient_erasing(folder_name)
    
    #folder_name = os.path.join(project_dir, "CNN_result", "CNN_erasing_multi_2", "CNN_retrain")
    #multi_patient_erasing(folder_name)
    #folder_name = os.path.join(project_dir, "CNN_result_2_epoch", "basic")
    #sigle_run_patient_result(folder_name)
if __name__ == '__main__':
    main()