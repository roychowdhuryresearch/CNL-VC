################################################################################
# utilities
# just some random functions
# borrow from the movie grouping project
################################################################################
import sys
sys.path.append(".")
sys.path.append("..")
import cv2
import random
import os
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from project_setting import episode_number
from Interval import Interval
if episode_number == 1:
    from data import character_dict, frame_dir, sr_final_movie_data,num_characters, path_to_cnn_result
elif episode_number == 2:
    from data2 import character_dict, frame_dir, sr_final_movie_data,num_characters,path_to_cnn_result
import more_itertools as mit
import csv

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        os.mkdir(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)
def join(a, b):
    return os.path.join(a,b)

def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        #print(fn)
    return lookup

def reverse_key_list_dict(list_dict):
    res = {}
    for k in list_dict.keys():
        values = list_dict[k]
        for v in values:
            res[v] = k
    return res

def sort_filename(frame_list: list):
    ## sort the filename by \u201cframeXX\u201d
    frame_dir = {}
    for f in frame_list:
        frame_number = int(f.split("frame_")[1].split(".")[0])
        frame_dir[frame_number] = f
    frame_dir_key = list(frame_dir.keys())
    frame_dir_key.sort()
    sorted_filename = []
    for frame_n in frame_dir_key:
        sorted_filename.append(frame_dir[frame_n])
    return sorted_filename
    
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] +20, c1[1] - t_size[1] - 3 +20
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] + 20), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def read_scene_list(scene_list_fn):
    res = {}
    with open(scene_list_fn) as f:
        lines = f.readlines()
    for line in lines:
        line_s = line.strip().split("|")
        if len(line_s) < 5:
            continue
        start = line_s[2].strip()
        end = line_s[4].strip()
        if start.isdigit() and end.isdigit():
            res[int(start)] = int(end)
    return res 

def removeDuplicates(arr): 
    res = []
    res_set = set()
    for a in arr: 
        if a not in res_set:
            res_set.add(a)
            res.append(a)
    return res

def get_confusion(res_label, res_pre,batch_index, person_index):
    res = np.zeros((3,3)) 
    res[res_label[batch_index][person_index],res_pre[batch_index][person_index]] = 1
    return res

def plot_frame(frame_list, saved_dir, frame_dir = frame_dir):
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    for frame in frame_list:
        if frame <= 0:
            continue
        file_name = "frame_"+str(frame)+".jpg"
        shutil.copyfile(os.path.join(frame_dir,file_name), os.path.join(saved_dir,file_name))

def plot_false_frame(character_frame_dict, saved_dir):
    clean_folder(saved_dir)
    if not os.path.exists(saved_dir):
        os.mkdir(saved_dir)
    for character in character_frame_dict.keys():
        frame_list = np.array(character_frame_dict[character])
        saved_dir_c = os.path.join(saved_dir, character_dict[character])
        plot_frame(frame_list, saved_dir_c)

def plot_var_confusion_matrix(all_person_confusion, saved_dir, filename= "confusion_matrix.jpg"):	
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    all_confusion = np.zeros((3,3)).astype(int)
    for person_index in all_person_confusion.keys():
        one_person_confusion = all_person_confusion[person_index]
        all_confusion = all_confusion+one_person_confusion
        df_cm = pd.DataFrame(one_person_confusion, index = [i for i in ["yes", "no", "dnk"]],columns = [i for i in ["yes_p", "no_p", "dnk_p"]])
        ax = sn.heatmap(df_cm, annot=True, ax=axs[int(person_index/3), person_index%3])
        ax.set(title = character_dict[person_index])
    df_cm = pd.DataFrame(all_confusion/np.sum(all_confusion,axis=1)[:,None], index = [i for i in ["yes", "no", "dnk"]],
                        columns = [i for i in ["yes_p", "no_p", "dnk_p"]])
    ax = sn.heatmap(df_cm, annot=True, ax=axs[1,1])
    ax.set(title = 'all')
    plt.savefig(os.path.join(saved_dir, filename))


def plot_confusion_matrix(all_person_confusion, saved_dir, filename= "confusion_matrix.jpg"):	
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    all_confusion = np.zeros((3,3)).astype(int)
    for person_index in all_person_confusion.keys():
        one_person_confusion = all_person_confusion[person_index]
        all_confusion = all_confusion+one_person_confusion
        df_cm = pd.DataFrame(one_person_confusion/np.sum(one_person_confusion, axis=1)[:,None], index = [i for i in ["yes", "no", "dnk"]],columns = [i for i in ["yes_p", "no_p", "dnk_p"]])
        ax = sn.heatmap(df_cm, annot=True, ax=axs[int(person_index/3), person_index%3])
        correct =  (one_person_confusion[0,0] + one_person_confusion[1,1])*1.0
        #print(correct)
        acc = correct/np.sum(one_person_confusion)
        #print(acc)
        ax.set(title = character_dict[person_index] + "_" +str(round(acc, 2)))
    df_cm = pd.DataFrame(all_confusion/np.sum(all_confusion,axis=1)[:,None], index = [i for i in ["yes", "no", "dnk"]],
                        columns = [i for i in ["yes_p", "no_p", "dnk_p"]])
    ax = sn.heatmap(df_cm, annot=True, ax=axs[1,1])
    ax.set(title = 'all')
    plt.savefig(os.path.join(saved_dir, filename))


def plot_tpfp(person_tpfp, person_label ,direct, fn= 'fp_tp.jpg'):
    plt.figure(figsize=(12, 10))
    for person_index in sorted(person_tpfp.keys()):
        one_person_tpfp = person_tpfp[person_index]
        one_person_label = person_label[person_index]
        plt.subplot(len(person_tpfp),1,person_index + 1)
        #one_person_tpfp = one_person_tpfp[:2000]
        start = 0
        plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), one_person_tpfp==1, s=5, c='tab:blue', label="TP",alpha=1)
        plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), -.1 + np.asarray(one_person_tpfp==2, dtype = float) , s=5, c='tab:orange',label="FP",alpha=1)
        plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_person_label==1, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
        plt.title("person index: " + str(person_index) +  " TP FP plot")
        plt.legend()
        plt.ylim(0.5, 2)
        if person_index is not 3:
            plt.xticks([], [])
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(direct, fn))
    plt.close('all')

def plot_validation_accuarcy(val_acc_all,this_patient,best_epoch, direct):
    plt.figure()
    plt.plot(np.arange(1,len(val_acc_all)+1), val_acc_all)
    plt.title(this_patient + '  Overall Accuracy - Best Epoch : ' + str(best_epoch))
    plt.xlabel('Epoch Number')
    plt.savefig(os.path.join(direct, 'overall_accuracy.jpg'))
    plt.close('all')

def get_stats(res_label, res_pre,batch_index, person_index):
    res = np.zeros(8)  #yes correct, yes_wrong, no_cor, no_wrong, dono_cor, dono_wrong, fp, tp
    if res_label[batch_index][person_index] == 0:
        if res_pre[batch_index][person_index] == 0:
            res[0] = 1 
            res[7] = 1 # TP
        else:
            res[1] = 1
    elif res_label[batch_index][person_index] == 1:
        if res_pre[batch_index][person_index] == 1:
            res[2] = 1
        else:
            res[3] = 1
            if res_pre[batch_index][person_index] == 0:
                res[6] = 1  #FP
    else:
        if res_pre[batch_index][person_index] == 2:
            res[4] = 1
        elif res_pre[batch_index][person_index] == 0:
            res[7] = 1 # TP
        res[5] = 1
    return res


def dump_region_stats(region_stats, saved_dir):
    saved_fn = os.path.join(saved_dir, "resgion_stats.csv")
    #print(region_stats)
    with open(saved_fn, mode='w') as f:
        fcsv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fieldnames = ['patient', "jack",'bill', "chloe","terro", "jack_loss",'bill_loss', "chloe_loss","terro_loss","jack_y","jack_n", 'bill_y', "bill_n","chloe_y", "chloe_n","terro_y", "terro_n" ,"jack_y_var","jack_n_var","bill_y_var", "bill_n_var","chloe_y_var", "chloe_n_var","terro_y_var", "terro_n_var"]
        fcsv.writerow(fieldnames)
        for k in region_stats.keys():
            avg_acc, avg_loss ,avg_cn_matrix, var_cn_matrix = region_stats[k]
            line = [k]
            for character_index in sorted(avg_acc.keys()):
                line.append(avg_acc[character_index])
            for character_index in sorted(avg_loss.keys()):
                line.append(avg_loss[character_index])
            for character_index in sorted(avg_cn_matrix.keys()):
                line.append(avg_cn_matrix[character_index][0, 0])
                line.append(avg_cn_matrix[character_index][1, 1])
            for character_index in sorted(var_cn_matrix.keys()):
                line.append(var_cn_matrix[character_index][0, 0])
                line.append(var_cn_matrix[character_index][1, 1])
            fcsv.writerow(line)

def dump_patient_stats(region_stats, saved_dir):
    saved_fn = os.path.join(saved_dir, "patient_stats.csv")
    with open(saved_fn, mode='w') as f:
        fcsv = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        fieldnames = ['patient', "jack",'bill', "chloe","terro","jack_y","jack_n", 'bill_y', "bill_n","chloe_y", "chloe_n","terro_y", "terro_n" ,"jack_y_var","jack_n_var","bill_y_var", "bill_n_var","chloe_y_var", "chloe_n_var","terro_y_var", "terro_n_var"]
        fcsv.writerow(fieldnames)
        for k in region_stats.keys():
            avg_acc, avg_cn_matrix, var_cn_matrix = region_stats[k]
            line = [k]
            for character_index in sorted(avg_acc.keys()):
                line.append(avg_acc[character_index])
                print(avg_acc[character_index])
            for character_index in sorted(avg_cn_matrix.keys()):
                print(avg_cn_matrix[character_index].shape)
                print(avg_cn_matrix[character_index])
                line.append(avg_cn_matrix[character_index][0, 0])
                line.append(avg_cn_matrix[character_index][1, 1])
            for character_index in sorted(var_cn_matrix.keys()):
                line.append(var_cn_matrix[character_index][0, 0])
                line.append(var_cn_matrix[character_index][1, 1])
            fcsv.writerow(line)

def dont_know_analysis(stats, character_index ,frame_num, to_dir ,from_dir = frame_dir):
    if stats[0] == 0 and stats[7] == 1:
        ## dnk predicted as yes
        to_dir_folder = os.path.join(to_dir, "dnk_as_yes")
        if not os.path.exists(to_dir_folder):
            os.mkdir(to_dir_folder)
        to_dir_folder = join(to_dir_folder, str(character_index))
        if not os.path.exists(to_dir_folder):
            os.mkdir(to_dir_folder)
        copy_one_image(from_dir, frame_num, to_dir_folder)
    elif stats[5] == 1:
        ## dnk predicted as No
        to_dir_folder = os.path.join(to_dir, "dnk_as_no")
        if not os.path.exists(to_dir_folder):
            os.mkdir(to_dir_folder)
        to_dir_folder = join(to_dir_folder, str(character_index))
        if not os.path.exists(to_dir_folder):
            os.mkdir(to_dir_folder)
        copy_one_image(from_dir, frame_num, to_dir_folder)

def copy_one_image(from_dir, frame_number, to_dir):
    if frame_number <= 0:
        return
    image_fn = "frame_"+str(frame_number)+".jpg"
    copy_to = join(to_dir,image_fn)
    copy_from = join(from_dir, image_fn)
    shutil.copyfile(copy_from, copy_to)

def cal_f1_score(tp,fp,fn):
    p = 1.0*tp/(tp+fp+10e-20)
    r = 1.0*tp/(tp+fn+10e-20)
    f1 = p*r*2/(p+r+10e-20)
    #print("p",p,"r",r)
    #print("f1",f1)
    return f1

def plot_prob(person_prob,direct, fn= 'men_test_prob.jpg'):
    plt.figure(figsize=(12, 10))
    for person_index in sorted(person_prob.keys()):
        one_person_prob = person_prob[person_index]
        plt.subplot(len(person_prob),1,person_index + 1)
        #one_person_tpfp = one_person_tpfp[:2000]
        start = 0 
        plt.plot(np.divide(range(start,start+len(one_person_prob)), sr_final_movie_data), one_person_prob)
        plt.title("person index: " + str(person_index) +  " men_test_prob plot")
        if person_index is not 3:
            plt.xticks([], [])
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(direct, fn))
    plt.close('all')

def plot_overall_prob(person_prob,result_board,label_board, direct, fn='mentest_overall.jpg'):
    plt.figure(figsize=(40, 10))
    for person_index in sorted(person_prob.keys()):
        one_person_prob = person_prob[person_index]
        plt.subplot(len(person_prob),1,person_index + 1)
        one_person_label = result_board[person_index,:]
        start = 0 
        x_axis = np.divide(range(start,start+len(one_person_prob)), sr_final_movie_data)
        x_axis_2 = np.divide(range(start,start+len(one_person_label)), sr_final_movie_data)
        appearance = one_person_label>0
        response = one_person_label==-1

        plt.plot(x_axis_2, response,color= "black")
        plt.plot(x_axis_2, appearance,color= "r")
        plt.plot(x_axis_2, label_board+2,color= "g")
        plt.plot(x_axis, one_person_prob)
        plt.title("person index: " + str(person_index) +  " mentest plot")
        if person_index is not 3:
            plt.xticks([], [])
    plt.xlabel('Time (s)')
    plt.savefig(os.path.join(direct, fn))
    plt.close('all')


def plot_prob_hist_all(prob, label, direct, fn='mentest_proboverall.jpg'):
    fig, ax = plt.subplots(2, 2, tight_layout=True)
    x_axis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for character_index in sorted(prob.keys()):
        character_label = label
        ep1_loc = np.where(character_label == 1)[0]
        ep2_loc = np.where(character_label == 2)[0]
        character_prob = prob[character_index]
        ep1_prob = character_prob[ep1_loc]
        ep2_prob = character_prob[ep2_loc]
        #ep1_prob = ep1_prob[np.where(ep1_prob>0.3)[0]]
        #ep2_prob = ep2_prob[np.where(ep2_prob>0.3)[0]]
        count_ep1, _ = np.histogram(ep1_prob, bins= 10, range=(0.1, 1.01))
        count_ep2, _ = np.histogram(ep2_prob, bins= 10, range=(0.1, 1.01))
        plot_x = int(character_index*1.0/2)
        plot_y = int(character_index*1.0%2)
        ax[plot_x, plot_y].plot(x_axis,count_ep1/len(ep1_loc), label = "ep1")
        ax[plot_x, plot_y].plot(x_axis,count_ep2/len(ep2_loc), label = "ep2")
        ax[plot_x, plot_y].set_title(str(character_index))
        ax[plot_x, plot_y].legend()
    plt.savefig(os.path.join(direct, "mentest_hist.jpg"))

def plot_prob_hist_char(prob, result,label,  direct, fn='mentest_proboverall.jpg'):
    fig, ax = plt.subplots(2, 2, tight_layout=True)
    x_axis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for character_index in sorted(prob.keys()):
        ep_label = label
        one_character_label = result[character_index,:]
        character_loc = set(np.where(one_character_label == 1)[0])
        character_prob = prob[character_index]
        plot_x = int(character_index*1.0/2)
        plot_y = int(character_index*1.0%2)
        for episode_number in [1,2]:
            ep_loc = np.where(ep_label == episode_number)[0]
            inter = np.array(list(set(ep_loc).intersection(character_loc)))
            ep_prob = character_prob[inter]
            count_ep, _ = np.histogram(ep_prob, bins= 10, range=(0.1, 1.01))
            ax[plot_x, plot_y].plot(x_axis,count_ep/len(inter), label = "ep"+str(episode_number))
        ax[plot_x, plot_y].set_title(str(character_index))
        ax[plot_x, plot_y].legend()
    plt.savefig(os.path.join(direct, "mentest_hist_condition.jpg"))

def plot_by_episode(prob_list, episode_number, ax):
    x_axis = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    count_ep, bins = np.histogram(prob_list, bins= x_axis)
    ax.plot(x_axis[:-1],count_ep/len(prob_list), label = "ep"+str(episode_number) + " : %0.3f" % give_overall_acc(prob_list))

def plot_interval_prob(interval_list,  direct, fn='mentest_proboverall.jpg'):
    fig, ax = plt.subplots(num_characters, num_characters,figsize=(20, 10))
    for character_index_conditon in range(num_characters):
        related_interval = []
        for i in interval_list:
            if i.check_character_prob(character_index_conditon):
                related_interval.append(i)
        plot_x = int(character_index_conditon)
        for character_index in range(num_characters):
            plot_y = int(character_index*1.0%num_characters)
            for episode_number in [1,2]:
                relative_character_prob = []
                for i in related_interval:
                    if i.episode == episode_number: 
                        relative_character_prob.append(i.get_character_prob(character_index))
                relative_character_prob = np.array(relative_character_prob).flatten()
                plot_by_episode(relative_character_prob, episode_number, ax[plot_x, plot_y])
            ax[plot_x, plot_y].set_title(str(character_index) + " condition on " + str(character_index_conditon))
            ax[plot_x, plot_y].legend()
    fig.suptitle("interval conditon on CNN prediction", fontsize="x-large")
    plt.savefig(os.path.join(direct, "mentest_cond_prediction.jpg"))

def give_overall_acc(relative_character_prob, threshold = 0.5):
    relative_character_prob = np.array(relative_character_prob)
    if len(relative_character_prob) == 0:
        return 0 
    return sum(relative_character_prob >= threshold)*1.0/len(relative_character_prob)

def plot_interval_label(interval_list,  direct, fn='mentest_proboverall.jpg'):
    fig, ax = plt.subplots(num_characters, num_characters, figsize=(20, 10))
    x_axis = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for character_index_conditon in range(num_characters):
        related_interval = []
        for i in interval_list:
            if i.check_character_label(character_index_conditon):
                related_interval.append(i)
        plot_x = int(character_index_conditon)
        for character_index in range(num_characters):
            plot_y = int(character_index*1.0%num_characters)
            for episode_number in [1,2]:
                relative_character_prob = []
                for i in related_interval:
                    if i.episode == episode_number: 
                        relative_character_prob.append(i.get_character_prob(character_index))
                relative_character_prob = np.array(relative_character_prob).flatten()
                plot_by_episode(relative_character_prob, episode_number, ax[plot_x, plot_y])
            ax[plot_x, plot_y].set_title(str(character_index) + " condition on " + str(character_index_conditon))
            ax[plot_x, plot_y].legend()
    fig.suptitle("interval conditon on CV label", fontsize="x-large")
    plt.savefig(os.path.join(direct, "mentest_cond_cv_label.jpg"))

def plot_prob_hist(character_label_board, episode_label_board, person_prob, direct, fn='mentest_proboverall.jpg'):
    # for each patient, plot the episode 1 prediction, episode 2 prediction and the response time predictions, GT distribution
    num_person = len(list(person_prob.keys()))
    label_list = ['E1 clip P','E1 Resp P', 'E2 clip P','E2 Resp P', 'GT clip','GT Resp' ]
    n_bins = 30
    n_fts = len(label_list)
    plt.figure(figsize=(20, 20))
    #print("num_person, n_fts", num_person, n_fts)

    for person_index in sorted(person_prob.keys()):
        one_person_prob = person_prob[person_index]
        one_person_label = character_label_board[person_index,:]
        for target_label in range(1,5):
            target_res = one_person_prob[episode_label_board==target_label]
            #print(person_index*n_fts + target_label)
            ax = plt.subplot(num_person, n_fts, person_index*n_fts + target_label)
            ax.hist(target_res, bins=np.linspace(0, 1, n_bins))
            ax.set_title(label_list[target_label - 1])
        for target_label in range(5,7):
            offset = target_label - 4 ## select label1,2;; select response 3,4
            select_clip = np.logical_or(episode_label_board==offset, episode_label_board ==(offset+1))
            target_res = one_person_label[select_clip]
            ax = plt.subplot(num_person, n_fts, person_index*n_fts + target_label)
            ax.hist(target_res)
            ax.set_title(label_list[target_label- 1])
        plt.title("person index: " + str(person_index) +  " men_test_prob plot")
    plt.savefig(os.path.join(direct, fn))
    plt.close('all')

def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]

def cross_coef(X, Y):
    num_observations = X.shape[1]
    EXY = X @ Y.T /(num_observations)
    #EXY = X @ Y.T
    EX = np.mean(X, axis=1).reshape((-1,1))
    EY = np.mean(Y, axis=1).reshape((-1,1))
    #m = EX @ EY.T
    #print(EX.shape, EY.shape, type(m))
    cov = EXY - EX @EY.T
    x_arr_var = np.var(X, axis=1).reshape((-1,1))
    y_arr_var = np.var(Y, axis=1).reshape((-1,1))
    diag_mat = np.sqrt(x_arr_var @ y_arr_var.T)
    #print(diag_mat)
    cov_coef = np.divide(cov, diag_mat)
    return cov, cov_coef
    
def pearson_corr_coef(X):
    return np.corrcoef(X)

def tlcc(x,y):
    ## Function returns the static time lag between x and y
    ## for mem test, set x as longer test clip + response time, set y as in movie signals
    r_xy = np.correlate(x, y, mode='full')
    len_valid_y = len(y)
    time_lag = np.argmax(r_xy) - len_valid_y + 1 
    #time_lag = np.ceil(len(r_xy)/2)-np.argmax(r_xy)
    return time_lag, np.max(r_xy), r_xy

