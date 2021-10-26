import sys
sys.path.append(".")
from neural_correlation.train_model import *
from neural_correlation.cnn import *
from neural_correlation.cnn import BasicCNN
from neural_correlation.cnn_dataloader import create_split_loaders
from neural_correlation.utilities import *
from lstm import MovieLSTM
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from project_setting import episode_number
if episode_number == 1:
    from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
        character_dict, path_to_matlab_generated_movie_data, num_characters, feature_time_width, model_option
elif episode_number == 2:
    from data2 import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
        character_dict, path_to_matlab_generated_movie_data, num_characters,feature_time_width
import copy 
from sklearn import metrics
from collections import OrderedDict 
torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
seed = 0
np.random.seed(seed)
random.seed(seed)

loss_modes = ["MSE", "KLD"]
loss_mode = "KLD"

def get_character_loss(res_label_temp, res_pre_temp,batch_index, person_index): 
    if loss_mode == "MSE":
        res = np.array((res_label_temp[batch_index][person_index,0:2] - res_pre_temp[batch_index][person_index,0:2])**2)
    if loss_mode == "KLD":
        y = res_label_temp[batch_index][person_index,0:2]
        yHat = res_pre_temp[batch_index][person_index,0:2]
        if y[1] == 0:
            if y[0] == 0 :
                return 0
            return y[0] * (np.log(y[0]) - yHat[0])
        if y[0] == 0:
            return y[1] * (np.log(y[1]) - yHat[1])
        
        res = np.sum(y * (np.log(y) - yHat))
    return res

def get_all_characters_confusion(person_confusion):
    all_characters_confusion = {}
    for person_index in sorted(person_confusion.keys()):
        one_person_frame_confusion = person_confusion[person_index]
        one_person_confusion = np.zeros((3,3))
        for frame_sorted in sorted(one_person_frame_confusion.keys()):
            one_person_confusion += one_person_frame_confusion[frame_sorted]
        all_characters_confusion[person_index] = one_person_confusion.astype(int)
    print(all_characters_confusion)

def get_all_characters_loss(loss_record):
    all_characters_loss = {}
    for person_index in sorted(loss_record.keys()):
        one_person_loss_record = loss_record[person_index]
        one_person_loss = np.array([0.0,0.0])
        for frame_sorted in sorted(one_person_loss_record.keys()):
            one_person_loss += one_person_loss_record[frame_sorted]
        all_characters_loss[person_index] = one_person_loss*1.0/len(one_person_loss_record.keys())
    return all_characters_loss


def inference(model, data_fn, label_fn, tag_fn ,frame_name_dir, this_patient , mask, direct, computing_device, extras, kfold = -1,erasing = True):
         # Setup: initialize the hyperparameters/variables
    batch_size = 64          # Number of samples in each minibatch
    seed = 0  # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
    # Setup the training, validation, and testing dataloaders
    train_loader, val_loader, test_loader = create_split_loaders(data_fn, label_fn,tag_fn ,frame_name_dir, \
        batch_size, seed, kfold=kfold ,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)


    outputs, labels, frame_name_all, N_minibatch_acc = test(model, train_loader, computing_device)
    
    ## get stats for every frame
    person_dict = {}
    person_confusion = {}
    label_record = {}
    loss_record = {}
    for person_index in range(num_characters):
        frame_stats = {}
        confusion_stats= {}
        one_person_label = {}
        loss_person = {}
        for i in range(len(outputs)):
            batch_label = labels[i]
            batch_output = outputs[i]
            frame_batch = frame_name_all[i]
            res_label_temp = np.squeeze(batch_label, axis=1)
            res_label = np.argmax(res_label_temp, axis=2)
            res_pre_temp = batch_output
            res_pre = np.argmax(res_pre_temp, axis=2)
            for batch_index in range(len(res_label)):
                frame_num = frame_batch[batch_index]
                frame_stats[frame_num] = get_stats(res_label, res_pre,batch_index, person_index)
                confusion_stats[frame_num] = get_confusion(res_label, res_pre,batch_index, person_index)
                loss_person[frame_num] = get_character_loss(res_label_temp, res_pre_temp, batch_index, person_index) 
                if res_label[batch_index][person_index]== 0 :
                    one_person_label[frame_num] = 1
        person_dict[person_index] = frame_stats
        person_confusion[person_index] = confusion_stats
        label_record[person_index] = one_person_label
        loss_record[person_index] = loss_person
    
    all_characters_loss = get_all_characters_loss(loss_record)
    person_prediction_stats = {}
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_stats = np.zeros(8)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            one_person_stats += one_person_frame_stats[frame_sorted]
        yes_c, yes_all = one_person_stats[0], one_person_stats[0]+ one_person_stats[1]
        no_c, no_all = one_person_stats[2], one_person_stats[2]+ one_person_stats[3]
        dnk_c, dnk_all = one_person_stats[4], one_person_stats[4]+ one_person_stats[5]
        person_prediction_stats[person_index] = 1.0*(yes_c + no_c)/(yes_all + no_all)

    return all_characters_loss, person_prediction_stats, loss_record


def test_cnn_model_results(this_patient, path_to_saved_folder,kfolds = -1 ,masked = False, erasing = True, path_to_model = path_to_model,path_to_training_data = path_to_training_data, mode = 'train'):
    print(kfolds)
    # seed
    seed = 10
    np.random.seed(seed)
    random.seed(seed)
    # Model_Setup:
    torch.cuda.empty_cache()
    num_characters = 4

    if len(path_to_model.split("/")[-1]) == 1:
        direct = path_to_model
    else:
        direct = os.path.join(path_to_model, this_patient)
    
    if len(path_to_saved_folder.split("/")[-1])==1:
        save_dir = path_to_saved_folder
    else:
        save_dir = os.path.join(path_to_saved_folder, this_patient)
        os.mkdir(save_dir)
    # generate data
    data_fn = os.path.join(path_to_training_data, this_patient, "feature.npy")
    label_fn = os.path.join(path_to_training_data, this_patient, "label.npy")
    frame_name_dir = os.path.join(path_to_training_data, this_patient, "frame_number.npy")
    tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_regions_clean.npy")

    raw_data_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_clean.npy")
    num_neurons = np.load(raw_data_fn).shape[0]

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 10, "pin_memory": True}
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False

    model_dirct = join(direct,'best.pth')
    print('Resume model: %s' % model_dirct)
    if model_option is "CNN":
        model = BasicCNN(num_characters, patient_features[this_patient])
    else:
        model = MovieLSTM(data_fn)
    model_test = model.cuda()
    check_point = torch.load(model_dirct)
    model_test.load_state_dict(check_point['state_dict'])
    
    ## mircowire
    microarr_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"channel_data.mat")
    microarr_temp = loadmat(microarr_fn)["channel_reg_info"][0]
    microarr = []
    for m in microarr_temp:
        microarr.append(m[0][0][0])
    print(microarr)
    microwire_loss_all, mircowire_all_characters_loss, microwire_all_characters_acc, _ = erasing_pipeline(model, microarr,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds)
    
    ### region 
    region_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn, allow_pickle=True)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_single = removeDuplicates(reg)
    reg = np.array(reg)
    reg_single = np.array(reg_single)
    region_loss_all, region_all_characters_loss, region_all_characters_acc, region_frame_loss = erasing_pipeline(model, reg,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds)

    #### 
    ### extract submatrix as microarr based loss for microwire, region
    ### neuron_id -> tag
    #print(microarr) 
    jump_diff = np.diff(microarr)
    #print(jump_diff)
    micro_arr_jump_index = np.where(jump_diff)[0]
    #print(micro_arr_jump_index) 
    #if jump_diff [-1] >= 1:
    #micro_arr_jump_index = np.append(micro_arr_jump_index, micro_arr_jump_index[-1]+1)
    
    #microwire_loss_all = region_loss_all
    #mircowire_all_characters_loss = region_all_characters_loss
    #microwire_all_characters_acc = region_all_characters_acc

    micro_arr_jump_index = np.append(micro_arr_jump_index, len(microarr)-1)
    #print("Micro arr compare index added", micro_arr_jump_pre, micro_arr_jump_index)
    #print(micro_arr_jump_index) 
    microwire_loss_all_trim = microwire_loss_all[:,micro_arr_jump_index]
    region_loss_all_trim = region_loss_all[:, micro_arr_jump_index]
    reg_trim = reg[micro_arr_jump_index]

    plot_graph(microwire_loss_all_trim,region_loss_all_trim, reg_single,reg_trim, save_dir)
    plot_loss_diff(region_frame_loss, save_dir)
    plot_acc_loss([conver2numpymatrix(region_all_characters_loss), conver2numpymatrix(region_all_characters_acc)],\
         sorted(list(character_dict.keys())), list(region_all_characters_loss.keys()), "loss_acc_region", save_dir)
    return region_all_characters_loss, region_all_characters_acc, mircowire_all_characters_loss, microwire_all_characters_acc, microwire_loss_all_trim, region_loss_all_trim, reg_trim

def covert2numpy(keys, dic):
    res = np.zeros(max(keys) + 1)
    for k in sorted(keys):
        res[k] = dic[k]
    return res

def conver2numpymatrix(dic):
    keys = dic.keys()
    res = np.zeros((len(keys), len(dic[list(keys)[0]])))
    klist = list(keys)
    for kidx in range(len(klist)):
        if type(dic[klist[kidx]]) is dict:
            res[kidx,:] = covert2numpy(dic[klist[kidx]].keys(),dic[klist[kidx]])
        else:
            res[kidx,:] = dic[klist[kidx]]
    return res

def plot_loss_diff(region_frame_loss, save_dir):
    baseline_loss = region_frame_loss["baseline"]
    for region_tag in sorted(region_frame_loss.keys()):
        plt.figure(figsize=(40, 10))
        loss = region_frame_loss[region_tag]
        for person_index in sorted(loss.keys()):
            reg_loss = covert2numpy(loss[person_index].keys(), loss[person_index])
            baseloss = covert2numpy(baseline_loss[person_index].keys(), baseline_loss[person_index])
            plt.subplot(len(loss),1,person_index + 1)
            #one_person_tpfp = one_person_tpfp[:2000]
            start = 0
            plt.plot(np.exp(-reg_loss) - np.exp(-baseloss), label="baseloss")
            #plt.plot(reg_loss, label="reg_loss")
            #plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_person_label==1, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
            plt.title("person index: " + str(person_index) +  " loss plot")
            plt.legend()
            plt.ylim(-1.1, 1.1)
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('Time (s)')
        fname =  "loss_" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')
        fig, ax =  plt.subplots(len(loss),1)
        for person_index in sorted(loss.keys()):
            reg_loss = covert2numpy(loss[person_index].keys(), loss[person_index])
            baseloss = covert2numpy(baseline_loss[person_index].keys(), baseline_loss[person_index])
            #one_person_tpfp = one_person_tpfp[:2000]
            start = 0
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            temp_ind = np.array(range(len(temp)))
            ax[int(person_index)].hist(temp_ind[temp<-0.5])
            #ax[int(person_index)].hist(np.where(temp<0)[0])
            #plt.plot(reg_loss, label="reg_loss")
            #plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_person_label==1, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
            ax[int(person_index)].set_title("person index: " + str(person_index) +  " hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('frame')
        fname =  "loss_hist_frame" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')

        fig, ax =  plt.subplots(len(loss),1, figsize=(40, 10))
        for person_index in sorted(loss.keys()):
            reg_loss = covert2numpy(loss[person_index].keys(), loss[person_index])
            baseloss = covert2numpy(baseline_loss[person_index].keys(), baseline_loss[person_index])
            #one_person_tpfp = one_person_tpfp[:2000]
            start = 0
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            width = 0.05
            hist, bin_edges = np.histogram(temp, bins=20, range=(-1, 1))
    
            #print(hist.shape)
            #print(bin_edges)
            #print(bin_edges.shape)
            #temp_ind = np.array(range(len(temp)))
            #ax[int(person_index)].hist(temp_ind[temp<0])
            #ax[int(person_index)].hist(temp, range=(-1,-0.001)))
            bin_center = np.diff(bin_edges)/2 + bin_edges[:-1]
            rects = ax[int(person_index)].bar(bin_center,hist, width = width)
            autolabel(rects, ax[int(person_index)])
            ax[int(person_index)].set_ylim(0, 10000)
            #plt.plot(reg_loss, label="reg_loss")
            #plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_person_label==1, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
            ax[int(person_index)].set_title("person index: " + str(person_index) +  " hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('Time (s)')
        fname =  "loss_hist_" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')

        fig, ax =  plt.subplots(len(loss),1, figsize=(40, 10))
        for person_index in sorted(loss.keys()):
            reg_loss = covert2numpy(loss[person_index].keys(), loss[person_index])
            baseloss = covert2numpy(baseline_loss[person_index].keys(), baseline_loss[person_index])
            #one_person_tpfp = one_person_tpfp[:2000]
            start = 0
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            base_val = np.exp(-reg_loss)
            width = 0.05
            selected_reg = np.logical_and(temp > -0.1, temp < 0.1 )
            hist, bin_edges = np.histogram(base_val[selected_reg], bins=20, range=(0, 1))
            #print(hist.shape)
            #print(bin_edges)
            #print(bin_edges.shape)
            #temp_ind = np.array(range(len(temp)))
            #ax[int(person_index)].hist(temp_ind[temp<0])
            #ax[int(person_index)].hist(temp, range=(-1,-0.001)))
            bin_center = np.diff(bin_edges)/2 + bin_edges[:-1]
            rects = ax[int(person_index)].bar(bin_center,hist, width = width)
            autolabel(rects, ax[int(person_index)])
            ax[int(person_index)].set_ylim(0, 10000)
            #plt.plot(reg_loss, label="reg_loss")
            #plt.scatter(np.divide(range(start,start+len(one_person_tpfp)), sr_final_movie_data), 0.1 + np.asarray(one_person_label==1, dtype = float) , s=5, c='tab:red',label="label",alpha=1)
            ax[int(person_index)].set_title("person index: " + str(person_index) +  " hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('Time (s)')
        fname =  "loss_zerohist_" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def erasing_pipeline(model, tag,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds):

    loss_frame = {}

    loss_base, acc_base, loss_person = inference(model.eval(), data_fn, label_fn, tag_fn, frame_name_dir, this_patient, None, direct, computing_device,extras, kfold=kfolds)

    tag_single = removeDuplicates(tag)
    tag = np.array(tag) 
    loss_all = np.zeros((num_characters,num_neurons))
    all_characters_loss = OrderedDict()
    all_characters_acc = OrderedDict()

    ## 
    r = "baseline"
    all_character = np.zeros(num_characters)
    for i in range(num_characters):
        #all_character[i] = loss_base[i][0] + loss_base[i][1] ### ?? Why?
        all_character[i] = loss_base[i][0]
    all_characters_loss[r] = all_character
    all_characters_acc[r] = acc_base
    loss_frame[r] = loss_person

    for r in sorted(tag_single):
        print(r)
        loc = np.where(tag==r)[0]
        all_character = np.zeros(num_characters)
        loss, acc, loss_person = inference(model.eval(),data_fn, label_fn, tag_fn ,frame_name_dir, this_patient, loc , direct, computing_device,extras, kfold=kfolds)
        for i in range(num_characters):
            #loss_all[i, loc] = loss[i][0] + loss[i][1] - all_characters_loss["baseline"][i]
            loss_all[i, loc] = loss[i][0] - all_characters_loss["baseline"][i]
            #all_character[i] = loss[i][0] + loss[i][1]
            all_character[i] = loss[i][0]
        all_characters_loss[r] = all_character
        all_characters_acc[r] = acc
        loss_frame[r] = loss_person
    #print(loss_all)
    return loss_all,all_characters_loss, all_characters_acc, loss_frame

def plot_graph(mircowire_loss, region_loss, reg_single, reg, save_dir):
    label = np.zeros(len(reg_single))
    sum_mircowire = region_loss * 0

    for k in range(len(reg_single)):
        loc = np.where(reg==reg_single[k])[0]
        print(loc, reg, reg_single[k])
        label[k] = int((loc.max() + loc.min())/2)
        if len(loc) == 1:
            sum_mircowire[:,loc] = mircowire_loss[:,loc]    
            continue
        sum_mircowire[:,loc] = np.tile(np.expand_dims(np.sum(mircowire_loss[:,loc], axis = 1), axis = 1), (1, len(loc)))
    plt.close("all")
    fig, axes = plt.subplots(mircowire_loss.shape[0],1, sharey=True ,figsize=(12,12)) 
    for character_number in range(mircowire_loss.shape[0]):
        axes[character_number].plot(mircowire_loss[character_number,:], 'bo-', label="mircowire loss")
        axes[character_number].plot(region_loss[character_number,:], 'r*-',label="region loss")
        axes[character_number].plot(sum_mircowire[character_number,:], 'gd-',label="sum mircowire loss")
        axes[character_number].set_xticks(label, minor=False)
        axes[character_number].set_xticklabels(reg_single, minor=False)
        axes[character_number].set_title("person index: " + str(character_number) +  " importance plot")
        axes[character_number].legend()
    plt.savefig(os.path.join(save_dir, "important_region.jpg"))
    save_stats = {"mircowire_loss":mircowire_loss,"region_loss":region_loss,"sum_mircowire_loss":sum_mircowire,"reg":reg}
    dump_pickle(os.path.join(save_dir, "knock_out_stats.pkl"), save_stats)

def plot_heatmap(data, hori_label, verti_label, title, loc):
    # Create a dataset (fake)
    plt.close("all")
    plt.figure()
    df = pd.DataFrame(data, index= verti_label, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    p1 = sn.heatmap(df, cmap=plt.cm.Blues, annot=True, fmt='0.2f')
    plt.title(title)
    fn = os.path.join(loc, title+".jpg")
    plt.savefig(fn)

def plot_acc_loss(data, hori_label, verti_label, title, loc):
    #print(data[0])
    #print(data[1])
    #print(hori_label)
    #print(verti_label)
    # Create a dataset (fake)
    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    df0 = pd.DataFrame(data[0], index= verti_label, columns=hori_label)
    df1 = pd.DataFrame(data[1], index= verti_label, columns=hori_label)
    # Default heatmap: just a visualization of this square matrix
    ax = sn.heatmap(df0, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[0])
    ax.set(title = "loss") 
    ax = sn.heatmap(df1, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[1])
    ax.set(title = "accuracy")
    fn = os.path.join(loc, title+".jpg")
    plt.savefig(fn)
    
    save_stats = {"confusion_matrix":data,"hori_label":hori_label,"verti_label":verti_label}
    dump_pickle(os.path.join(loc, "knockout_confusion_"+title+".pkl"), save_stats)

def stats_region_kfold(kfolds_stats):
    [loss_region, acc_region, loss_mircowire, acc_mircowire,  microwire_loss_trim, region_loss_trim, reg_trim] = kfolds_stats[0]
    regions = list(loss_region.keys())
    loss_region_overall = np.zeros((len(regions), num_characters))
    acc_region_overall = np.zeros((len(regions), num_characters))

    mircowires = list(loss_mircowire.keys())
    loss_mircowires_overall = np.zeros((len(mircowires), num_characters))
    acc_mircowires_overall = np.zeros((len(mircowires), num_characters))
    
    microwire_loss_trim_overall = microwire_loss_trim * 0
    region_loss_trim_overall = region_loss_trim * 0

    for i in kfolds_stats:
        [loss_region, acc_region, loss_mircowire, acc_mircowire, microwire_loss_trim, region_loss_trim, _] = i
        for idx in range(len(regions)):
            for character_idx in range(num_characters):
                loss_region_overall[idx, character_idx] += loss_region[regions[idx]][character_idx] 
                acc_region_overall[idx, character_idx] += acc_region[regions[idx]][character_idx]
        for idx in range(len(mircowires)):
            for character_idx in range(num_characters):
                loss_mircowires_overall[idx, character_idx] += loss_mircowire[mircowires[idx]][character_idx] 
                acc_mircowires_overall[idx, character_idx] += acc_mircowire[mircowires[idx]][character_idx]
        microwire_loss_trim_overall += microwire_loss_trim
        region_loss_trim_overall += region_loss_trim
    loss_region_overall = loss_region_overall/len(kfolds_stats)
    acc_region_overall = acc_region_overall/len(kfolds_stats)
    
    loss_mircowires_overall = loss_mircowires_overall/len(kfolds_stats)
    acc_mircowires_overall = acc_mircowires_overall/len(kfolds_stats)

    region_loss_trim_overall = region_loss_trim_overall/len(kfolds_stats)
    microwire_loss_trim_overall = microwire_loss_trim_overall/len(kfolds_stats)

    return loss_region_overall, acc_region_overall, regions, loss_mircowires_overall, acc_mircowires_overall, mircowires, region_loss_trim_overall, microwire_loss_trim_overall, reg_trim

def single_run(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    path_to_saved_folder = os.path.join(project_dir, "important_neurons_loss_clean")
    clean_folder(path_to_saved_folder)
    for this_patient in patientNums:
        test_cnn_model_results(this_patient, path_to_saved_folder ,masked= True, erasing = True, path_to_model = path_to_model)

def multi_run(patientNums = patientNums):
    path_to_saved_folder = os.path.join(project_dir, "knockout_test_" + model_option + "_" + loss_mode)
    cnn_result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
    #cnn_result_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD"
    clean_folder(path_to_saved_folder)
    for this_patient in sorted(patientNums):
        print(patientNums)
        path_to_patient = os.path.join(cnn_result_dir, this_patient)
        #path_to_patient = os.path.join(project_dir, 'CNN_result','CNN_multi_'+str(feature_time_width), this_patient)
        path_to_saved_folder_patient = os.path.join(path_to_saved_folder, this_patient)
        clean_folder(path_to_saved_folder_patient)
        kfolds = os.listdir(path_to_patient)
        kfolds_stats = []
        kfolds_stats = []
        for k in kfolds:
            folder_name = os.path.join(path_to_patient, k)
            plot_saved_folder = os.path.join(path_to_saved_folder_patient, k)
            clean_folder(plot_saved_folder)
            if not os.path.isdir(folder_name):
                continue
            loss_region, acc_region, loss_neuron, acc_neuron,  microwire_loss_trim, region_loss_trim, reg_trim = test_cnn_model_results(this_patient, plot_saved_folder, kfolds = int(k) ,masked= True, erasing = True, path_to_model = folder_name)
            kfolds_stats.append([loss_region, acc_region, loss_neuron, acc_neuron,  microwire_loss_trim, region_loss_trim, reg_trim])
        np.savez(os.path.join(path_to_patient, 'importance_analysis'), kfolds_stats = kfolds_stats)
        loss_region_overall, acc_region_overall, regions, loss_mircowires_overall, acc_mircowires_overall, mircowires, region_loss_trim_overall, microwire_loss_trim_overall, reg_trim = stats_region_kfold(kfolds_stats)
        plot_acc_loss([loss_region_overall,acc_region_overall], sorted(list(character_dict.keys())), regions, "loss_acc_region", path_to_patient)
        plot_acc_loss([loss_mircowires_overall,acc_mircowires_overall], sorted(list(character_dict.keys())), mircowires, "loss_acc_mircowire", path_to_patient)
        plot_graph(microwire_loss_trim_overall, region_loss_trim_overall, removeDuplicates(reg_trim), reg_trim, path_to_patient)

def draft():
    path_to_patient = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/CNN_multi_4/431/"
    aa = np.load("/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/CNN_multi_4/431/importance_analysis.npz")
    loss_region_overall, acc_region_overall, regions = stats_region_kfold(aa["kfolds_stats"])
    plot_acc_loss([loss_region_overall,acc_region_overall], sorted(list(character_dict.keys())), regions, "loss", path_to_patient)
def main(patientNums = patientNums):
    multi_run()
    #draft()
    
if __name__ == '__main__':
    main()