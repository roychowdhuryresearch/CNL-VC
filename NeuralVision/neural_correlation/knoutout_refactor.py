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

from KnockoutPerformance import KnockoutPerformance, KnockoutKfoldPerformance
from ModelPerformance import ModelPerformance

torch.manual_seed(0)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
seed = 0
np.random.seed(seed)
random.seed(seed)

loss_modes = ["MSE", "KLD"]
loss_mode = "KLD"


def inference(model, data_fn, label_fn, tag_fn ,frame_name_dir, this_patient , mask, direct, computing_device, extras, kfold = -1,erasing = True):
         # Setup: initialize the hyperparameters/variables
    batch_size = 64          # Number of samples in each minibatch
    seed = 0  # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
    # Setup the training, validation, and testing dataloaders
    train_loader, _, _ = create_split_loaders(data_fn, label_fn,tag_fn ,frame_name_dir, \
        batch_size, seed, kfold=kfold ,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)

    outputs, labels, frame_name_all, _ = test(model, train_loader, computing_device)
    
    return outputs, labels, frame_name_all

###weird plots 
def plot_loss_diff(region_frame_loss, save_dir):
    baseline_loss = region_frame_loss["baseline"]
    for region_tag in sorted(region_frame_loss.keys()):
        plt.figure(figsize=(40, 10))
        loss = region_frame_loss[region_tag]
        for person_index in range(len(loss)):
            reg_loss = loss[person_index,:]
            baseloss = baseline_loss[person_index,:]
            plt.subplot(len(loss),1, person_index + 1)
            start = 0
            plt.plot(np.exp(-reg_loss) - np.exp(-baseloss), label="baseloss")
            plt.title("character index: " + str(person_index) +  " loss plot")
            plt.legend()
            plt.ylim(-1.1, 1.1)
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('Time (s)')
        fname =  "loss_" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')
        
        fig, ax =  plt.subplots(len(loss),1, figsize=(20, 15))
        for person_index in range(len(loss)):
            reg_loss = loss[person_index,:]
            baseloss = baseline_loss[person_index,:]
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            temp_ind = np.array(range(len(temp)))
            ax[int(person_index)].hist(temp_ind[temp<-0.5])
            ax[int(person_index)].set_title("character index: " + str(person_index) +  " prob diff < " + str(-0.5) + " hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('frame')
        fname =  "loss_hist_frame" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')

        fig, ax =  plt.subplots(len(loss),1, figsize=(40, 15))
        for person_index in range(len(loss)):
            reg_loss = loss[person_index]
            baseloss = baseline_loss[person_index]
            start = 0
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            #width = 0.05
            #my_zoom_zero_bins = np.array(list(range(-1, -0.9, 0.01)) + list(range(-0.9,-0.1, 0.1))\
            #+ list(range(-0.1,0.1, 0.02)) + list(range(0.1,1, 0.1)))
            my_zoom_zero_bins = np.array(list(np.linspace(-1.0, -0.9, num=5)) + list(np.linspace(-0.9,-0.1, num=5))\
            + list(np.linspace(-0.1,0.1, num=5)) + list(np.linspace(0.1,1,num=5)))

            width = 0.05
            hist, bin_edges = np.histogram(temp, bins=my_zoom_zero_bins, range=(-1, 1))
            bin_center = np.diff(bin_edges)/2 + bin_edges[:-1]
            rects = ax[int(person_index)].bar(bin_center,hist, width = width)
            autolabel(rects, ax[int(person_index)])
            ax[int(person_index)].set_ylim(0, 10000)
            ax[int(person_index)].set_title("character index: " + str(person_index) +  " hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('diff of prob')
        fname =  "loss_hist_" + str(region_tag) + ".jpg"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close('all')

        fig, ax =  plt.subplots(len(loss),1, figsize=(40, 10))
        for person_index in range(len(loss)):
            reg_loss = loss[person_index]
            baseloss = baseline_loss[person_index]
            start = 0
            temp = np.exp(-reg_loss) - np.exp(-baseloss)
            base_val = np.exp(-reg_loss)
            width = 0.05
            selected_reg = np.logical_and(temp > -0.1, temp < 0.1 )
            hist, bin_edges = np.histogram(base_val[selected_reg], bins=20, range=(0, 1))
            bin_center = np.diff(bin_edges)/2 + bin_edges[:-1]
            rects = ax[int(person_index)].bar(bin_center,hist, width = width)
            autolabel(rects, ax[int(person_index)])
            ax[int(person_index)].set_ylim(0, 10000)
            ax[int(person_index)].set_title("character index: " + str(person_index) +  " perturb hist plot")
            if person_index is not 3:
                plt.xticks([], [])
        plt.xlabel('diff_prob')
        fname =  "loss_zero_hist_" + str(region_tag) + ".jpg"
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

def erasing_pipeline(model,tag,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds, erasing_tag= None):

    ## 
    r = "baseline"
    if erasing_tag is not None:
        loc = get_mask_index(tag, None, erasing_tag)
        tag_removed = np.delete(tag, loc)
        knockout = KnockoutPerformance(this_patient,tag_removed)
        tag_single = removeDuplicates(tag_removed)
    else:
        loc = None
        knockout = KnockoutPerformance(this_patient,tag)
        tag_single = removeDuplicates(tag)
    outputs, labels, frame_name_all = inference(model.eval(), data_fn, label_fn, tag_fn, frame_name_dir, this_patient, loc, direct, computing_device,extras, kfold=kfolds)
    model_performance = ModelPerformance(kfolds,this_patient,model_option=model_option)
    model_performance.construct(outputs,labels,frame_name_all)
    knockout.add(r, model_performance)

    tag = np.array(tag) 

    for r in sorted(tag_single):
        loc = get_mask_index(tag, r, erasing_tag)
        outputs, labels, frame_name_all = inference(model.eval(),data_fn, label_fn, tag_fn ,frame_name_dir, this_patient, loc , direct, computing_device,extras, kfold=kfolds)
        knockout.add(r, ModelPerformance.generate(kfolds,this_patient,model_option,outputs,labels,frame_name_all))
    #print(loss_all)
    return knockout

def get_mask_index(tag, r, erasing_tags):
    if erasing_tags is None:
        loc = np.where(tag==r)[0]
    else:
        loc = []
        if r is not None:
            loc.append(np.where(tag == r)[0])
        for rr in erasing_tags:
            loc.append(np.where(tag == rr)[0])
        loc = np.concatenate(loc)
    return loc


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

def plot_acc_loss(data, hori_label, verti_label, fn, loc, title=None):
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
    ax.set(title = "loss") 
    ax = sn.heatmap(df1, cmap=plt.cm.Blues, annot=True, fmt='0.2f', ax=axs[1])
    ax.set(title = "accuracy")
    if title is not None:
        plt.suptitle(title)
    fn_save = os.path.join(loc, fn+".jpg")
    plt.savefig(fn_save)
    save_stats = {"confusion_matrix":data,"hori_label":hori_label,"verti_label":verti_label}
    dump_pickle(os.path.join(loc, "knockout_confusion_"+fn+".pkl"), save_stats)

def model_inference_results(this_patient, path_to_saved_folder,kfolds = -1 ,masked = False, erasing = True, path_to_model = path_to_model,path_to_training_data = path_to_training_data, mode = 'train'):
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
        extras = {"num_workers": 1, "pin_memory": True}
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
    print("______________________microarr_______________________")
    knoutout_mircowire = erasing_pipeline(model, microarr,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds)
    dump_pickle(os.path.join(save_dir, "mircowire_stats.pkl"), knoutout_mircowire.get_knockout_stats())

    ### region 
    region_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn, allow_pickle=True)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_single = removeDuplicates(reg)
    reg = np.array(reg)
    reg_single = np.array(reg_single)
    print("______________________region_______________________")
    knockout_region = erasing_pipeline(model, reg,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds)

    acc_region, regions = knockout_region.get_acc_stats()
    loss_region, _ = knockout_region.get_loss_stats()
    print(acc_region)
    print(loss_region)
    plot_acc_loss([loss_region, acc_region],\
         sorted(list(character_dict.keys())), list(regions), "loss_acc_region", save_dir)
    dump_pickle(os.path.join(save_dir, "reg_stats.pkl"), knockout_region.get_knockout_stats())
    #### 
    ### extract submatrix as microarr based loss for microwire, region
    ### neuron_id -> tag
    #print(microarr) 

    #print(micro_arr_jump_index) 
    #if jump_diff [-1] >= 1:
    #micro_arr_jump_index = np.append(micro_arr_jump_index, micro_arr_jump_index[-1]+1)
    print(reg)
    trim_idx = get_trim_index(microarr)
    reg_trim = reg[trim_idx]
    print(reg_trim)
    microwire_loss_trim = knoutout_mircowire.get_loss_tag_expand()[trim_idx].T
    region_loss_trim = knockout_region.get_loss_tag_expand()[trim_idx].T

    plot_graph(microwire_loss_trim,region_loss_trim, reg_single,reg_trim, save_dir)
    
    region_frame_loss = knockout_region.get_frame_loss()
    plot_loss_diff(region_frame_loss, save_dir)

    return knockout_region, knoutout_mircowire, reg_trim


def model_inference_results_retrain(this_patient, path_to_saved_folder, current_region=None,kfolds = -1 ,masked = False, erasing = True, path_to_model = path_to_model,path_to_training_data = path_to_training_data, mode = 'train'):
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
    
    ### region 
    region_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_single = removeDuplicates(reg)
    reg = np.array(reg)
    reg_single = np.array(reg_single)
    
    knockout_region = erasing_pipeline(model, reg,num_neurons, data_fn, label_fn, tag_fn, frame_name_dir, this_patient, direct, computing_device,extras, kfolds,erasing_tag=current_region)

    acc_region, regions = knockout_region.get_acc_stats()
    loss_region, _ = knockout_region.get_loss_stats()
    print(acc_region)
    print(loss_region)
    plot_acc_loss([loss_region, acc_region],\
         sorted(list(character_dict.keys())), list(regions), "loss_acc_region", save_dir, title = "loss_acc_region with knockout " + "_".join(current_region))

    return knockout_region

def multi_run(patientNums = patientNums):
    #path_to_model = os.path.join(project_dir, 'CNN_result','CNN_70_middletest')]
    path_to_saved_folder = os.path.join(project_dir, "knockout_test_" + model_option + "_" + loss_mode)
    #cnn_result_dir = os.path.join(project_dir, 'CNN_result','CNN_multi_'+str(feature_time_width)+"_"+loss_mode)
    cnn_result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
    #cnn_result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/CNN_multi_2_KLD"
    #cnn_result_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD"
    #cnn_result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
    clean_folder(path_to_saved_folder)
    for this_patient in sorted(patientNums):
        print(patientNums)
        path_to_patient = os.path.join(cnn_result_dir, this_patient)
        #path_to_patient = os.path.join(project_dir, 'CNN_result','CNN_multi_'+str(feature_time_width), this_patient)
        path_to_saved_folder_patient = os.path.join(path_to_saved_folder, this_patient)
        clean_folder(path_to_saved_folder_patient)
        kfolds = os.listdir(path_to_patient)
        knockoutKfoldRegion = KnockoutKfoldPerformance(len(kfolds), this_patient)
        knockoutKfoldMircowire = KnockoutKfoldPerformance(len(kfolds), this_patient)
        for k in kfolds:
            folder_name = os.path.join(path_to_patient, k)
            plot_saved_folder = os.path.join(path_to_saved_folder_patient, k)
            clean_folder(plot_saved_folder)
            if not os.path.isdir(folder_name):
                continue
            knockout_region, knockout_micorwire, _ = model_inference_results(this_patient, plot_saved_folder, kfolds = int(k) ,masked= True, erasing = True, path_to_model = folder_name)
            knockoutKfoldRegion.add(knockout_region)
            knockoutKfoldMircowire.add(knockout_micorwire)
        ## zaishuo 
        #np.savez(os.path.join(path_to_patient, 'importance_analysis'), kfolds_stats = kfolds_stats)
        acc_region_kfold, regions = knockoutKfoldRegion.get_acc_stats()
        loss_region_kfold, _ = knockoutKfoldRegion.get_loss_stats()
        acc_mircowire_kfold, mircowires = knockoutKfoldMircowire.get_acc_stats()
        loss_mircowire_kfold, _ = knockoutKfoldMircowire.get_loss_stats()
        plot_acc_loss([loss_region_kfold, acc_region_kfold], sorted(list(character_dict.keys())), regions, "loss_acc_region_1", path_to_patient)
        plot_acc_loss([loss_mircowire_kfold,acc_mircowire_kfold], sorted(list(character_dict.keys())), mircowires, "loss_acc_mircowire_1", path_to_patient)
        
        trim_idx = get_trim_index(knockoutKfoldMircowire.tags)
        reg_trim = knockoutKfoldRegion.tags[trim_idx]
        region_loss_kfold = knockoutKfoldRegion.get_loss_tag_expand()[trim_idx].T
        microwire_loss_kfold = knockoutKfoldMircowire.get_loss_tag_expand()[trim_idx].T
        print("Before REMOVE ::::::::: ", reg_trim)
        print("After REMOVE ::::::::: ", removeDuplicates(reg_trim))
        plot_graph(microwire_loss_kfold, region_loss_kfold, removeDuplicates(reg_trim), reg_trim, path_to_patient)

def get_trim_index(tag1):
    jump_diff = np.diff(tag1)
    jump_index = np.where(jump_diff)[0]
    jump_index = np.append(jump_index, len(tag1)-1) 
    return jump_index

def parse_erased_tag(tag):
    return tag.split("_")

def multi_rerun(patientNums = patientNums ):
    path_to_saved_folder = os.path.join(project_dir, "knockout_test_final_" + model_option + "_" + loss_mode)
    #result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
    #result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD"
    clean_folder(path_to_saved_folder)
    for this_patient in sorted(patientNums):
        print(patientNums)
        path_to_patient = os.path.join(result_dir, this_patient)
        path_to_saved_folder_patient = os.path.join(path_to_saved_folder, this_patient)
        clean_folder(path_to_saved_folder_patient)
        erased_tags = os.listdir(path_to_patient)
        #print(erased_tags)
        for erased_tag in erased_tags:
            path_to_erased = os.path.join(path_to_patient, erased_tag)
            if not os.path.isdir(path_to_erased):     
                continue
            kfolds = os.listdir(path_to_erased)
            print(path_to_erased)
            knockoutKfoldRegion = KnockoutKfoldPerformance(len(kfolds), this_patient)            
            path_to_saved_folder_patient_eras = os.path.join(path_to_saved_folder_patient,erased_tag)
            erased_tag = parse_erased_tag(erased_tag)
            clean_folder(path_to_saved_folder_patient_eras)
            for k in kfolds:
                path_to_model = os.path.join(path_to_erased, k)
                #path_to_model = os.path.join(folder_name,erased_tag)
                plot_saved_folder_k = os.path.join(path_to_saved_folder_patient_eras, k)
                clean_folder(plot_saved_folder_k)
                if not os.path.isdir(path_to_model):     
                    continue
                knockout_region = model_inference_results_retrain(this_patient, plot_saved_folder_k,current_region=erased_tag ,kfolds = int(k) ,masked= True, erasing = True, path_to_model = path_to_model)
                knockoutKfoldRegion.add(knockout_region)
            acc_region_kfold, regions = knockoutKfoldRegion.get_acc_stats()
            loss_region_kfold, _ = knockoutKfoldRegion.get_loss_stats()
            plot_acc_loss([loss_region_kfold, acc_region_kfold], sorted(list(character_dict.keys())), regions, "loss_acc_region", path_to_erased, title = "Kfold loss_acc_region with knockout " + "_".join(erased_tag))

def main(patientNums = patientNums):
    multi_run()
    #draft()
    
if __name__ == '__main__':
    main()