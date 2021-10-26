import sys
sys.path.append(".")
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
sys.path.append("./movietag_processing")
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np 
import sys
from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
    character_dict, path_to_matlab_generated_movie_data, num_characters, feature_time_width, model_option
from neural_correlation.KnockoutPerformance import KnockoutPerformance, KnockoutKfoldPerformance
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
from neural_correlation.utilities import load_pickle, dump_pickle
from neural_correlation.MemtestKfoldStats import KfoldStats
from movietag_processing.EpisodeStats import EpisodeStates

#for Z to run these
#from code_refactor.Neural_analysis.Neural_network.src.utilities import *
#from code_refactor.Neural_analysis.Neural_network.src.performance_analysis import *
#from data import len_dict, num_characters, patientNums
#from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance

'''
utilities
'''
def removeDuplicates(arr): 
    res = []
    res_set = set()
    for a in arr: 
        if a not in res_set:
            res_set.add(a)
            res.append(a)
    return res

def get_trim_index(tag1):
    jump_diff = np.diff(tag1)
    jump_index = np.where(jump_diff)[0]
    jump_index = np.append(jump_index, len(tag1)-1) 
    return jump_index

def get_sum(reg,reg_single,region_loss,mircowire_loss):
    sum_mircowire = region_loss * 0
    for k in range(len(reg_single)):
        loc = np.where(reg==reg_single[k])[0]
        if len(loc) == 1:
            sum_mircowire[:,loc] = mircowire_loss[:,loc]    
            continue
        sum_mircowire[:,loc] = np.tile(np.expand_dims(np.sum(mircowire_loss[:,loc], axis = 1), axis = 1), (1, len(loc)))
    return sum_mircowire

def parse_key(char, cond_char):
    return str(char) + "|"+ str(cond_char)

def normalize(prob, num_character):
    res = np.zeros((num_character, num_character))
    for cond_char in range(num_character):
        for char in range(num_character):
            if cond_char == char:
                continue
            res[char, cond_char] = prob[parse_key(char,cond_char)]
    res_normalize = np.divide(res, np.sum(res,axis=0))
    return res_normalize 

def two_step(prob):
    res = np.zeros((len(prob), len(prob)))
    for cond_char in range(len(prob)):
        for char in range(len(prob)):
            if char == cond_char:
                res[char, cond_char] = 1 
                continue  
            walk_prob = prob[char,cond_char]
            for walk_char in range(len(prob)):      
                if walk_char == char or walk_char == cond_char :
                    continue
                walk_prob = walk_prob + 1.0*prob[char, walk_char] * prob[walk_char,cond_char]
            res[char,cond_char] = walk_prob 
    return res


'''
main
'''
def figure2b(path, p, fold, output_folder, model_option="LSTM"):
    path = os.path.join(path, p, str(fold), "model_results.npz")
    model_info = np.load(path, allow_pickle=True)
    onefold_model = ModelPerformance.generate(fold, p, model_option, model_info['outputs'],model_info['labels'], model_info['frame_names'])
    stats_board, label_board = onefold_model.get_tpfp()
    fn = os.path.join(output_folder,"figure2b.mat")
    res = {'stats_tpfp':stats_board,'stats_label':label_board}
    savemat(fn,res)

def figure2c(path, p, fold, output_folder, model_option="LSTM"):
    path = os.path.join(path, p, str(fold), "model_results.npz")
    model_info = np.load(path, allow_pickle=True)
    onefold_model = ModelPerformance.generate(fold, p, model_option, model_info['outputs'],model_info['labels'], model_info['frame_names'])
    character_confusion_mat = onefold_model.get_character_confusion_mat()
    fn = os.path.join(output_folder,"figure2c.mat")
    res = {'character_confusion_mat':character_confusion_mat}
    savemat(fn,res)

def figure2def(path, output_folder):
    patientNums = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
    stats = {}
    for p in patientNums:
        path_p = os.path.join(path, p )
        kfoldPerformance = KfoldPerformance.generate_bypath(path_p)
        character_acc= kfoldPerformance.get_character_accuracy()
        #overall_acc = kfoldPerformance.get_accuracy()
        #confusion_mat = kfoldPerformance.get_confusion_mat()
        character_confusion_mat = kfoldPerformance.get_character_confusion_mat()
        #character_confusion_mat_var = kfoldPerformance.get_character_confusion_mat_var()
        character_f1_score = kfoldPerformance.get_character_f1_score()
        stats["p_" + p] = {"character_confusion_mat":character_confusion_mat, "character_acc":character_acc,"character_f1_score":character_f1_score}
    fn = os.path.join(output_folder,"figure2def.mat")
    res = {'stats':stats}
    savemat(fn,res)

'''
patient region knockout test
'''
def figure3a(preprocess_filename, patient, output_folder):
    preprocess = load_pickle(preprocess_filename)
    print(preprocess.keys())
    loss_normalized = preprocess[patient]["region_loss_normalized"]
    tag_unique = preprocess[patient]["region_label"]
    res = {"loss":loss_normalized,"tag_unique":tag_unique, "patientNum":patient}
    fn = os.path.join(output_folder,"figure3a.mat")
    savemat(fn,res)

'''
ALL patient region knockout test
'''
def figure3b(preprocess_filename, output_folder):
    preprocess = load_pickle(preprocess_filename)
    patientNums = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
    res = {}
    for p in patientNums:
        loss_normalized = preprocess[p]["region_loss_normalized"]
        tag_unique = preprocess[p]["region_label"]
        loss_normalized_mw = preprocess[p]["mircowire_loss_normalized"]
        tag_unique_mw = preprocess[p]["reg_trim"]
        res["p_" + p] = {"loss":loss_normalized,"tag_unique":tag_unique, "loss_mw":loss_normalized_mw,"tag_unique_mw":tag_unique_mw, "patientNum":p}
    fn = os.path.join(output_folder,"figure3b.mat")
    savemat(fn,res)

'''
patient mircowire knockout test
'''       
def figure3c(preprocess_filename, patient, output_folder):
    preprocess = load_pickle(preprocess_filename)
    loss_normalized = preprocess[patient]["mircowire_loss_normalized"]
    tag_unique = preprocess[patient]["mircowire_label"]
    res = {"loss":loss_normalized,"tag_unique":tag_unique, "patientNum":patient}
    fn = os.path.join(output_folder,"figure3c.mat")
    savemat(fn,res)

def figure3d():
    pass


'''
patient sum_mircowire - region
'''  
def figure3e(preprocess_filename, patient, output_folder):
    preprocess = load_pickle(preprocess_filename)
    region_loss = preprocess[patient]["region_loss_exp"]
    sum_mircowire_loss = preprocess[patient]["sum_mircowire_loss_region"]
    res = {"region_loss":region_loss,"sum_mircowire_loss":sum_mircowire_loss}
    fn = os.path.join(output_folder,"figure3e.mat")
    savemat(fn,res)

'''
ALL patient sum_mircowire - region
''' 
def figure3f(preprocess_filename, output_folder):
    preprocess = load_pickle(preprocess_filename)
    patientNums = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
    res = {}
    for p in patientNums:
        region_loss = preprocess[p]["region_loss_exp"]
        sum_mircowire_loss = preprocess[p]["sum_mircowire_loss_region"]
        res["p_" + p] = {"region_loss":region_loss,"sum_mircowire_loss":sum_mircowire_loss, "patientNum":p}
    fn = os.path.join(output_folder,"figure3f.mat")
    savemat(fn,res)

def figure3_preprocess(folder, input_folder ,output_folder):
    patientNums = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
    stats = {}
    kfold = 5
    for patient in patientNums:
        reg_np = load_pickle(os.path.join(input_folder,patient,"regions.pkl"))
        microarr_np = load_pickle(os.path.join(input_folder,patient,"microwire.pkl"))
        knockoutKfold_region = KnockoutKfoldPerformance(kfold, patient)
        knockoutKfold_mircowire = KnockoutKfoldPerformance(kfold, patient)
        for k in range(kfold):
            knockout_region = KnockoutPerformance.construct_analysis(folder,patient,k,"reg")
            knockout_region.set_tag(reg_np)
            knockoutKfold_region.add(knockout_region)

            knockout_mircowire = KnockoutPerformance.construct_analysis(folder,patient,k,"mircowire")
            knockout_mircowire.set_tag(microarr_np)
            knockoutKfold_mircowire.add(knockout_mircowire)
        
        knockoutKfold_region_f1, region_label = knockoutKfold_region.get_f1_stats()
        knockoutKfold_region_acc, _ = knockoutKfold_region.get_acc_stats()
        knockoutKfold_region_loss, _ = knockoutKfold_region.get_loss_stats()
        knockoutKfold_region_loss_normalized, _ = knockoutKfold_region.get_loss_stats_normalized()

        knockoutKfold_mircowire_f1, mircowire_label = knockoutKfold_mircowire.get_f1_stats()
        knockoutKfold_mircowire_acc, _ = knockoutKfold_mircowire.get_acc_stats()
        knockoutKfold_mircowire_loss, _ = knockoutKfold_mircowire.get_loss_stats()
        knockoutKfold_mircowire_loss_normalized, _ = knockoutKfold_mircowire.get_loss_stats_normalized()

        trim_idx = get_trim_index(microarr_np)
        reg_trim = reg_np[trim_idx]

        knockoutKfold_region_loss_exp = knockoutKfold_region.get_loss_tag_expand()[trim_idx].T
        knockoutKfold_mircowire_loss_exp = knockoutKfold_mircowire.get_loss_tag_expand()[trim_idx].T

        reg_single = removeDuplicates(reg_np)    
        knockoutKfold_region_mircowire_exp = get_sum(reg_trim, reg_single,knockoutKfold_region_loss_exp,knockoutKfold_mircowire_loss_exp)

        res = {
            "regionPerneuron":reg_np,
            "region_label" :region_label,
            "region_f1": knockoutKfold_region_f1,
            "region_acc":knockoutKfold_region_acc,
            "region_loss":knockoutKfold_region_loss,
            "region_loss_normalized":knockoutKfold_region_loss_normalized,

            "mircowirePreregion":microarr_np,
            "mircowire_label":mircowire_label,
            "mircowire_f1":knockoutKfold_mircowire_f1,
            "mircowire_loss":knockoutKfold_mircowire_loss,
            "mircowire_acc":knockoutKfold_mircowire_acc,
            "mircowire_loss_normalized":knockoutKfold_mircowire_loss_normalized,

            "reg_trim":reg_trim,
            "region_loss_exp":knockoutKfold_region_loss_exp,
            "mircowire_loss_exp":knockoutKfold_mircowire_loss_exp,
            "sum_mircowire_loss_region":knockoutKfold_region_mircowire_exp
        }
        stats[patient] = res
    dump_pickle(os.path.join(output_folder, "figure3_preprocess.pkl"),stats)

'''
character_in and out
'''
def figure4a(path, output_folder,input_folder,mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    stats = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5,input_folder, mem_test_fn)
        kfold_prediction_4sec, character_each_clip, episode_each_clip = kfoldStats.prediction_on_edge()
        stats["p_" + p] = {"kfold_prediction_4sec":kfold_prediction_4sec, "character_each_clip":character_each_clip, "episode_each_clip": episode_each_clip}
    fn = os.path.join(output_folder,"figure4a.mat")
    savemat(fn,stats)

'''
size and percentage
'''
def figure4b_percentage(path, output_folder,input_folder,mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    benchmark = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5,input_folder,mem_test_fn)
        patient_benchmark = kfoldStats.memtest_benchmark_search(mode="percentage")
        benchmark["p_" + p] = np.array(patient_benchmark)
    fn = os.path.join(output_folder,"figure4b_percentage.mat")
    savemat(fn,benchmark)

def figure4b_size(path, output_folder, input_folder, mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    benchmark = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5, input_folder,mem_test_fn)
        patient_benchmark = kfoldStats.memtest_benchmark_search(mode="size")
        benchmark["p_" + p] = np.array(patient_benchmark)
    fn = os.path.join(output_folder,"figure4b_size.mat")
    savemat(fn,benchmark)

'''
response
'''
def figure4c(path, output_folder,input_folder, mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    stats = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5,input_folder, mem_test_fn)
        kfold_prediction_4sec, character_each_clip, episode_each_clip = kfoldStats.prediction_on_edge()
        responses = kfoldStats.get_all_response()
        stats["p_" + p] = {"kfold_prediction_4sec":kfold_prediction_4sec, "character_each_clip":character_each_clip, "responses":responses,"episode_each_clip": episode_each_clip}
    fn = os.path.join(output_folder,"figure4c.mat")
    savemat(fn,stats)

def figure4_generate_cv_association(output_folder, input_folder, mem_test_fn):
    annotation_file = os.path.join(input_folder, "compare_movie_annotation_allresearchers_v2_40m_act_24_S06E01_30fps_ft1.mat")
    cv_label_file = os.path.join(input_folder, "cv_label")
    episodeStates = EpisodeStates(annotation_file, 1, cv_label_file)
    one_step = episodeStates.character_association()
    one_step_res = []

    for i in range(4):
        for j in range(4):
            key = str(i) + "|" + str(j)
            one_step_res.append(one_step[key])

    one_step_normalize = normalize(one_step,9)
    two_step_ass = two_step(one_step_normalize)
    dump_pickle(os.path.join(output_folder, "figure4_one_step.pkl"), one_step_res)
    dump_pickle(os.path.join(output_folder, "figure4_two_step.pkl"), two_step_ass[:4,:4])

'''
first order
'''
def figure4d(path, output_folder, input_folder, mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    stats = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5, input_folder, mem_test_fn)
        average_stats = kfoldStats.average_stats_kfold(mode="prob", clip_mode="clip")
        stats["p_" + p] = np.array(average_stats)
        #print(p, average_stats["patient_acc_stats"])
    fn = os.path.join(output_folder,"figure4d.mat")
    data = load_pickle(os.path.join(output_folder,"figure4_one_step.pkl"))
    
    stats["reference"] = data
    savemat(fn,stats)

'''
second order
'''
def figure4e(path, output_folder, input_folder, mem_test_fn):
    patientNums = ['431', '435', '436', '441']
    stats = {}
    for p in patientNums:
        kfoldStats = KfoldStats(p, path, 5, input_folder, mem_test_fn)
        average_stats = kfoldStats.average_stats_kfold(mode="prob", clip_mode="response")
        stats["p_" + p] = np.array(average_stats)
    fn = os.path.join(output_folder,"figure4e.mat")
    data = load_pickle(os.path.join(output_folder,"figure4_two_step.pkl"))
    stats["reference"] = data.flatten()
    #print(data.T)
    #print(data.T.flatten())
    savemat(fn,stats)

def main():
    
    """
    Usage:
        you basically need 4 external folders, 
        "LSTM_multi_2_KLD, CNN_multi_2_KLD, knockout_test_LSTM_KLD, knockout_test_CNN_KLD"
        First two is for Figure2 and Figure4
        Second one is for Figure3 
        How to run this file:
        1. you need to change the project_dir path in data/__init__.py to the current top level
        2. you change the path in the following lines.
        3. all the necessary inputs of running file are in the input_folder/
        4. memtest will be done soon
    """
    
    project_folder = "/media/yipeng/data/movie_2021/Movie_Analysis"
    path = "/media/yipeng/data/movie_2021/Movie_Analysis/paper_results/LSTM_multi_2_KLD"
    #path = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
    #path = "/media/yipeng/data/movie_2021/Movie_Analysis/paper_results/CNN_multi_2_KLD"
    #path = "/media/yipeng/data/movie/data_movie_analysis_final/d"
    result_folder = os.path.join(project_folder,"final_result_outputs_old_feb")
    MTL = True
    use_original = True
    if "LSTM" in path:
        if MTL:
            model_option = "LSTM_MTL"
            mem_test_fn = "memory_test_prob_MTL.pkl"
        else:
            model_option = "LSTM_noMTL"
            mem_test_fn = "memory_test_prob_noMTL.pkl"
        if use_original:
            model_option = "LSTM"
            mem_test_fn = "memory_test_prob.pkl"
    else:
        model_option = "CNN"
    output_folder = os.path.join(result_folder,model_option)
    

    ######Figure 2#####
    figure2_patient = "431"
    figure2_fold = 0
    figure2b(path,figure2_patient,figure2_fold,output_folder=output_folder)
    figure2c(path,figure2_patient,figure2_fold,output_folder=output_folder)
    figure2def(path,output_folder)
    print("*****************Figure 2 Done ************************")
    
    ######Figure 3#####
    ## what to do with retrain (Fscore and loss both has their own reason)

    knockout_path = "knockout_test_LSTM_KLD"
    #knockout_path = "knockout_test_CNN_KLD"
    input_folder = os.path.join(project_folder,"final_result_inputs")
    if "LSTM" in path:
        if MTL:
            model_option = "LSTM_MTL"
            mem_test_fn = "memory_test_prob_MTL.pkl"
        else:
            model_option = "LSTM_noMTL"
            mem_test_fn = "memory_test_prob_noMTL.pkl"
        if use_original:
            model_option = "LSTM"
            mem_test_fn = "memory_test_prob.pkl"
    else:
        model_option = "CNN"
    output_folder = os.path.join(result_folder,model_option)
    figure3_preprocess(knockout_path, input_folder,output_folder) ##you only need to run this one time
    preprocess_filename = os.path.join(output_folder, "figure3_preprocess.pkl") 
    figure3_patient = "431"
    figure3a(preprocess_filename, figure3_patient, output_folder)
    figure3b(preprocess_filename, output_folder)
    figure3c(preprocess_filename, figure3_patient, output_folder)
    figure3d() #same as figure 3f
    figure3e(preprocess_filename, figure3_patient, output_folder)
    figure3f(preprocess_filename, output_folder)
    # print("*****************Figure 3 Done ************************")

    ######Figure 4#####
    input_folder = os.path.join(project_folder,"final_result_inputs")
    figure4_generate_cv_association(output_folder, input_folder, mem_test_fn)
    figure4a(path, output_folder,input_folder, mem_test_fn)
    figure4b_size(path, output_folder, input_folder, mem_test_fn)
    figure4b_percentage(path, output_folder, input_folder,mem_test_fn)
    figure4c(path, output_folder, input_folder, mem_test_fn)
    figure4d(path, output_folder, input_folder,mem_test_fn)
    figure4e(path, output_folder, input_folder,mem_test_fn)
    print("*****************Figure 4 Done ************************")


if __name__ == "__main__":
    main()