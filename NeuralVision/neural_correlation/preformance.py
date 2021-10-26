import numpy as np
import sys
sys.path.append(".")
import matplotlib.pyplot as plt
from project_setting import episode_number
from utilities import *
if episode_number == 1:
    from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode, path_to_cnn_result
elif episode_number == 2:
    from data2 import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode
import os 

num_epoch = 20 



def plot_accuarcy(acc_all,this_patient, direct):
    plt.figure()
    plt.plot(np.arange(1,len(acc_all)+1), acc_all)
    plt.title(this_patient + '  Testing Accuracy  ')
    plt.xlabel('Epoch Number')
    plt.savefig(os.path.join(direct, 'overall_test_accuracy.jpg'))
    plt.close('all')

def get_fp_correlation(character_stats, character_prediction):
    res = {}
    for character_index in sorted(character_stats.keys()):
        res[character_index] = {}
        for character_index_cond in sorted(character_stats.keys()):
            res[character_index][character_index_cond] = []

    for character_index in sorted(character_stats.keys()):
        one_character_stats = character_stats[character_index]
        for frame_number in one_character_stats.keys():
            one_frame_stats = one_character_stats[frame_number]
            if one_frame_stats[-2] == 1: #FP 
                for character_index_cond in sorted(character_stats.keys()):
                    res[character_index][character_index_cond].append(character_prediction[character_index_cond][frame_number])
    return res

def one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters, save_dnk_result = False):
    ## get stats for every frame
    person_dict = {}
    person_confusion = {}
    label_record = {}
    person_prediction = {} #YES prediction for each character
    for person_index in range(num_characters):
        frame_stats = {}
        confusion_stats= {}
        one_person_label = {}
        one_person_prediction = {}
        for i in range(len(outputs)):
            batch_label = labels[i]
            batch_output = outputs[i]
            frame_batch = frame_name_all[i]
            res_label = np.argmax(np.squeeze(batch_label, axis=1), axis=2)
            res_pre = np.argmax(batch_output, axis=2)
            for batch_index in range(len(res_label)):
                frame_num = frame_batch[batch_index]
                frame_stats[frame_num] = get_stats(res_label, res_pre,batch_index, person_index)
                confusion_stats[frame_num] = get_confusion(res_label, res_pre,batch_index, person_index)
                one_person_prediction[frame_num] =np.exp(batch_output[batch_index][person_index][0])
                if res_label[batch_index][person_index]==0 :
                    one_person_label[frame_num] = 1
        person_dict[person_index] = frame_stats
        person_confusion[person_index] = confusion_stats
        label_record[person_index] = one_person_label
        person_prediction[person_index] = one_person_prediction

    # stats over the whole movie
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_stats = np.zeros(8)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            one_person_stats += one_person_frame_stats[frame_sorted]
        print(person_index, one_person_stats.astype(int))

    # confusion over the whole movie
    all_characters_confusion = {}
    for person_index in sorted(person_confusion.keys()):
        one_person_frame_confusion = person_confusion[person_index]
        one_person_confusion = np.zeros((3,3))
        for frame_sorted in sorted(one_person_frame_confusion.keys()):
            one_person_confusion += one_person_frame_confusion[frame_sorted]
        all_characters_confusion[person_index] = one_person_confusion.astype(int)
    #plot_confusion_matrix(all_characters_confusion, direct)

    person_tpfp = {}
    person_label = {}
    person_fp_fram_num = {}
    person_fn_fram_num = {}
    for person_index in sorted(person_dict.keys()):
        person_fp_fram_num[person_index] = []
        person_fn_fram_num[person_index] = []
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_frame_label_stats = label_record[person_index]
        one_person_tpfp = np.zeros(18576)
        one_person_label = np.zeros(18576)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            if frame_sorted <= 0:
                continue 
            if frame_sorted in one_person_frame_label_stats:
                one_person_label[int(frame_sorted/4)] = 1
            if one_person_frame_stats[frame_sorted][-1] == 1:
                one_person_tpfp[int(frame_sorted/4)] = 1
            elif one_person_frame_stats[frame_sorted][-2] == 1:
                one_person_tpfp[int(frame_sorted/4)] = 2
                person_fp_fram_num[person_index].append(frame_sorted)
            elif one_person_frame_stats[frame_sorted][1] == 1:
                person_fn_fram_num[person_index].append(frame_sorted)
        person_tpfp[person_index] = one_person_tpfp
        person_label[person_index] = one_person_label
    correlation_dict = get_fp_correlation(person_dict, person_prediction)
    #plot_avg_stats(correlation_dict, direct)
    return all_characters_confusion

def plot_avg_stats( overall_stats, save_dir):
    fig, ax = plt.subplots(num_characters, num_characters, figsize=(20, 10))
    episode_number = 1
    for character_index_conditon in range(num_characters):
        plot_x = int(character_index_conditon)
        for character_index in range(num_characters):
            plot_y = int(character_index*1.0%num_characters)
            relative_character_prob = overall_stats[character_index][character_index_conditon]
            plot_by_episode(relative_character_prob, episode_number, ax[plot_x, plot_y])
            ax[plot_x, plot_y].set_title(str(character_index) + " condition on " + str(character_index_conditon))
            ax[plot_x, plot_y].legend()
    fig.suptitle("prob conditon FP ",fontsize="x-large")
    plt.savefig(os.path.join(save_dir, "test_cond_FP.jpg"))

def stack_up(kfold_characters_confusion,characters_confusion):
    for key in characters_confusion.keys():
        stat = np.expand_dims(characters_confusion[key], 0)
        if key in kfold_characters_confusion:
            kfold_characters_confusion[key] = np.concatenate((kfold_characters_confusion[key],stat))
        else:
            kfold_characters_confusion[key] = stat

def plot_stats_kfold(kfold_characters_confusion, save_dir):
    all_character_confusion = {}
    var_character_confusion = {}
    for character_idx in kfold_characters_confusion.keys():
        all_character_confusion[character_idx] = np.sum(kfold_characters_confusion[character_idx], axis=0)
    plot_confusion_matrix(all_character_confusion, save_dir, filename="avg_confusion_matrix")
    

    each_character_confusion = {}
    for character_idx in kfold_characters_confusion.keys():
        each_character_confusion[character_idx] = []
        for i in kfold_characters_confusion[character_idx] :
            matrix = i/np.sum(i,axis=1)[:,None]
            each_character_confusion[character_idx].append(matrix)
        var_character_confusion[character_idx] = np.var(each_character_confusion[character_idx], axis=0)
    plot_var_confusion_matrix(var_character_confusion, save_dir, filename="var_confusion_matrix")


def multiple_test(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        #path_to_patient = os.path.join("CNN_result_zero",'CNN_multi_'+str(feature_time_width)+"_KLD", this_patient)
        path_to_patient = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD/" + this_patient
        test_acc_all = np.zeros(num_epoch)
        kfolds = os.listdir(path_to_patient)
        kfold_characters_confusion = {}
        for k in kfolds:
            folder_name = os.path.join(path_to_patient, k)
            if not os.path.isdir(folder_name):
                continue
            fn = os.path.join(folder_name, "model_results.npz")
            stats = np.load(fn, allow_pickle=True)
            labels = stats["labels"]
            outputs = stats["outputs"]
            frame_name = stats["frame_names"]
            characters_confusion = one_run_stats(outputs,labels,frame_name,folder_name,num_characters=num_characters, save_dnk_result = False)
            stack_up(kfold_characters_confusion, characters_confusion)
            #print(np.array(stats["test_acc_epoch"]))
            #test_acc_all += np.array(stats["test_acc_epoch"])
        #print(test_acc_all)
        #test_acc_all = test_acc_all/5
        #plot_accuarcy(test_acc_all,this_patient, path_to_patient)
        plot_stats_kfold(kfold_characters_confusion, path_to_patient)

def single(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        #path_to_patient = os.path.join(path_to_cnn_result,'CNN_multi_'+str(feature_time_width)+"_KLD", this_patient)
        path_to_patient = os.path.join("/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/LSTM_70_30_batch_64", this_patient)
        test_acc_all = np.zeros(num_epoch)
        kfolds = os.listdir(path_to_patient)
        folder_name = os.path.join(path_to_patient)
        if not os.path.isdir(folder_name):
            continue
        fn = os.path.join(folder_name, "model_results.npz")
        stats = np.load(fn, allow_pickle=True)
        labels = stats["labels"]
        outputs = stats["outputs"]
        frame_name = stats["frame_names"]
        one_run_stats(outputs,labels,frame_name,folder_name,num_characters=num_characters, save_dnk_result = False)
            #print(np.array(stats["test_acc_epoch"]))
            #test_acc_all += np.array(stats["test_acc_epoch"])
        #print(test_acc_all)
        #test_acc_all = test_acc_all/5
        #plot_accuarcy(test_acc_all,this_patient, path_to_patient)


def main(patientNums = patientNums):
    multiple_test()

if __name__ == '__main__':
    main()
