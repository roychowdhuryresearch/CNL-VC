################################################################################
# CNN_train
# Main function just run it
################################################################################

import sys
#sys.path.append(".")
sys.path.insert(0, "/media/yipeng/data/movie/Movie_Analysis")
from neural_correlation.train_model import *
from neural_correlation.cnn import *
from neural_correlation.lstm import *
from neural_correlation.cnn import BasicCNN
from neural_correlation.cnn_dataloader import create_split_loaders
from neural_correlation.utilities import *
from random_shuffle import RandomShuffle
from random_region import RandomRegion
from random_blank import RandomBlank
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from adabound import AdaBound
from project_setting import episode_number
if episode_number == 1:
    from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode, zero_signal,model_option
elif episode_number == 2:
    from data2 import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode

# seed
'''
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
'''
def train_cnn_model_results(this_patient,masked = None, erasing = True, path_to_model = path_to_model, path_to_training_data = path_to_training_data, mode = 'train', kfold = 1, p_val=0.1):

    # Model_Setup:
    torch.cuda.empty_cache()
    if masked is None:
        direct = os.path.join(path_to_model, this_patient) # saved dir
        if not os.path.exists(direct):
            os.mkdir(direct)
    else: 
        direct = path_to_model
    # data filename
    data_fn = os.path.join(path_to_training_data, this_patient, "feature.npy")
    label_fn = os.path.join(path_to_training_data, this_patient, "label.npy")
    frame_name_dir = os.path.join(path_to_training_data, this_patient, "frame_number.npy")
    if data_mode == "neuron" :
        tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_regions_clean.npy")
    if data_mode == "channel" :
        tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "channels.npy")
    # Setup: initialize the hyperparameters/variables
    num_epochs = 60         # Number of full passes through the dataset
    batch_size = 128          # Number of samples in each minibatch
    #batch_size = 64
    #batch_size = 16
    learning_rate = 0.001
    #learning_rate = 0.00001
    seed = 0 # Seed the random number generator for reproducibility
    #p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing
	
    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda:0")
        extras = {"num_workers": 1, "pin_memory": True}
        print("CUDA is supported")
    else: # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")
    # apply mask
    if masked is not None:
        mask = masked
    else: 
        mask = None
    # Setup the training, validation, and testing dataloaders
    #transforms = torchvision.transforms.Compose([RandomShuffle()])
    #transforms = torchvision.transforms.Compose([RandomRegion()])
    aa = {2: 21, 5: 19, 7: 22, 8: 18, 1: 19, 0: 17, 6: 17, 3: 19, 4: 16}
    #transforms = torchvision.transforms.Compose([RandomBlank(aa)])
    transforms = None
    
    if model_option == "CNN":
        model = BasicCNN(num_characters, patient_features[this_patient])
    else:
        model = MovieLSTM(data_fn)
        #model = MovieLSTM_reg(data_fn, tag_fn)

    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # Weights
    weight_np = np.zeros((num_characters,3)) + 1
    weight_np[:, -1] = 0
    weight_np[:, 0] = 1
    weights = Variable(torch.Tensor(weight_np)).cuda()

    # criterion
    #criterion = nn.MSELoss(reduce=False)
    criterion = nn.KLDivLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-6)
    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    #optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
    train_loader, val_loader, test_loader = create_split_loaders(data_fn, label_fn, tag_fn , frame_name_dir, batch_size, seed, kfold=-1, transform=transforms ,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)
    if mode is 'train':
        if kfold == 1 or kfold == -1:
            one_train_session(direct,this_patient ,model, train_loader, val_loader,test_loader,optimizer, criterion, weights,
                    num_epochs, computing_device)
        else:
            for k in range(kfold):
                direct_one_fold = os.path.join(direct, str(k))
                clean_folder(direct_one_fold)
                if model_option == "CNN":
                    model = BasicCNN(num_characters, patient_features[this_patient])
                    model = model.to(computing_device) 
                    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-6)
                else:
                    model = MovieLSTM(data_fn)
                    model = model.to(computing_device) 
                    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-6)
                    #optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
                    #optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
                train_loader, val_loader, test_loader = create_split_loaders(data_fn, label_fn, tag_fn ,frame_name_dir ,batch_size, seed,kfold=k, p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)
                one_train_session(direct_one_fold, this_patient, model, train_loader, val_loader,test_loader,optimizer, criterion, weights,
                    num_epochs, computing_device)
    elif mode is 'test':
        #load the best epoch number from the saved "model_results" structure
        model_dirct = direct + '/best.pth'
        print('Resume model: %s' % model_dirct)
        model_test = model.to(computing_device)
        check_point = torch.load(model_dirct)
        model_test.load_state_dict(check_point['state_dict'])
        outputs, labels, frame_name_all, _ = test(model_test, test_loader, computing_device)
        np.savez(os.path.join(direct, 'test_model_results'),  outputs = outputs, labels = labels, frame_number = frame_name_all)
        one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters)
def one_train_session(direct,this_patient ,model, train_loader, val_loader,test_loader,optimizer, criterion, weights,
                    num_epochs, computing_device):
    total_loss, val_acc_all, val_acc_all, best_epoch, test_acc_epoch = train(model, train_loader,val_loader, test_loader, optimizer, criterion, weights,
                                                                    num_epochs, computing_device, direct)
    #model_dirct = direct + '/model_epoch' + str(best_epoch) + '.pth'
    model_dirct = direct + "/best.pth"
    print('Resume model: %s' % model_dirct)
    #shutil.copyfile(model_dirct,direct+"/best.pth")
    model_test = model.to(computing_device)
    check_point = torch.load(model_dirct)
    model_test.load_state_dict(check_point['state_dict'])
    outputs, labels, frame_name_all, _ = test(model_test, test_loader, computing_device)
    plot_validation_accuarcy(val_acc_all,this_patient,best_epoch, direct)
    np.savez(os.path.join(direct, 'model_results'), val_acc_all = val_acc_all, outputs = outputs, labels = labels, frame_names = frame_name_all , best_epoch = best_epoch, test_acc_epoch= test_acc_epoch)
    one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters)

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
    plot_confusion_matrix(all_characters_confusion, direct)

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
            if save_dnk_result :
                dont_know_analysis(one_person_frame_stats[frame_sorted],person_index, frame_sorted, direct)
        person_tpfp[person_index] = one_person_tpfp
        person_label[person_index] = one_person_label
    #plotting
    plot_tpfp(person_tpfp, person_label ,direct)
    #plot_false_frame(person_fp_fram_num, saved_dir=os.path.join(direct, "fp_plot"))
    #plot_false_frame(person_fn_fram_num, saved_dir=os.path.join(direct, "fn_plot"))

def erasing_multiple_region(this_patient, dirct = os.path.join(project_dir,'important_retrain'), kfold = 5):
    stats_folder = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
    #stats_folder = "/media/yipeng/data/movie/data_movie_analysis_final/CNN_multi_2_KLD"
    factor = 0.8 
    path_to_model_dir = os.path.join(dirct,model_option+'_important_retrain_normalized_loss_'+str(factor))
    if not os.path.exists(path_to_model_dir):
        os.mkdir(path_to_model_dir)
    path_to_model_dir = os.path.join(path_to_model_dir, this_patient)
    if not os.path.exists(path_to_model_dir):
        os.mkdir(path_to_model_dir)
    region_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn,allow_pickle=True)

    reg = []
    for d in region:
        reg.append(d[0])
    knockout_stats = load_pickle(os.path.join(stats_folder, this_patient , "knockout_final.pkl"))
    knockout_region = knockout_stats["region_label"][1:]
    knockout_f1 = np.mean(knockout_stats["region_loss_normalized"][1:], axis=1)
    tags = ["important", "non_important"]
    reg = np.array(reg)
    print(knockout_f1)
    print(np.mean(knockout_f1))
    reg_non_important = np.where(knockout_f1>=factor *np.mean(knockout_f1))[0]    
    reg_important = np.where(knockout_f1 < factor * np.mean(knockout_f1))[0]
    
    '''
    _erasing_hemisphere_
    reg_non_important = []
    reg_important = []
    for idx in range(len(knockout_region)):
        if knockout_region[idx][0] == "R":
            reg_important.append(idx)
        else:
            reg_non_important.append(idx)
    reg_non_important = np.array(reg_non_important)
    reg_important = np.array(reg_important)
    '''
    print(knockout_region)
    print(reg_non_important)
    print(reg_important)
    if len(reg_important)==0 or len(reg_non_important) == 0:
        return
    for idx ,reg_index in enumerate([reg_important, reg_non_important]):
        reg_single = knockout_region
        masks = []
        for r_idx in reg_index:
            r = reg_single[int(r_idx)]
            mask = np.where(reg==r)[0]
            masks.append(mask)
        masks = np.concatenate(masks)
        region_erased = np.array(reg_single)[np.array(reg_index)]
        path_to_model = os.path.join(path_to_model_dir, "_".join(region_erased) + "_"+str(len(masks)))
        print("____________________________________")
        print(tags[idx])
        print(reg_index) 
        print(knockout_f1[reg_index])
        print("num of neurons erased is ",len(masks))
        print(path_to_model)
        if not os.path.exists(path_to_model):
           os.mkdir(path_to_model)
        train_cnn_model_results(this_patient, masked=masks, erasing = True, path_to_model = path_to_model, mode = "train", kfold=kfold)

def multiple_train(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        if zero_signal:
            path_to_model = os.path.join(project_dir, 'CNN_result',model_option+'_multi_'+str(feature_time_width) +'_KLD_null')
        else:
            path_to_model = os.path.join(project_dir, 'CNN_result','CNN_multi_check_'+str(feature_time_width)+'_KLD')
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "train", kfold=5)

def multiple_erasing(patientNums = patientNums):
    for this_patient in patientNums:
        #erasing_region(this_patient,path_to_model ,kfold=5)
        erasing_multiple_region(this_patient ,kfold=5)

def main(patientNums = patientNums):
    '''
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    this_patient = "431"
    train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")
    '''
    #multiple_keep()
    multiple_erasing()
    #multiple_train()
    #standard_test()

if __name__ == '__main__':
    main()
