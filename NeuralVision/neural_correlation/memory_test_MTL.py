################################################################################
# CNN_train
# Main function just run it
################################################################################

import sys
sys.path.append(".")
from neural_correlation.train_model import *
from neural_correlation.cnn import *
from neural_correlation.lstm import *
from neural_correlation.cnn import BasicCNN
from neural_correlation.memory_test_dataloader import create_split_loaders
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
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode, model_option
elif episode_number == 2:
    from data2 import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
        character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width, data_mode

# seed
'''
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''

def train_cnn_model_results(this_patient,masked = None, erasing = True, path_to_model = path_to_model, path_to_training_data = path_to_training_data, mode = 'train', kfold = 1, p_val=0.1):

    # Model_Setup:
    torch.cuda.empty_cache()
    if masked is None:
        direct = os.path.join(path_to_model, this_patient) # saved dir
        if not os.path.exists(direct):
            os.mkdir(direct)
        if mode is "test" and kfold != -1:
            direct = os.path.join(direct, str(kfold))     
    else: 
        direct = path_to_model
    # data filename
    path_to_training_data = "/media/yipeng/data/movie_2021/Movie_Analysis/mem_test"
    path_to_training_data_MTL = "/media/yipeng/data/movie_2021/Movie_Analysis/data_MTL_split/data_noMTL"
    data_fn = os.path.join(path_to_training_data_MTL, this_patient, "features.npy")
    label_fn = os.path.join(path_to_training_data, this_patient, "label.npy")
    frame_name_dir = os.path.join(path_to_training_data, this_patient, "frame_number.npy")
    if data_mode == "neuron" :
        tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_regions_clean.npy")
    if data_mode == "channel" :
        tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "channels.npy")
    # Setup: initialize the hyperparameters/variables
    num_epochs = 20          # Number of full passes through the dataset
    batch_size = 128          # Number of samples in each minibatch
    learning_rate = 0.001
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
    #transforms = torchvision.transforms.Compose([RandomBlank(aa)])
    transforms = None
    if model_option is "CNN":
        print(">>>")
        model = BasicCNN(num_characters, patient_features[this_patient])
    else:
        model = MovieLSTM(data_fn)
    model = model.to(computing_device)
    print("Model on CUDA?", next(model.parameters()).is_cuda)

    # Weights
    weight_np = np.zeros((num_characters,3)) + 1
    weight_np[:, -1] = 0
    weight_np[:, 0] = 2
    weights = Variable(torch.Tensor(weight_np)).cuda()

    # criterion
    #criterion = nn.MSELoss(reduce=False)
    criterion = nn.KLDivLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-6)
    #optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.1)
    train_loader = create_split_loaders(data_fn, label_fn, tag_fn , frame_name_dir, batch_size, seed, kfold=-1, transform=transforms ,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)
    if mode is 'train':
        if kfold == 1:
            one_train_session(direct,this_patient ,model, train_loader, val_loader,test_loader,optimizer, criterion, weights,
                    num_epochs, computing_device)
        else:
            for k in range(kfold):
                direct_one_fold = os.path.join(direct, str(k))
                clean_folder(direct_one_fold)
                model = BasicCNN(num_characters, patient_features[this_patient])
                model = model.to(computing_device) 
                optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay = 1e-6)
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
        outputs, labels, frame_name_all, _ = test(model_test, train_loader, computing_device)
        np.savez(os.path.join(direct, 'memtest_model_results'),  outputs = outputs, labels = labels, frame_number = frame_name_all)
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
    np.savez(os.path.join(direct, 'mem_test_model_results'), val_acc_all = val_acc_all, outputs = outputs, labels = labels, frame_names = frame_name_all , best_epoch = best_epoch, test_acc_epoch= test_acc_epoch)
    one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters)

def one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters, save_dnk_result = True):
    ## get stats for every frame
    person_dict = {}
    person_confusion = {}
    label_record = {}
    prob_record = {}
    for person_index in range(num_characters):
        frame_stats = {}
        confusion_stats= {}
        one_person_label = {}
        one_person_prob = {}
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
                one_person_prob[frame_num] = np.exp(batch_output[batch_index,person_index,0])
                if res_label[batch_index][person_index]==0 :
                    one_person_label[frame_num] = 1
        person_dict[person_index] = frame_stats
        person_confusion[person_index] = confusion_stats
        label_record[person_index] = one_person_label
        prob_record[person_index] = one_person_prob
        
    # stats over the whole movie
    prob_record_sorted = {}
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_stats = np.zeros(8)
        one_person_prob_sorted = []
        for i in range(23):
            one_person_prob_sorted.append(0)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            one_person_stats += one_person_frame_stats[frame_sorted]
            one_person_prob_sorted.append(prob_record[person_index][frame_sorted])
        prob_record_sorted[person_index] = np.array(one_person_prob_sorted)
        print(person_index, one_person_stats.astype(int))

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
    plot_tpfp(person_tpfp, person_label ,direct, fn="memory_test_noMTL.jpg")
    dump_pickle(os.path.join(direct,"memory_test_noMTL.pkl"),person_tpfp)
    dump_pickle(os.path.join(direct,"memory_test_prob_noMTL.pkl"),prob_record_sorted)

def multiple(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        if model_option is "CNN":
            print("?>")
            path_to_model = os.path.join(project_dir, 'CNN_result_zero','CNN_multi_'+str(feature_time_width)+'_KLD')
        else:
            path_to_model = os.path.join(project_dir,'CNN_result', 'LSTM_zero')
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        for k in range(5):  
            train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test", kfold=k)
    # for this_patient in patientNums:
    #     path_to_model = os.path.join(project_dir, 'CNN_result', 'LSTM_zero')
    #     if not os.path.exists(path_to_model):
    #         os.mkdir(path_to_model)

    #     train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test", kfold=5)
def multiple_train(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        if model_option is "CNN":
            print("?>")
            path_to_model = os.path.join(project_dir, 'CNN_result_zero','CNN_multi_'+str(feature_time_width)+'_KLD')
            path_to_model = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/CNN_multi_2_KLD"
        else:
            path_to_model = os.path.join(project_dir,'CNN_result', 'LSTM_multi_2_KLD')
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        for k in range(5):  
            train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test", kfold=k)

def standard_test(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        if model_option is "CNN":
            print("?>")
            path_to_model = os.path.join(project_dir, 'CNN_result_zero','CNN_multi_'+str(feature_time_width)+'_KLD')
            path_to_model = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
        else:
            path_to_model = os.path.join(project_dir,'CNN_result', 'LSTM_-50-50_shuffle')
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model) 
        train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test", kfold=-1)


def main(patientNums = patientNums):
    '''
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    this_patient = "431"
    train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")
    '''
    #standard_train_region_tag()
    multiple_train()
    #standard_test()

if __name__ == '__main__':
    main()
