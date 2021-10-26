################################################################################
# CNN_train
# Main function just run it
################################################################################

import sys
sys.path.append(".")
from itertools import combinations, permutations
from neural_correlation.train_model import *
from neural_correlation.cnn import *
from neural_correlation.cnn import BasicCNN
from neural_correlation.cnn_dataloader import create_split_loaders
from neural_correlation.utilities import *
from random_shuffle import RandomShuffle
from random_region import RandomRegion
from random_blank import RandomBlank
from image_expand import ImageExpand
from random_repeat import RandomRepeat
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from adabound import AdaBound
from collections import Counter
from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features, \
    character_dict, frame_dir, path_to_matlab_generated_movie_data, num_characters, feature_time_width
from itertools import permutations, combinations
# seed

'''
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed = 10
np.random.seed(seed)
random.seed(seed)
'''
def train_cnn_model_results(this_patient,exact_loc = None,masked = None, erasing = True, path_to_model = path_to_model, path_to_training_data = path_to_training_data, mode = 'train', kfold = 1):

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
    tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_regions.npy")
    # Setup: initialize the hyperparameters/variables
    num_epochs = 100          # Number of full passes through the dataset
    batch_size = 128          # Number of samples in each minibatch
    learning_rate = 0.0001
    seed = 10 # Seed the random number generator for reproducibility
    p_val = 0.1              # Percent of the overall dataset to reserve for validation
    p_test = 0.2             # Percent of the overall dataset to reserve for testing

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()
    # Setup GPU optimization if CUDA is supported
    if use_cuda:
        computing_device = torch.device("cuda")
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
    #transforms = torchvision.transforms.Compose([ImageExpand(exact_loc)])
    transforms = torchvision.transforms.Compose([RandomShuffle()])
    #transforms = None
    train_loader, val_loader, test_loader = create_split_loaders(data_fn, label_fn, tag_fn , frame_name_dir, batch_size, seed, kfold=-1, transform= transforms ,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)

    model = BasicCNN(num_characters, patient_features[this_patient])
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
                train_loader, val_loader, test_loader = create_split_loaders(data_fn, label_fn, tag_fn ,frame_name_dir ,batch_size, seed,p_val=p_val, p_test=p_test,shuffle=True,mask = mask, erasing= erasing , show_sample=False, extras=extras)
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
        np.savez(os.path.join(direct, 'test_model_results'),  outputs = outputs, labels = labels, frame_number = frame_name_all)
        aa = one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters)
        if aa:
            return True
def one_train_session(direct,this_patient ,model, train_loader, val_loader,test_loader,optimizer, criterion, weights,
                    num_epochs, computing_device):
    total_loss, val_acc_all, val_acc_all, best_epoch = train(model, train_loader, val_loader, optimizer, criterion, weights,
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
    np.savez(os.path.join(direct, 'model_results'), val_acc_all = val_acc_all, outputs = outputs, labels = labels, best_epoch = best_epoch)
    one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters)

def one_run_stats(outputs,labels,frame_name_all,direct ,num_characters=num_characters):
    ## get stats for every frame
    person_dict = {}
    person_confusion = {}
    label_record = {}
    for person_index in range(num_characters):
        frame_stats = {}
        confusion_stats= {}
        one_person_label = {}
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
                if res_label[batch_index][person_index]==0 :
                    one_person_label[frame_num] = 1
        person_dict[person_index] = frame_stats
        person_confusion[person_index] = confusion_stats
        label_record[person_index] = one_person_label
        
    # stats over the whole movie
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_stats = np.zeros(8)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            one_person_stats += one_person_frame_stats[frame_sorted]
        ##if one_person_stats[0] * 0.1/ (one_person_stats[1] + one_person_stats[1]) < 0.3:
        ##    return False
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
    #plotting
    #plot_tpfp(person_tpfp, person_label ,direct)
    #plot_false_frame(person_fp_fram_num, saved_dir=os.path.join(direct, "fp_plot"))
    #plot_false_frame(person_fn_fram_num, saved_dir=os.path.join(direct, "fn_plot"))
    return True

def standard_train(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    for this_patient in patientNums:
        train_cnn_model_results(this_patient,masked=None, erasing = False, path_to_model = path_to_model, mode = "train")

def standard_test(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    for this_patient in patientNums:
        train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")

def erasing_region(this_patient, dirct = os.path.join(project_dir,'CNN_result'), kfold = 1):
    path_to_model_dir = os.path.join(dirct,'CNN_retrain')
    if not os.path.exists(path_to_model_dir):
        os.mkdir(path_to_model_dir)
    path_to_model_dir = os.path.join(path_to_model_dir, this_patient)
    if not os.path.exists(path_to_model_dir):
        os.mkdir(path_to_model_dir)
    region_fn = os.path.join(path_to_matlab_generated_movie_data,this_patient,"features_mat_regions.npy")
    region = np.load(region_fn,allow_pickle=True)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_single = removeDuplicates(reg, len(reg))
    reg = np.array(reg) 
    for r in reg_single:
        path_to_model = os.path.join(path_to_model_dir, r)
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        mask = np.where(reg==r)[0]
        train_cnn_model_results(this_patient, masked=mask, erasing = True, path_to_model = path_to_model, mode = "train", kfold=kfold)

def multiple_train(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        path_to_model = os.path.join(project_dir, 'CNN_result','CNN_multi_'+str(feature_time_width))
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "train", kfold=5)

def multiple_erasing(patientNums = patientNums):
    for this_patient in patientNums:
        path_to_model = os.path.join(project_dir, 'CNN_result','CNN_erasing_multi_'+str(feature_time_width))
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        erasing_region(this_patient,path_to_model ,kfold=5)

def multiple_70(patientNums = patientNums, mask=None, project_dir = project_dir ):
    for this_patient in patientNums:
        path_to_model = os.path.join(project_dir, 'CNN_result','CNN_multi_70_'+str(feature_time_width))
        if not os.path.exists(path_to_model):
            os.mkdir(path_to_model)
        train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "train", kfold=5)

def standard_train_region_tag(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','region_tag')
    for this_patient in patientNums:
        train_cnn_model_results(this_patient,masked=None, erasing = False, path_to_model = path_to_model, mode = "train")

def standard_train_region_tag_70(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','region_tag')
    for this_patient in patientNums:
        train_cnn_model_results(this_patient,masked=None, erasing = False, path_to_model = path_to_model, mode = "train")

def location_test(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','region_tag')
    for this_patient in patientNums:
        aa = np.load("/media/yipeng/toshiba/movie/Movie_Analysis/data/431/features_mat_regions.npy")
        classnames, indices = np.unique(aa , return_inverse=True)
        a = dict(Counter(indices))
        bb = np.load("/media/yipeng/toshiba/movie/Movie_Analysis/data2/431/features_mat_regions.npy")
        classnames, indices = np.unique(bb , return_inverse=True)
        b = dict(Counter(indices))
        current_i = 0
        res = {}
        for k in sorted(a.keys()):
            perm_infor = []
            num_a = a[k]
            num_b = b[k]
            diff = num_a - num_b
            if diff > 0:
                index_list = list(range(current_i, num_a))
                combs = list(combinations(index_list, num_b))
            else:
                index_list = list(range(current_i, num_b))
                combs = list(combinations(index_list, num_a))
            perm_infor.append([diff])
            perm_infor.append([max(num_a, num_b)])
            perm_infor.append(combs)
            res[k] = perm_infor
        for itr in range(2000):
            final_comb = {}
            for k in res.keys():
                aa = list(random.choice(res[k][2]))
                random.shuffle(aa)
                final_comb[k] = [[res[k][0]], [res[k][1]],aa]
            train_cnn_model_results(this_patient, exact_loc=final_comb, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")

def location_test_2(patientNums = patientNums):
    path_to_model = os.path.join(project_dir, 'CNN_result','repeat_test')
    for this_patient in patientNums:
        aa = np.load("/media/yipeng/toshiba/movie/Movie_Analysis/data/431/features_mat_regions.npy")
        classnames, indices = np.unique(aa , return_inverse=True)
        a = dict(Counter(indices))
        bb = np.load("/media/yipeng/toshiba/movie/Movie_Analysis/data2/431/features_mat_regions.npy")
        classnames, indices = np.unique(bb , return_inverse=True)
        b = dict(Counter(indices))
        current_i = 0
        res = {}
        for k in sorted(a.keys()):
            perm_infor = []
            num_a = a[k]
            num_b = b[k]
            diff = num_a - num_b
            index_list = list(range(current_i, num_b))
            perm_infor.append([diff])
            perm_infor.append([num_a])
            perm_infor.append(index_list)
            res[k] = perm_infor
        for itr in range(1000):
            final_comb = {}
            for k in res.keys():
                num_a = a[k]
                num_b = b[k]
                aa = res[k][2] 
                random.shuffle(aa)
                final_comb[k] = [[res[k][0]], [res[k][1]],aa, aa[:abs(num_b - num_a)]]
            aa = train_cnn_model_results(this_patient, exact_loc=final_comb, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")
            if aa:
                return



def main(patientNums = patientNums):
    '''
    path_to_model = os.path.join(project_dir, 'CNN_result','CNN')
    this_patient = "431"
    train_cnn_model_results(this_patient, masked=None, erasing = False, path_to_model = path_to_model, mode = "test")
    '''
    standard_test(patientNums = patientNums)
    

if __name__ == '__main__':
    main()
