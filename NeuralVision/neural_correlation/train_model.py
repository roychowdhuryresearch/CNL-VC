################################################################################
# train_model
# define major training functions for CNN
################################################################################

import numpy as np
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
from project_setting import episode_number
if episode_number == 1:
    from data import num_characters
elif episode_number == 2:
    from data2 import num_characters
import matplotlib.pyplot as plt
import numpy as np
import os
from utilities import get_stats, get_confusion, cal_f1_score
def check_first_second(pred,label):
    pre = pred.cpu().detach().numpy()
    res_pre = np.argmax(np.squeeze(pre), axis=2) 
    label = label.cpu().detach().numpy()
    res_label = np.argmax(np.squeeze(label), axis=2)
    first_correct = 0 
    first_wrong = 0 
    second_correct = 0
    second_wrong = 0
    third_correct = 0 
    third_wrong = 0 
    forth_correct = 0
    forth_wrong = 0
    for batch_index in range(len(res_label)):
        if res_label[batch_index][0] == 0:
            if res_pre[batch_index][0] == 0:
                first_correct += 1
            else:
                first_wrong += 1
        if res_label[batch_index][1] == 0:
            if res_pre[batch_index][1] == 0:
                second_correct += 1
            else:
                second_wrong += 1
        if res_label[batch_index][2] == 0:
            if res_pre[batch_index][2] == 0:
                third_correct += 1
            else:
                third_wrong += 1
        if res_label[batch_index][3] == 0:
            if res_pre[batch_index][3] == 0:
                forth_correct += 1
            else:
                forth_wrong += 1
    return first_correct , first_wrong, second_correct, second_wrong,  third_correct , third_wrong, forth_correct, forth_wrong 
def accuracy(pred,label):
    pre = pred.cpu().detach().numpy()
    res_pre = np.argmax(pre, axis=2) 
    label = label.cpu().detach().numpy()
    res_label = np.argmax(np.squeeze(label, axis = 1), axis=2)
    sum_correct = np.sum(res_pre==res_label)

    return sum_correct/float(pre.shape[0]*pre.shape[1])

def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

#deprecated method     
def train_input(model, test_loader, computing_device):
    N = 0   
    N_minibatch_acc = 0.0  
    outputs_all = [] 
    label_all = [] 
    frame_name_all = [] 
    for minibatch_count, (images, labels, frame_name) in enumerate(test_loader, 0):
        model.eval()
        images, labels = images.to(computing_device), labels.to(computing_device)
        images = images.float()
        labels = labels.float()
        images.requires_grad = True
        labels = labels.cpu().detach().numpy()
        res_label = np.argmax(np.squeeze(labels, axis = 1), axis=2)
        for ii in range(len(res_label[:,0])):
            print(ii, str(" ") , res_label[ii,0])
        optimizer = optim.Adam([images],lr=0.001)  
        images_t = images.cpu().detach().numpy()
        for ii in range(len(images_t)):
            plt.figure()
            fig, ax = plt.subplots()
            a = ax.imshow(np.squeeze(images_t[ii]))
            fig.colorbar(a)
            plt.savefig("./jack_in/plot_jack" + str(ii)+ "_" + str(res_label[ii,0])+".pdf")
        for ii in range(500):
            outputs = model(images)
            optimizer.zero_grad()
            loss = -1*outputs[:,0,0].sum()
            loss.backward()
            if ii % 100 == 0:
                print(loss)
            optimizer.step()
        images = images.cpu().detach().numpy()
        for ii in range(len(images)):
            plt.figure()
            fig, ax = plt.subplots()
            a = ax.imshow(np.squeeze(images[ii]))
            fig.colorbar(a)
            plt.savefig("./jack_out/plot_jack" + str(ii)+ "_" + str(res_label[ii,0])+".pdf")
        break
    ##N_minibatch_acc /= N
    print('Testing acc: %.3f' % (N_minibatch_acc))
    return outputs_all, label_all, frame_name_all, N_minibatch_acc

def test(model, test_loader, computing_device):
    N = 0   
    N_minibatch_acc = 0.0  
    outputs_all = [] 
    label_all = [] 
    frame_name_all = [] 
    for minibatch_count, (images, labels, region_tag, frame_name)  in enumerate(test_loader, 0):
        model.eval()
        with torch.no_grad(): 
            images, labels, region_tag = images.to(computing_device), labels.to(computing_device), region_tag.to(computing_device)
            labels = labels.float() 
            region_tag = region_tag.float()
            outputs = model(images, region_tag)
            N_minibatch_acc += accuracy(outputs,labels)
            N = N + 1
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs_all.append(outputs)
            label_all.append(labels)
            #print(frame_name)
            frame_name_all.append(frame_name.cpu().detach().numpy())
    N_minibatch_acc /= N
    print('Testing acc: %.3f' % (N_minibatch_acc))
    return outputs_all, label_all, frame_name_all, N_minibatch_acc

def validation(model, vali_loader,optimizer ,criterion, epoch ,weights, computing_device, saved_model_dir):
    N = 0
    N_minibatch_loss = 0.0   
    N_minibatch_acc = 0.0
    first_correct = 0
    first_wrong = 0 
    second_correct = 0 
    second_wrong = 0
    third_correct = 0 
    third_wrong = 0 
    forth_correct = 0
    forth_wrong = 0
    labels_all = []
    outputs_all = []
    frame_name_all = []
    for minibatch_count, (images, labels, region_tag, frame_name) in enumerate(vali_loader, 0):
        model.eval()
        optimizer.zero_grad()
        with torch.no_grad():  
            images, labels, region_tag = images.to(computing_device), labels.to(computing_device), region_tag.to(computing_device)
            labels = labels.float() 
            region_tag = region_tag.float()
            outputs = model(images, region_tag)
            loss = criterion(outputs,labels.squeeze())
            outputs_all.append(outputs.cpu().detach().numpy())
            labels_all.append(labels.cpu().detach().numpy())
            frame_name_all.append(frame_name.numpy())
            loss[:,:,-1] = 0
            #loss[:,:,0] = loss[:,:,0]*2
            loss = loss.mean()
            N_minibatch_loss += loss.cpu().detach().numpy()
            N_minibatch_acc += accuracy(outputs,labels)
            first_correct_t, first_wrong_t, second_correct_t, second_wrong_t, third_correct_t , third_wrong_t, forth_correct_t, forth_wrong_t   = check_first_second(outputs, labels)
            first_correct += first_correct_t
            second_wrong += second_wrong_t
            first_wrong += first_wrong_t 
            second_correct += second_correct_t
            third_correct += third_correct_t
            third_wrong += third_wrong_t
            forth_wrong += forth_wrong_t 
            forth_correct += forth_correct_t
        N = N + 1
    N_minibatch_loss /= N
    N_minibatch_acc /= N
    print('Validation, average %d loss: %.3f acc: %.3f' %
        (minibatch_count, N_minibatch_loss, N_minibatch_acc))
    print("first occurance ", first_correct+first_wrong, " correct: ", first_correct, " wrong: ", first_wrong)
    print("second occurance ", second_correct+second_wrong, " correct: ", second_correct, " wrong: ", second_wrong)
    print("third occurance ", third_correct+third_wrong, " correct: ", third_correct, " wrong: ", third_wrong)
    print("forth occurance ", forth_correct+forth_wrong, " correct: ", forth_correct, " wrong: ", forth_wrong)
    _ , f1_score = run_stats(outputs_all, labels_all, frame_name_all)
    return N_minibatch_acc, N_minibatch_loss, f1_score

def train(model, train_loader,vali_loader, test_loader, optimizer, criterion, weights ,num_epochs, computing_device , saved_model_dir):
    total_loss = []
    avg_minibatch_loss = []
    val_acc_all = []
    val_loss_all = []
    test_acc_epoch = []
    f1_score_all = []
    for epoch in range(num_epochs):
        N = 100
        N_minibatch_loss = 0.0   
        N_minibatch_acc = 0.0    
        for minibatch_count, (images, labels, region_tag ,_ ) in enumerate(train_loader, 0):
            model.train()
            images, labels, region_tag = images.to(computing_device).float(), labels.to(computing_device).float(), region_tag.to(computing_device).float()
            optimizer.zero_grad()
            outputs = model(images, region_tag)
            loss = criterion(outputs,labels.squeeze())
            #loss[:,:, 0] = loss[:,:,0]
            loss[:,:,-1] = 0
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            loss = loss.cpu().detach().numpy()
            total_loss.append(loss)
            N_minibatch_loss += loss
            acc = accuracy(outputs,labels)
            #print(acc)
            N_minibatch_acc += acc
            # im very unhappy with this part
            if minibatch_count % N == 0 and minibatch_count != 0: 
                N_minibatch_loss /= N
                N_minibatch_acc /= N
                print('Epoch %d, average minibatch %d loss: %.3f acc: %.3f' %
                    (epoch, minibatch_count, N_minibatch_loss, N_minibatch_acc))
                avg_minibatch_loss.append(N_minibatch_loss)
                N_minibatch_loss = 0.0
                N_minibatch_acc = 0.0
        print("Finished", epoch, "epochs of training")
        val_acc, val_loss, f1_score = validation(model,vali_loader,optimizer,criterion,epoch,weights,computing_device, saved_model_dir)
        val_acc_all.append(val_acc)
        val_loss_all.append(val_loss)
        f1_score_all.append(f1_score)
        outputs_all, label_all, frame_name_all, N_minibatch_acc = test(model, test_loader, computing_device)
        overall_acc, _ = run_stats(outputs_all, label_all, frame_name_all)
        test_acc_epoch.append(overall_acc)
        if np.argmax(np.array(f1_score_all)) == epoch:
        #if True:
            save_checkpoint({'epoch': epoch,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict(),},
                    filename= saved_model_dir+'/best.pth')
    best_epoch = np.argmax(np.array(f1_score_all))
    print("Training complete after", epoch, "epochs")
    print("Best epoch is in epoch" , best_epoch )

    return total_loss, val_acc_all, val_acc_all, best_epoch, test_acc_epoch

def weighted_mse_loss(inp, target, weights):
    out = abs(inp - target)
    out = out * weights.expand_as(out)
    loss = out.mean() 
    return loss

def run_stats(outputs,labels,frame_name_all,direct = None ,num_characters=num_characters):
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
    overall_c = 0
    overall_label = 0 
    person_prediction_stats = {}
    f_1 = 0.0
    for person_index in sorted(person_dict.keys()):
        one_person_frame_stats = person_dict[person_index]
        one_person_stats = np.zeros(8)
        for frame_sorted in sorted(one_person_frame_stats.keys()):
            one_person_stats += one_person_frame_stats[frame_sorted]
        tp, fp, fn = one_person_stats[0], one_person_stats[3], one_person_stats[1]
        f1_character = cal_f1_score( tp, fp, fn )
        f_1 += f1_character
        print(person_index, one_person_stats.astype(int), f1_character)
        yes_c, yes_all = one_person_stats[0], one_person_stats[0]+ one_person_stats[1]
        no_c, no_all = one_person_stats[2], one_person_stats[2]+ one_person_stats[3]
        dnk_c, dnk_all = one_person_stats[4], one_person_stats[4]+ one_person_stats[5]
        person_prediction_stats[person_index] = [yes_c+ no_c + dnk_c, yes_all+no_all+dnk_all]
        overall_c += yes_c+ no_c + dnk_c 
        overall_label += yes_all+no_all+dnk_all
    overall_acc = overall_c * 1.0 / overall_label
    return overall_acc, f_1*1.0/num_characters


