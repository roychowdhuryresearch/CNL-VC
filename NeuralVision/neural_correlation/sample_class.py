################################################################################
# sample_classes
# generate the most import neuron for each frame
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random 

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim
from cnn import BasicCNN
from utilities import *
seed = 10
np.random.seed(seed)
random.seed(seed)

data_fn = "/home/yipeng/Desktop/Movie_Analysis/training_data/feature.npy"
label_fn = "/home/yipeng/Desktop/Movie_Analysis/training_data/label.npy"
frame_name_dir = "/home/yipeng/Desktop/Movie_Analysis/training_data/frame_number.npy"
character_dict = {"bill":1, "jack":0, "chloe":2, "terro":3}
check_character = "bill"
character_num = character_dict[check_character]
data = np.load(data_fn)
label= np.load(label_fn)
frame_name = np.load(frame_name_dir)

model = BasicCNN(4)
direct = "/home/yipeng/Desktop/Movie_Analysis/neural_correlation/CNN/"
model_number = "17"
model_dirct = direct + 'model_epoch' + str(model_number) + '.pth'
print('Resume model: %s' % model_dirct)
model_test = model
check_point = torch.load(model_dirct)
model_test.load_state_dict(check_point['state_dict'])
all_frame_dir = "/home/yipeng/Desktop/video_process/frames/"

def occlusion(model, image, label, occ_size = 5, occ_stride = 5, occ_pixel = 0.5):
  
    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]
  
    #setting the output image width and height
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))
  
    #create a white image of sizes we defined
    #heatmap = np.zeros((output_height, output_width))
    heaplist = []
    #iterate all the pixels in each column
    for w in range(0, width):
        #h_start = h*occ_stride
        #h_end = min(height, h_start + occ_size)
        input_image = image.clone().detach()
        #replacing all the pixel information in the image with occ_pixel(0) in the specified location
        input_image[:, : ,w, :] = 0
        input_image = input_image
        #run inference on modified image
        output = model(input_image)
        output = np.squeeze(output.detach())
        #output = nn.functional.softmax(output, dim=2)
        prob = output[label,0]
        #setting the heatmap location to probability value
        #heatmap[, w] = prob 
        heaplist.append(prob)
    return np.array(heaplist)

for check_character in character_dict.keys():
#for check_character in ["terro"]:
    print(check_character)
    character_num = character_dict[check_character]
    #all_idx = list(range(int(0.7*len(data))))
    all_idx = list(range(int(len(data))))
    random.shuffle(all_idx)
    jack_yes_im = [] 
    jack_yes_label = []
    jack_yes_frame = [] 
    for idx in range(int(len(data))):
        if label[all_idx[idx]][character_num,0] == 1:
            jack_yes_im.append(data[all_idx[idx]])
            jack_yes_label.append(label[all_idx[idx]])
            jack_yes_frame.append(frame_name[all_idx[idx]])
    #print(jack_yes_label[1:10])
    heat_list_all = np.zeros(138)
    plot_fig = True
    save_dir = "/home/yipeng/Desktop/Movie_Analysis/neural_correlation/"+check_character + "/"
    clean_folder(save_dir)
    index_max = min(1000,len(jack_yes_label))
    for i in range(index_max):
        heat_list = occlusion(model_test.eval(), torch.from_numpy(np.array([[jack_yes_im[i]]])), character_num)
        heat_list_all += heat_list
        if plot_fig: 
            plt.close("all")
            plt.figure()
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.subplots_adjust(hspace=0.5)
            ax1.imshow(cv2.imread(all_frame_dir+"frame_"+ str(int(jack_yes_frame[i]))+".jpg"))
            ax2.plot(heat_list)
            plt.savefig(save_dir+"plot_"+ str(jack_yes_frame[i])+".jpg")
    heat_list_all = heat_list_all*1.0/index_max
    print(heat_list_all)
    plt.close("all")
    plt.figure()
    plt.plot(heat_list_all)
    plt.savefig(save_dir+"/Avg_plot"+".jpg")
