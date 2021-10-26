# PyTorch and neural network imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils

import matplotlib.pyplot as plt
import numpy as np
import os

class BasicCNN(nn.Module):
    """ A basic convolutional neural network model for baseline comparison. 
    
    conv1 -> conv2 -> conv3 -> maxpool -> fc1 -> fc2 (outputs)
    
    """
    
    def __init__(self, nunber_of_characters, num_features):
        super(BasicCNN, self).__init__()
        ## parameters:
        self.nunber_of_characters = nunber_of_characters
        self.num_features = num_features
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1)
        self.conv0_normed = nn.BatchNorm2d(64)
        #self.conv1 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=(2, 0))
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=(4, 0))
        self.conv1_normed = nn.BatchNorm2d(128)
        torch_init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)
        self.conv2_normed = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5)
        self.conv3_normed = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, ceil_mode=True)
        self.fc1 = nn.Linear(in_features=num_features,out_features=512)
        self.fc1_normed = nn.BatchNorm1d(512)
        self.fc1_r = nn.Linear(in_features=512,out_features=512)
        self.fc1_r_normed = nn.BatchNorm1d(512)
        self.fc15 = nn.Linear(in_features=512,out_features=256)
        self.fc15_normed = nn.BatchNorm1d(256)
        self.fc16 = nn.Linear(in_features=256,out_features=128)
        self.fc16_normed = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=nunber_of_characters*3)
        self.fc2_normed = nn.BatchNorm1d(nunber_of_characters*3)
        self.softmax = nn.LogSoftmax(dim = 2)
        self.relu = nn.LeakyReLU()
    def forward(self, batch, region_tag):
        batch = batch.float()
        region_tag = region_tag.float()
        #print("in",batch.shape)
        batch = self.relu(self.conv0_normed(self.conv0(batch)))
        #print("conv0", batch.shape)
        batch = self.relu(self.conv1_normed(self.conv1(batch)))
        #print("conv1",batch.shape)
        batch = self.relu(self.conv2_normed(self.conv2(batch))) 
        #print("conv2",batch.shape)
        batch = self.relu(self.conv3_normed(self.conv3(batch))) 
        #print("conv3",batch.shape)
        batch = self.pool(batch)     
        #print("pool",batch.shape)
        #batch = torch.cat((batch.view(-1, self.num_flat_features(batch)), region_tag), 1)
        batch = batch.view(-1, self.num_flat_features(batch))
        #print(batch.shape)
        #batch = self.fc1_dropout(self.relu(self.fc1_normed(self.fc1(batch))))  
        batch = self.relu(self.fc1_normed(self.fc1(batch))) 
        batch = self.relu(self.fc1_r_normed(self.fc1_r(batch))) 
        batch = self.relu(self.fc15_normed(self.fc15(batch)))
        batch = self.relu(self.fc16_normed(self.fc16(batch)))  
        batch = self.fc2(batch)
        batch = batch.reshape(-1, self.nunber_of_characters, 3)
        #batch = func.softmax(batch, dim=2)
        batch = self.softmax(batch)
        return batch
    
    
    def num_flat_features(self, inputs):
        # Get the dimensions of the layers excluding the inputs
        size = inputs.size()[1:]
        # Track the number of features
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
