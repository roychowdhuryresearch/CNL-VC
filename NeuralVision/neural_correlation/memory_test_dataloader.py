################################################################################
# Dataloader
# load the neural image to CNN
################################################################################
import sys
sys.path.append(".")
# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler
import torch.nn.functional as func
# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random

from project_setting import episode_number
if episode_number == 1:
    from data import project_dir, path_to_matlab_generated_movie_data, path_to_training_data, \
    sr_final_neural_data, sr_original_movie_data, feature_time_width, patientNums,\
         patient_features, path_to_model, model_option
elif episode_number == 2:
    from data2 import project_dir, path_to_matlab_generated_movie_data, path_to_training_data, \
    sr_final_neural_data, sr_original_movie_data, feature_time_width, patientNums,\
         patient_features, path_to_model


class NeuralImageDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, tag_dir, frame_name_dir ,mask=None, erasing = True, transform = None):
        """
        Args:
        -----
        - data_dir (str) : data file
        - label_dir (str) : label file
        - frame_name_dir(str) : frame_names
        Attributes:
        -----------
        - data: data list (np array)
        - label: label list  (np array)
        - frame_name: frame_name list (np array)
        """
        
        self.data = np.load(data_dir)
        self.label= np.load(label_dir)
        self.frame_name = np.load(frame_name_dir)
        tag_mat = np.load(tag_dir, allow_pickle=True)
        classnames, indices = np.unique(tag_mat , return_inverse=True)
        indices.sort()
        self.indices = indices
        self.reg_tag = self.one_hot_encoding(indices)
        self.transform = transform
        print(data_dir)
        if mask is None:
            print("mask is none")
            self.mask = None
        else:
            if erasing:
                self.mask = np.ones((self.data.shape[1],sr_final_neural_data*feature_time_width))
                #print("mask is ", mask)
                self.mask[mask] = 0 
            else:
                self.mask = np.zeros((self.data.shape[1],sr_final_neural_data*feature_time_width))
                self.mask[mask] = 1
        '''
        print(np.shape(self.data))
        print(np.shape(self.label))
        print(np.shape(self.frame_name))
        '''
    def __len__(self):
        
        # Return the total number of data samples
        return self.label.shape[0]

    def __data_shape__(self):

        return (self.data.shape[0], self.data.shape[1])
    def one_hot_encoding(self, data):
        ecd = np.zeros((data.size, data.max()+1))
        ecd[np.arange(data.size),data] = 1
        return ecd

    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label, frame_name)
        """
        if self.mask is not None:
            im = np.multiply(self.data[ind], self.mask)
            image = np.array([im])
        else:
            image = np.array([self.data[ind]])

        if model_option is "LSTM":
            image = image.squeeze().T

        image = torch.from_numpy(image)

        label = torch.from_numpy(np.array([self.label[ind]]))
        frame_name = self.frame_name[ind]
        
        if self.transform is not None:
            im_tag = {'image': image, 'region_tag': self.reg_tag, 'indices':self.indices}
            transformed = self.transform(im_tag)
            image = transformed['image']
            region_tag = transformed['region_tag'].flatten()
        else:
            region_tag = self.reg_tag.flatten()
        # Return the image and its label
        #print("region_tag", region_tag.shape)
        #rint("image_shape", image.shape)
        return (image, label, region_tag ,frame_name)


def create_split_loaders(data_dir,label_dir, tag_fn ,frame_name_dir ,batch_size, seed, kfold = -1, transform = None,p_val=0.1, p_test=0.2, shuffle=True, 
                         mask= None, erasing = True ,show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
    - data_dir (str) : data file
    - label_dir (str) : label file
    - frame_name_dir(str) : frame_names
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict) 
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory 
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Get create a ChestXrayDataset object
    dataset = NeuralImageDataset(data_dir, label_dir, tag_fn ,frame_name_dir, mask, erasing, transform = transform)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        random.seed(seed)
    fixed_val_percent = int(np.floor(0.1 * dataset_size))
    test_split = int(np.floor(p_test * dataset_size))
    val_split = int(np.floor(p_val * dataset_size))
    if kfold == -1:   
        train_ind, val_ind = all_indices[: -(val_split + test_split)], all_indices[-(val_split + test_split):]
        random.shuffle(val_ind)
        val_ind, test_ind = val_ind[:val_split], val_ind[val_split:]
        '''
        train_ind, val_ind = all_indices[: int(2.5*fixed_val_percent)], all_indices[int(2.5*fixed_val_percent): 4*fixed_val_percent]
        random.shuffle(train_ind)
        random.shuffle(val_ind)
        val_ind, test_ind = val_ind[:int(0.1 * int(len(val_ind)))], val_ind[int(0.1 * int(len(val_ind))):]
        
        train_ind, val_ind = all_indices[: int(4*fixed_val_percent)], all_indices[int(4*fixed_val_percent):int(7*fixed_val_percent) ]
        random.shuffle(train_ind)
        random.shuffle(val_ind)
        val_ind, test_ind = val_ind[:int(0.1 * int(len(val_ind)))], val_ind[int(0.1 * int(len(val_ind))):]
        '''
    else:
        test_start = kfold*test_split
        test_end = test_start + test_split
        if test_end > dataset_size:
            raise("wrong k")
        test_ind = all_indices[test_start: test_end]
        remain_ind = list(set(all_indices) - set(test_ind))
        np.random.shuffle(remain_ind)
        #val_ind, train_ind = remain_ind[:fixed_val_percent] , remain_ind[fixed_val_percent: fixed_val_percent+ val_split]
        val_ind, train_ind = remain_ind[:fixed_val_percent] , remain_ind[val_split:]
        
    print("train", len(train_ind))
    print("vali", len(val_ind))
    print("test", len(test_ind))

    #np.random.shuffle(train_ind)
    #np.random.shuffle(val_ind)
    #np.random.shuffle(test_ind)

    '''
    np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    # Separate a test split from the training dataset
    train_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    '''
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SequentialSampler(all_indices)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    
    # Return the training, validation, test DataLoader objects
    return train_loader
