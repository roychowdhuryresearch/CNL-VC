################################################################################
# CSE 253: Programming Assignment 3
# Winter 2019
# Code author: Jenny Hamer
#
#
# Description: 
# This code defines a custom PyTorch Dataset object suited for the
# NIH ChestX-ray14 dataset of 14 common thorax diseases. This dataset contains
# 112,120 images (frontal-view X-rays) from 30,805 unique patients. Each image
# may be labeled with a single disease or multiple (multi-label). The nominative
# labels are mapped to an integer between 0-13, which is later converted into 
# an n-hot binary encoded label.
# 
#
# Dataset citation: 
# X. Wang, Y. Peng , L. Lu Hospital-scale Chest X-ray Database and Benchmarks on
# Weakly-Supervised Classification and Localization of Common Thorax Diseases. 
# Department of Radiology and Imaging Sciences, September 2017. 
# https://arxiv.org/pdf/1705.02315.pdf
################################################################################

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as data

from setting import group_set_all, num_character

# Uncomment for Python2
# from __future__ import print_function

class_dict = {0:193, 1:501, 2:725,3:744, 4:954, 5:2209, 6:3169, 7:3567,8:8994}
inv_map = {v: k for k, v in class_dict.items()}
class CharacterDataset(Dataset):
    """Custom Dataset class for the Chest X-Ray Dataset.

    The expected dataset is stored in the "/datasets/ChestXray-NIHCC/" on ieng6
    """
    
    def __init__(self, root_dir ,transform=transforms.ToTensor(), color='L'):
        """
        Args:
        -----
        - transform: A torchvision.transforms object - 
                     transformations to apply to each image
                     (Can be "transforms.Compose([transforms])")
        - color: Specifies image-color format to convert to 
                 (default is L: 8-bit pixels, black and white)

        Attributes:
        -----------
        - image_dir: The absolute filepath to the dataset on ieng6
        - image_info: A Pandas DataFrame of the dataset metadata
        - image_filenames: An array of indices corresponding to the images
        - labels: An array of labels corresponding to the each sample
        - classes: A dictionary mapping each disease name to an int between [0, 13]
        """
        self.root_dir = root_dir
        self.image_fns, self.labels = self.get_images_labels(root_dir)
        self.transform = transform
        self.length = len(self.image_fns)

    def get_images_labels(self, root_dir): 
        classes = os.listdir(root_dir)
        image_fns_yes = []
        image_fns_no = []
        labels = []
        for c in classes:
            classes_fn = os.listdir(os.path.join(root_dir, c))
            in_c = int(c)
            if in_c in group_set_all:
                labels.extend([group_set_all[in_c]] * len(classes_fn))
                for k in classes_fn:
                    image_fns_yes.append(os.path.join(c, k))
            else:
                for k in classes_fn:
                    image_fns_no.append(os.path.join(c, k))
        one_class_length = int(len(image_fns_yes)/num_character)
        np.random.shuffle(image_fns_no)
        no_image_fn = image_fns_no[:2*one_class_length]
        image_fns = image_fns_yes + no_image_fn
        labels = labels + [num_character] * (2 * one_class_length)
        return image_fns, labels

        
    def __len__(self):
        
        # Return the total number of data samples
        return self.length


    def __getitem__(self, ind):
        """Returns the image and its label at the index 'ind' 
        (after applying transformations to the image, if specified).
        
        Params:
        -------
        - ind: (int) The index of the image to get

        Returns:
        --------
        - A tuple (image, label)
        """
        image = Image.open(os.path.join(self.root_dir,self.image_fns[ind]))
        #If a transform is specified, apply it
        if self.transform is not None:
            image = self.transform(image)
            
        # Verify that image is in Tensor format
        #if type(image) is not torch.Tensor:
        #    image = transform.ToTensor(image)

        # Convert multi-class label into binary encoding 

        label = self.labels[ind]
        
        # Return the image and its label
        return (image, label)
    

def create_split_loaders(root_dir, batch_size, seed=0, transform=transforms.ToTensor(),
                         p_val=0.1, p_test=0.2, shuffle=True, 
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets. 

    Params:
    -------
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
    

    # once all single json datasets are created you can concat them into a single one:
    quickdraw_dataset = CharacterDataset(root_dir=root_dir, transform=transform)
    
    # Dimensions and indices of training set
    dataset_size = len(quickdraw_dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)
    
    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]
    
    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    print(len(train_ind), len(val_ind), len(test_ind))
    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
        
    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(quickdraw_dataset, batch_size=batch_size, 
                              sampler=sample_train, num_workers=num_workers, 
                              pin_memory=pin_memory)

    test_loader = DataLoader(quickdraw_dataset, batch_size=batch_size, 
                             sampler=sample_test, num_workers=num_workers, 
                              pin_memory=pin_memory)

    val_loader = DataLoader(quickdraw_dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers, 
                              pin_memory=pin_memory)

    
    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)
