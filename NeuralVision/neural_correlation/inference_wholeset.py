import sys
sys.path.append(".")
# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as func
# Other libraries for data manipulation and visualization
import os
from PIL import Image
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import random
from data import path_to_matlab_generated_movie_data, patientNums, project_dir, path_to_training_data
from lstm import MovieLSTM
from utilities import dump_pickle, clean_folder
np.random.seed(0)
random.seed(0)

class NeuralImageDataset(Dataset):
    
    def __init__(self, data_dir, label_dir, tag_dir, frame_name_dir ,mask=None, erasing = True, transform = None):

        self.data = np.load(data_dir)
        self.label= np.load(label_dir)
        self.frame_name = np.load(frame_name_dir)
        tag_mat = np.load(tag_dir, allow_pickle=True)
        classnames, indices = np.unique(tag_mat , return_inverse=True)
        indices.sort()
        self.indices = indices
        self.reg_tag = self.one_hot_encoding(indices)
        self.transform = transform
        #print(self.reg_tag.shape[0] *self.reg_tag.shape[1])

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
        image =  np.array([self.data[ind]])
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
        return (image, label, region_tag ,frame_name)


def create_loaders(data_dir,label_dir, tag_fn ,frame_name_dir ,batch_size, seed=1, kfold = -1, transform = None,p_val=0.1, p_test=0.2, shuffle=True, 
                         mask= None, erasing = True ,show_sample=False, extras={}):
    dataset = NeuralImageDataset(data_dir, label_dir, tag_fn ,frame_name_dir, mask, erasing, transform = transform)
    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader

def inference(model,iterator):
    outputs = []
    N = 0    
    outputs_all = [] 
    label_all = [] 
    frame_name_all = [] 
    for minibatch_count, (images, labels, region_tag, frame_name)  in enumerate(iterator, 0):
        with torch.no_grad(): 
            images, labels, region_tag = images.cuda(), labels.cuda(), region_tag.cuda()
            labels = labels.float() 
            region_tag = region_tag.float()
            outputs = model(images, region_tag)
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs_all.append(outputs)
            label_all.append(labels)
            frame_name_all.append(frame_name.cpu().detach().numpy())
    outputs_all = np.concatenate(outputs_all)
    label_all = np.concatenate(label_all)
    frame_name_all = np.concatenate(frame_name_all)
    prediction = np.argmax(outputs_all, axis=-1)
    print(prediction.shape)
    return outputs_all, prediction, frame_name_all

def run_inference(path_to_training_data, this_patient, path_to_matlab_generated_movie_data, model_dir, fold_num, output_dir):
    data_fn = os.path.join(path_to_training_data, this_patient, "feature.npy")
    label_fn = os.path.join(path_to_training_data, this_patient, "label.npy")
    frame_name_dir = os.path.join(path_to_training_data, this_patient, "frame_number.npy")
    tag_fn = os.path.join(path_to_matlab_generated_movie_data, this_patient, "features_mat_regions_clean.npy")
    data_loader = create_loaders(data_fn,label_fn, tag_fn ,frame_name_dir, 128)
    model = MovieLSTM(data_fn)
    model = model.cuda()
    model_dirct =  os.path.join(model_dir, str(fold_num) ,"best.pth")
    check_point = torch.load(model_dirct)
    model.load_state_dict(check_point['state_dict'])
    model.eval()
    _, predictions, frames = inference(model,data_loader)

    output_folder = os.path.join(output_dir, str(fold_num))
    clean_folder(output_folder)
    for i in range(4):
        idx = np.where(predictions[:, i] == 0)[0]
        candidate_frames = frames[idx.astype(np.int)] 
        output_fn = os.path.join(output_folder, str(i)+".pkl")
        dump_pickle(output_fn, candidate_frames)

def multiple(patientNums = patientNums, project_dir = project_dir ):
    #path_to_training_data = "/media/yipeng/toshiba/movie/Movie_Analysis/memory_test_2"
    path_to_model = os.path.join(project_dir,'CNN_result', 'LSTM_multi_2_KLD')
    path_to_model = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
    output_dir = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_fromNeural"
    for this_patient in patientNums:
        patient_model = os.path.join(path_to_model, this_patient)
        output_folder_p = os.path.join(output_dir, this_patient)
        clean_folder(output_folder_p)
        for fold_num in range(5):
            run_inference(path_to_training_data, this_patient, path_to_matlab_generated_movie_data, patient_model, str(fold_num), output_folder_p)

if __name__ == '__main__':
    multiple()