import sys
sys.path.append(".")
from neural_correlation.utilities import *
# from dipy.io import *
import os
import numpy as np
from project_setting import episode_number
if episode_number == 1: 
    from data import project_dir, path_to_matlab_generated_movie_data, path_to_training_data, \
        sr_final_neural_data, sr_original_movie_data, feature_time_width, patientNums, data_mode
elif episode_number == 2:    
    from data2 import project_dir, path_to_matlab_generated_movie_data, path_to_training_data, \
        sr_final_neural_data, sr_original_movie_data, feature_time_width, patientNums, data_mode

def convert_to_set(mylist, delation=0):
    res = set()
    for i in mylist:
        i[0] = i[0] + delation
        i[1] = i[1] + delation
        rang_list = list(range(i[0], i[1]))
        res = res.union(set(rang_list))
    return res
        
def create_save_features_labels(patientNum, project_dir = project_dir, path_to_training_data = path_to_training_data,
                                path_to_matlab_generated_movie_data = path_to_matlab_generated_movie_data,
                                feature_time_width = feature_time_width, sr_final_neural_data = sr_final_neural_data,
                                sr_original_movie_data = sr_original_movie_data):

    # load the neural data and create the features
   
    if data_mode == "neuron":
        feature_vectors = np.array(np.load(os.path.join(path_to_matlab_generated_movie_data, patientNum, 'features_mat_clean2.npy')))
    if data_mode == "channel":
        feature_vectors = np.load(os.path.join(path_to_matlab_generated_movie_data, patientNum, 'features_channel.npy'))
    print(feature_vectors.shape)


    feature_all = []
    label_all = []
    frame_numbers = []

    label = np.zeros((4,3))
    label[:,1] = 1 
    for i in range(24, 16371-24):
        frame_number = i * 4
        #print(frame_number)
        second = frame_number*1.0/sr_original_movie_data
        window_left = second-feature_time_width/2
        index_left = int(window_left*sr_final_neural_data)
        index_right = index_left + sr_final_neural_data*feature_time_width
        one_feature = feature_vectors[:,index_left:index_right]
        feature_all.append(one_feature)
        label_all.append(label)
        frame_numbers.append(frame_number)
    
    label_all = np.array(label_all)
    feature_all = np.array(feature_all)
    print(feature_all.shape)
    frame_numbers = np.array(frame_numbers)
    
    np.save(os.path.join(path_to_training_data, patientNum, 'feature.npy'),feature_all)
    np.save(os.path.join(path_to_training_data, patientNum,'label.npy'),label_all)
    np.save(os.path.join(path_to_training_data, patientNum, 'frame_number.npy'),frame_numbers)

def main(patientNums = patientNums):
    path_to_training_data = "/media/yipeng/data/movie_2021/Movie_Analysis/mem_test"
    for this_patient in patientNums:
        path_to_patient = os.path.join(path_to_training_data, this_patient)
        print(path_to_patient)
        if not os.path.exists(path_to_patient):
            os.mkdir(path_to_patient)
        create_save_features_labels(this_patient, path_to_training_data=path_to_training_data)

if __name__ == '__main__':
    main()