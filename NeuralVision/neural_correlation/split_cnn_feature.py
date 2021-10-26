################################################################################
# split_cnn_feature
# very messy.....
################################################################################

import sys
sys.path.append(".")
from neural_correlation.utilities import *
# from dipy.io import *
import os
import numpy as np
from project_setting import episode_number
if episode_number == 1: 
    from data import project_dir, path_to_matlab_generated_movie_data, path_to_training_data, \
        sr_final_neural_data, sr_original_movie_data, feature_time_width, patientNums, data_mode, zero_signal
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

    root_dir = os.path.join(project_dir, 'Character_TimeStamp_resnet')

    bill = set(load_pickle(os.path.join(root_dir , "1.pkl")))
    jack = set(load_pickle(os.path.join(root_dir , "0.pkl")))
    chole = set(load_pickle(os.path.join(root_dir , "2.pkl" )))
    terro = set(load_pickle(os.path.join(root_dir , "3.pkl")))

    jack_list = [[15684,15760],[33352,33368],[38912,41312],[41576,41640],[21528,21592],[39100,39244],[39340,39352],
    [39432,39464],
    [39676,40052],
    [40216,40308],
    [41120,41244],
    [41292,41296],
    [41424,41600],
    [42300,42316],
    [42300, 42316],
    [42296, 42372],
    [62360, 62424],
    [64484, 65412],
    [38740, 38852],
    [51468, 51832],
    [64532, 65536],
    [66552, 66632],
    [68560, 68648]]

    bill_list = [[56932,57584],[23424,23500],[56952,57048],[42296,42372],[38600, 38736],[38856, 38908],
    [66300, 66404],[38600, 38736],[15684,15760],[33352,33368],[38912,41312],[41576,41640],[21528,21592],[39100,39244],[39340,39352],
    [39432,39464],[39676,40052],[40216,40308],[41120,41244],[41292,41296],[41424,41600],[42300,42316]]

    chole_list =[[56932,57584],
    [23424,23500],
    [56952,57048],
    [42296,42372],
    [55292, 55404],
    [55776, 55860],
    [55352, 55400],
    [55776, 55860],
    [57076, 57376],
    [57480, 57640],
    [42296, 42372]]

    terro_list = [[38600, 38736],
    [38856, 38908],
    [66300, 66404],
    [38600, 38736],
    [55292, 55404],
    [55776, 55860],
    [55352, 55400],
    [55776, 55860],
    [57076, 57376],
    [57480, 57640],
    [42296, 42372],
    [62360, 62424],
    [64484, 65412],
    [38740, 38852],
    [51468, 51832],
    [64532, 65536],
    [66552, 66632],
    [68560, 68648]]


    jack_interval = convert_to_set(jack_list)
    bill_interval = convert_to_set(bill_list)
    chole_interval = convert_to_set(chole_list)
    terro_interval = convert_to_set(terro_list)

    # load the neural data and create the features

    if data_mode == "neuron":
        feature_vectors = np.load(os.path.join(path_to_matlab_generated_movie_data, patientNum, 'features_mat_clean.npy'))
    if data_mode == "channel":
        feature_vectors = np.load(os.path.join(path_to_matlab_generated_movie_data, patientNum, 'features_channel.npy'))
    print(feature_vectors.shape)

    frame_len = 18576 #18500 #18576

    label = np.zeros((4,3))
    feature_all = []
    label_all = []
    frame_numbers = []
    '''
    root_dir = os.path.join(project_dir, 'time_stamp_2')
    bill = set(load_pickle(os.path.join(root_dir , "1.pkl")))
    jack = set(load_pickle(os.path.join(root_dir , "0.pkl")))
    chole = set(load_pickle(os.path.join(root_dir , "2.pkl" )))
    terro = set(load_pickle(os.path.join(root_dir , "3.pkl")))
    jack_interval = set()
    bill_interval = set()
    chole_interval = set()
    terro_interval = set()
    '''
    for i in range(23,frame_len-24):
        frame_number = i * 4
        #print(frame_number)
        second = frame_number*1.0/sr_original_movie_data
        window_left = second-feature_time_width/2
        window_right = second+feature_time_width/2
        index_left = int(window_left*sr_final_neural_data)
        index_right = index_left + sr_final_neural_data*feature_time_width

        label = label * 0
        one_feature = feature_vectors[:,index_left:index_right]
        feature_all.append(one_feature)

        # jack
        #print(frame_number)
        if frame_number in jack:
            label[0,0] = 1
        elif frame_number in jack_interval:
            label[0,2] = 1
        else:
            label[0,1] = 1
        # bill
        if frame_number in bill:
            label[1,0] = 1
        elif frame_number in bill_interval:
            label[1,2] = 1
        else:
            label[1,1] = 1
        #cholo
        if frame_number in chole:
            label[2,0] = 1
        elif frame_number in chole_interval:
            label[2,2] = 1
        else:
            label[2,1] = 1

        #terros
        if frame_number in terro:
            label[3,0] = 1
        elif frame_number in terro_interval:
            #print("here")
            label[3,2] = 1
        else:
            label[3,1] = 1
        label_all.append(label)
        frame_numbers.append(frame_number)
    
    if zero_signal:
        for i in range(300):
            label = label*0
            label[:,1] = 1
            one_feature = one_feature*0.0
            frame_number = -1 * i
            label_all.append(label)
            frame_numbers.append(frame_number)
            feature_all.append(one_feature)
    print(len(feature_all))
    label_all = np.array(label_all)
    feature_all = np.array(feature_all)
    print(feature_all.shape)
    frame_numbers = np.array(frame_numbers)

    print(patientNum)
    #print(label_all.shape)
    #print(feature_all.shape)
    #print(frame_number.shape)

    np.save(os.path.join(path_to_training_data, patientNum,'label.npy'),label_all)
    np.save(os.path.join(path_to_training_data, patientNum, 'feature.npy'),feature_all)
    np.save(os.path.join(path_to_training_data, patientNum, 'frame_number.npy'),frame_numbers)

def main(patientNums = patientNums):
    for this_patient in patientNums:
        path_to_patient = os.path.join(path_to_training_data, this_patient)
        print(path_to_patient)
        if not os.path.exists(path_to_patient):
            os.mkdir(path_to_patient)
        create_save_features_labels(this_patient)

if __name__ == '__main__':
    main()
