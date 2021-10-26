from utilities import *
from scipy.io import loadmat
import numpy as np
original = "/media/yipeng/toshiba/movie/Movie_Analysis/data/"
training_data_location = "/media/yipeng/toshiba/movie/Movie_Analysis/data/"
converted_data_location = training_data_location
patients = os.listdir(original)
for one_patient in patients:
    if one_patient ==  "437":
        continue
    patient_folder = os.path.join(original, one_patient)
    print(one_patient)
    patient_channel_num = []
    patient_region_list = []
    if not os.path.isdir(patient_folder) or not one_patient.isdigit():
        continue
    mat_file = loadmat(join(patient_folder, "units.mat"))['units']
    for i in range(len(mat_file)):
        patient_channel_num.append(mat_file[i,0][2].ravel()[0])
        patient_region_list.append(mat_file[i,0][3][0])
    patient_channel_list = list(set(patient_channel_num))
    matrix = np.load( join(patient_folder,"features_mat.npy") )
    ln = len(patient_channel_list)
    res_mat = np.zeros((ln,matrix.shape[1]))
    patient_channel_list_sorted = sorted(patient_channel_list)
    channe_regin_list = []
    for i in range(len(patient_channel_list_sorted)):
        channel = patient_channel_list_sorted[i]
        idx = np.where(patient_channel_num == channel)[0]
        channe_regin_list.append(patient_region_list[idx[0]])
        res_mat[i,:] = np.sum(matrix[idx,:], axis=0)
    np.save(join(patient_folder,"features_channel.npy"), res_mat)
    np.save(join(patient_folder,"channels.npy"), channe_regin_list)
