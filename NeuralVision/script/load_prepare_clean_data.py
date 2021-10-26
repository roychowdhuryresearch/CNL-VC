# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:03:11 2019

@author: zahra
Modified 2/11 based on the cleaned data 
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.interpolate import interp1d
import sys
sys.path.append(".")
from data import path_to_matlab_generated_movie_data, sr_final_neural_data, patientNums, movie_duration


#%% helper function to turn the matlab structs into numpy arrays
def build_units_firing(matlab_units_struct, temp_tBins = None):
    temp_firing = np.asarray(matlab_units_struct['Firing']).squeeze()
    channel_nums = np.asarray(matlab_units_struct['channelNum'], dtype = int).squeeze()
    channel_regions = np.asarray(matlab_units_struct['brainRegion']).squeeze()
    sorted_ind = np.argsort(channel_nums)

    if temp_tBins is None:
        temp_tBins = np.asarray(np.ravel(temp_firing[0]['Centers'])[0], dtype = float)

    units_firing = np.empty((len(temp_firing), len(temp_tBins)))
    for i in range(len(matlab_units_struct)):
        units_firing[i,:] = np.asarray(np.ravel(temp_firing[i]['Rate'])[0], dtype = float).squeeze()

    return units_firing[sorted_ind,:], channel_regions[sorted_ind], channel_nums[sorted_ind]
    
def build_lfp(matlab_lfp_struct, temp_lfp_ts = None):
    channel_nums = np.squeeze(np.asarray(matlab_lfp_struct['channelNum'], dtype = int))
    sorted_ind = np.argsort(channel_nums)
    
    temp_lfp = np.asarray(matlab_lfp_struct['data']).squeeze()
    
    if temp_lfp_ts is None:
        temp_lfp_ts = np.asarray(temp_lfp[0], dtype = float)
        
    all_lfp = np.empty((len(temp_lfp), len(temp_lfp_ts)))
    for i in range(len(matlab_lfp_struct)):
        all_lfp[i,:] = np.asarray(np.ravel(temp_lfp[i]), dtype = float).squeeze()
        
    return all_lfp[sorted_ind,:], channel_nums[sorted_ind]

def interpolate_neural_data(data, original_timestamps, neural_sampling = sr_final_neural_data):
    new_timestamps = np.arange(original_timestamps[0],stop = original_timestamps[-1], step = 1/neural_sampling )
    f = interp1d(original_timestamps, data, axis = 1)
    new_data = f(new_timestamps)
    return new_data, new_timestamps

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]

def save_final_clean_neural_firing(patientID, path_to_matlab_generated_movie_data = path_to_matlab_generated_movie_data):
    path_to_data = os.path.join(path_to_matlab_generated_movie_data, patientID)
    #datamat = sio.loadmat(os.path.join(path_to_data, 'clean_data2.mat'))['patient']
    datamat = sio.loadmat(os.path.join(path_to_data, 'clean_data.mat'))['patient']
    units_ts = datamat['tbins'][0,0][0]
    #speficied_time_stamp = []
    units_org_firing = datamat['firing'][0,0]
    units_regions = datamat['region'][0,0][0]
    units_org_firing = datamat['firing'][0,0]

    print("units_org_ts", units_ts.shape, units_ts)

    #%% we need to comment one or the other to make it feature clean or feature clean 2include data within selected range 
    time_range = [0, movie_duration*60]  # in movie
    #time_range = [movie_duration*60, 4699.8 + 3]  # testing parts
    ##%%for testing clips after movie, find the max value of last (spiking time recorded across patients) + 3s as end of test;  
    in_movie = np.logical_and(units_ts >= time_range[0], units_ts <= time_range[1])
    units_ts = units_ts[in_movie]
    units_firing = units_org_firing[:, in_movie]
    
    #print("processed unit ts:",units_ts.shape, units_ts)
    #print("processed regions:",units_regions.shape, units_regions)
 
    #%% interpolate the neural data to the final sampling rate
    units_firing, units_ts = interpolate_neural_data(units_firing, units_ts)


    print('the duration of the firing data for patient ' + patientID + 'is: ' + str(units_firing.shape[1]))
    np.save(os.path.join(path_to_data, 'features_mat_clean.npy'), units_firing)
    np.save(os.path.join(path_to_data, 'features_mat_ts_clean.npy'), units_ts)
    np.save(os.path.join(path_to_data, 'features_mat_regions_clean.npy'), units_regions)

def main(patientNums = patientNums, path_to_matlab_generated_movie_data = path_to_matlab_generated_movie_data):
    for this_patient in patientNums:
        print("patient id", this_patient)
        save_final_clean_neural_firing(this_patient, path_to_matlab_generated_movie_data)

if __name__ == '__main__':
    main()