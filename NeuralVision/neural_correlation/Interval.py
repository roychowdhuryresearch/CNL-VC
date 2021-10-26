import numpy as np 
import sys
sys.path.append(".")
from data import *
import numpy as np
import scipy
from scipy import signal
import scipy.misc
import pickle

episode1_patch_size = load_pickle("/media/yipeng/data/movie/Movie_Analysis/draft_result/episode1_character_size.pkl")
episode2_patch_size = load_pickle("/media/yipeng/data/movie/Movie_Analysis/draft_result/episode2_character_size.pkl")


def findLength(arr):
    n = len(arr)
    if n == 0:
        return 0
    max_len = 1
    for i in range(n - 1):
        mn = arr[i]
        mx = arr[i]
        for j in range(i + 1, n):
            mn = min(mn, arr[j])
            mx = max(mx, arr[j])
            if ((mx - mn) == j - i):
                max_len = max(max_len, mx - mn + 1)         
    return max_len


class Interval:
    def __init__(self, time_start, time_end):
        self.time_start = time_start # in term of second
        self.time_end = time_end # interm of second
        self.start = 0 # in term of sample
        self.end = 0
        self.num_samples = 0 
        self.prediction = [] ## overall prediction
        self.cv_label = []  ## only the clip label
        self.annotation_label = [] ## only the annotation label
        self.episode = -1
        self.frame_number = []
        self.patient_response = -1
        self.response_time = -1 
        self.response_sample_end = -1
        self.compute_start_end()
        self.original_neural_signal:np.array() = None
        self.memtest_neural_signal:np.array() = None
    def __str__(self):
        return str(self.start) + " " +str(self.end) + " " + str(self.episode) + " " + str(self.prediction.shape) + " " + str(self.cv_label.shape)

    def __eq__(self,other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
    
    def __lt__(self, other):
        return self.time_start < other.time_start

    def compute_start_end(self):
        self.start, self.end = (np.array([self.time_start, self.time_end])*7.5).astype(int) 
        self.num_samples = self.end - self.start
    def set_response_time(self, time):
        self.response_time = time
        self.response_sample_end = int((self.time_end + self.response_time)*7.5)
    def set_prediction(self, prediction):
        self.prediction = prediction
    def set_cv_label(self, label):
        self.cv_label = label
    def set_episode(self, epidode):
        self.episode = epidode
    def set_patient_response(self, response):
        self.patient_response = response
    def set_orignal_neural_signal(self, data):
        self.original_neural_signal = data
    def set_memtest_neural_signal(self, data):
        self.memtest_neural_signal = data
    
    def get_patient_response_ans(self):
        return int(self.patient_response)
    def get_character_prob(self, character_index, mode = "all"):
        if mode is "all":
            prob = self.prediction[character_index,:]
        elif mode is "clip":
            prob = self.prediction[character_index,:self.num_samples]
        else:
            prob = self.prediction[character_index,self.num_samples:]
        ## ADDED
        # if np.sum(prob >= 0.5) > 2:
        #     return 1
        # else:
        #     return 0
        ## aDDED end
        return np.max(prob)
    def get_patient_response_time(self):
        return self.response_time
    def get_start_end(self):
        return self.start, self.end
    def get_response_time(self):
        return self.response_time
    def get_shifted_time(self):
        return np.array(range(self.response_sample_end - self.start))- self.end + self.start
    def get_cvlabel(self, threshold=0.5, mode="percentage"):
        #return np.max(self.cv_label, axis = 1)
        #cv_labels = np.max(self.cv_label, axis = 1)
        if mode == "percentage":
            cv_labels = np.sum(self.cv_label, axis = 1)/len(self.cv_label[1,:]) > threshold
            return cv_labels
        ## this part is added for a quick check
        else:
            if self.episode == 1:
                check = episode1_patch_size
            else:
                check = episode2_patch_size
            size = []
            for ch_id in range(4):
                ch_size = []
                for frame in range(self.frame_number[0], self.frame_number[-1], 4):
                    check_ch = check[ch_id]
                    if frame not in check_ch:
                        ch_size.append(0) 
                    else:
                        ch_size.append(check_ch[frame])
                size.append(np.array(ch_size).mean())
            #print(size)

            size_check = np.array(size) > 219648/2*threshold #canvas/3
            #size_check = np.array(size) > 109824 #canvas/2
            return size_check
        #return res
        #print(size,cv_labels, self.episode)
        #return cv_labels
        #return size_check
        #return size_check * cv_labels
    def get_episode_number(self):
        return int(self.episode)
    def check_character_label(self, character_index, threshold = 0.5):
        return np.max(self.cv_label[character_index,:]) == 1
        #return np.sum(self.cv_label[character_index,:])/len(self.cv_label[character_index,:]) >= threshold
    def check_character_prob(self, character_index, threshold = 0.5, mode="all"):
        if mode is "all":
            prob = self.prediction[character_index,:]
        elif mode is "clip":
            prob = self.prediction[character_index, :self.num_samples]
        else:
            prob = self.prediction[character_index, self.num_samples:]
        return np.max(prob) >= threshold
    '''
    MTL experiment_scripts
    '''
    def check_character_prob_MTL(self, character_index, threshold = 0.5, threshold_num = 1, mode="all"):
        if mode is "all":
            prob = self.prediction[character_index,:]
        elif mode is "clip":
            prob = self.prediction[character_index, :self.num_samples]
        else:
            prob = self.prediction[character_index, self.num_samples:]
        check = np.sum(prob >= threshold)
        #index = np.where((prob >= threshold) > 0)[0]
        #print(findLength(index))
        #return findLength(index) >= threshold_num
        return check >=threshold_num  


    def get_predictions(self):
        #return self.prediction[:,:self.num_samples]
        #return np.max(self.prediction, axis= 1) >= 0.5
        return np.max(self.prediction[:,:self.num_samples], axis= 1) >= 0.5
    '''
    only movie datapoinst 
    find the data in data_X x:depends on length of the image 
    for example 
    result_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD" in run_memtest_analysis.py 
    we should use "training_data_1"
    '''
    def get_movie_datapoint(self):
        return self.frame_number[0]*1.0/4, self.frame_number[-1]*1.0/4
    '''
    only mentest datapoints 
    find the data in memory_test_X x:depends on length of the image 
    for example 
    result_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD" in run_memtest_analysis.py 
    we should use "memory_test_2"
    ## sorry the it is not same as the training data using 1 will do the modification soon 
    '''
    def get_mentest_datapoint(self):
        return self.start, self.start+len(self.frame_number)
    '''
    only response time
    '''
    def get_response_datapoint(self): 
        return self.start+len(self.frame_number), self.end

    def set_frame_number(self, frame_number):
        self.frame_number = list(frame_number[np.where(frame_number!=0)[0]])

    def neural_correlation(self, other):
        time_lag, r_xy_max, r_xy = self.cal_neural_correlation(other.memtest_neural_signal, self.original_neural_signal)
        return time_lag, r_xy_max, r_xy
    
    def self_correlation(self):
        time_lag, r_xy_max, r_xy  = self.cal_neural_correlation(self.memtest_neural_signal, self.original_neural_signal)
        return time_lag, r_xy_max, r_xy 

    def cal_neural_correlation(self, x, y):
        ## for mem test, set x as longer test clip + response time, set y as in movie signals
        #return np.sum(np.dot(x,y))
        #num_neuron = x.shape[0]
        neural_corr = scipy.signal.correlate2d(x, y, 'valid') 
        time_lag = np.argmax(neural_corr)
        #time_lag = np.ceil(len(r_xy)/2)-np.argmax(r_xy)
        return time_lag, np.max(neural_corr), neural_corr
