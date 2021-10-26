import numpy as np
from data import character_dict, frame_dir, sr_final_movie_data,num_characters, path_to_cnn_result
from performance_utilities import *


class CharacterPerformance(object):
    """Computes and stores the average and current value"""
    def __init__(self, charecter_name):
        self.charecter_name = charecter_name
        self.reset()

    @staticmethod
    def construct_analysis(character_name, probs, labels, frames):
        characterPerformance = CharacterPerformance(character_name)
        for idx in range(len(probs)):
            if labels[idx] == 1:
                label = np.array([0,1,0])
            elif labels[idx] == 0:
                label = np.array([1,0,0])
            else:
                label = np.array([0,0,1])
            '''
            if probs[idx].min() == 0:
                print(probs[idx])
                #print(np.log(probs[idx]))
                #print(idx)
                #print(probs[idx])
            '''
            characterPerformance.update(probs[idx], label, frames[idx], mode="raw")
        return characterPerformance

    def reset(self):
        self.data_samples:[DataSample] = []
        self.labels = None
        self.probs = None 

    def update(self, prob, label, frame_number, mode = "log", label_mode="include_dnk"):
        data_sample = DataSample.construct(frame_number,label, prob, mode)
        self.data_samples.append(data_sample)
    
    def get_accuracy(self):
        cnf = self.get_confusion_mat()
        tp = 0.0
        for i in range(len(cnf)):
            tp += cnf[i, i]
        return tp *1.0/np.sum(cnf)

    def get_confusion_mat(self):
        matrix = np.zeros((3, 3))
        for dp in self.data_samples:
           matrix[dp.label, dp.prediction] += 1
        return matrix

    def get_len(self):
        return len(self.data_samples)

    def get_loss(self):
        loss = 0.0
        for dp in self.data_samples:
           loss += dp.loss
        return loss*1.0 / len(self.data_samples)
    
    def get_frame_loss(self):
        stats = np.zeros(self.get_len())
        self.data_samples.sort()
        for i, dp in enumerate(self.data_samples):
            stats[i] = dp.loss
        return statsdata_samplesples.sort()
        stats = []
        for dp in self.data_samples:
            stats.append(dp.prediction)
        return np.array(stats)


    def get_all_probalities(self):
        self.data_samples.sort()
        stats = []
        for dp in self.data_samples:
            stats.append(dp.prob)
        return np.array(stats)

    def get_all_labels(self):
        stats = []
        for dp in self.data_samples:
            label = dp.label
            if label == 2:
                label = 1
            stats.append(label)
        return np.array(stats)

    def get_all_frames(self):
        stats = []
        for dp in self.data_samples:
            stats.append(dp.frame_number)
        return np.array(stats)
    
    def roc_curve(self):
        if self.labels is None or self.probs is None:
            self.labels = self.get_all_labels()
            self.probs = self.get_all_probalities()
        fpr, tpr, thresholds = roc_curve(self.labels, self.probs)
        return fpr, tpr, thresholds

    def roc_optimum(self):
        fpr, tpr, thresholds = self.roc_curve()
        max_diff_index = np.argmax(tpr - fpr)
        optimal_prediction = self.probs < thresholds[max_diff_index]
        return optimal_prediction
    
    def get_all_stats(self):
        stats = {}
        stats["probs"] = self.get_all_probalities()
        stats["labels"] = self.get_all_labels()
        stats["frames"] = self.get_all_frames()
        return stats

class DataSample(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.__reset()

    def __reset(self):
        self.frame_number = -1
        self.label = -1 
        self.prediction = -1
        self.prob = []
        self.loss = -1
    
    def __gt__(self, other):
        return self.frame_number < other.frame_number

    def tp_fp_tn_fn(self):
        #TP 1 FP 2 TN 3 FN 4
        if self.label == 0:
            # TP # label yes, prediction yes 
            if self.prediction == 0:
                return 1
            # FN # label yes, prediction NO or DNK   
            else:
                return 4
        elif self.label == 1:
            # TN # label no, prediction no 
            if self.prediction == 1:
                return 3
            # FP # label no, prediction no 
            elif self.prediction == 0:
                return 2
        else:
            # TN # label DNK, prediction YES 
            if self.prediction == 0:
                return 1
        return 0

    @staticmethod
    def KLD_loss(y, yHat, mode="log"): 
        if y[2] == 1:
            return 0 
        loss = 0.0
        #print(y, yHat)
        if mode == "log":
            for idx in range(len(y)):
                if y[idx] != 0:
                    loss += y[idx] * (np.log(y[idx]) - yHat[idx])
        else:
            for idx in range(len(y)):
                if y[idx] != 0 and yHat[idx]!= 0:
                    loss += y[idx] * (np.log(y[idx]) - np.log(yHat[idx]))
        return loss

    @staticmethod ##[include_dnk,dnktono]
    def construct(frame_number, label, prob, mode="log", label_mode="dnktono"):
        data_sample = DataSample()
        data_sample.frame_number = frame_number
        if label_mode == "include_dnk":
            label == label
        else:
            if label[2] == 1:
                label = np.array([0,1,0])
        data_sample.label = np.argmax(label)
        if mode is "log":
            data_sample.prob = np.exp(prob)
        else:
            data_sample.prob = prob
        data_sample.prediction = np.argmax(prob)
        data_sample.loss = DataSample.KLD_loss(label,prob, mode)
        return data_sample
        