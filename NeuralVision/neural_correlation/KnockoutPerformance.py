import numpy as np
import os
from ModelPerformance import KfoldPerformance, ModelPerformance
from data import character_dict, frame_dir, sr_final_movie_data,num_characters, path_to_cnn_result
from performance_utilities import removeDuplicates
from collections import OrderedDict 
from utilities import *
class KnockoutPerformance(object):
    def __init__(self, patient, tags):
        self.patient = patient
        self.tags = np.array(tags) 
        self.tag_single = np.array(removeDuplicates(tags))
        self.tagErasingStats = OrderedDict()

    @staticmethod
    def construct_analysis(folder_dir, patient ,kfold=0, mode = "reg"):
        model_option = folder_dir.split("/")[-1].split("_")[-2]
        patient = patient
        file_name = os.path.join(folder_dir, patient, str(kfold), mode+"_stats.pkl")
        stats = load_pickle(file_name)
        knockoutPerformance = KnockoutPerformance(patient, stats.keys())
        for tag in stats.keys():
            knockoutPerformance.add(tag, ModelPerformance.construct_analysis(str(kfold),patient,model_option,stats[tag]))
        return knockoutPerformance 

    def add(self, tag, modelPerformance):
        self.tagErasingStats[tag] = modelPerformance
    
    def set_tag(self, tag):
        self.tags = tag
        self.tag_single = np.array(removeDuplicates(self.tags))

    def get_acc_stats(self):
        tag_all = list(self.tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for tag_idx in range(len(tag_all)):
            tag = tag_all[tag_idx]
            stats[tag_idx,:] = self.tagErasingStats[tag].get_character_accuracy()
        return stats, tag_all
    
    def get_loss_stats(self):
        tag_all = list(self.tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        baseline = self.tagErasingStats["baseline"].get_character_loss() 
        for tag_idx in range(len(tag_all)):
            tag = tag_all[tag_idx]
            if tag == "baseline":
                baseline = 0
            stats[tag_idx,:] = self.tagErasingStats[tag].get_character_loss() - baseline
        return stats, tag_all
    
    def get_loss_stats_normalized(self):
        tag_all = list(self.tagErasingStats.keys())
        tag_sorted, tag_counts = np.unique(np.array(self.tags), return_counts=True)
        stats = np.zeros((len(tag_all),num_characters))
        baseline = self.tagErasingStats["baseline"].get_character_loss()
        for tag_idx in range(len(tag_all)):
            tag = tag_all[tag_idx]
            tag_sorted = np.array(list(tag_sorted))
            if tag != "baseline":
                difference = self.tagErasingStats[tag].get_character_loss() - baseline
                index = np.where(tag_sorted == tag)[0]
                cnt = tag_counts[index]
                stats[tag_idx,:] = difference/cnt
            else:
                stats[tag_idx,:] = self.tagErasingStats["baseline"].get_character_loss()
        return stats, tag_all


    def get_f1_stats(self):
        tag_all = list(self.tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for tag_idx in range(len(tag_all)):
            tag = tag_all[tag_idx]
            stats[tag_idx,:] = self.tagErasingStats[tag].get_character_f1_score()
        return stats, tag_all

    def get_loss_tag_expand(self):
        expand_loss = np.zeros((len(self.tags), num_characters))
        print(self.tag_single)
        print(self.tagErasingStats.keys())
        print(self.tags)
        for tag in self.tag_single:
            loc = np.where(self.tags == tag)[0]
            stats_temp = self.tagErasingStats[tag].get_character_loss() - self.tagErasingStats["baseline"].get_character_loss()  
            expand_loss[loc,:] = np.tile(stats_temp, (len(loc), 1))
        return expand_loss
    
    def get_acc_tag_expand(self):
        expand_loss = np.zeros((len(self.tags), num_characters))
        for tag in self.tag_single:
            loc = np.where(self.tags == tag)[0]
            stats_temp = self.tagErasingStats[tag].get_character_accuracy()
            expand_loss[loc,:] = np.tile(stats_temp, (len(loc),1))
        return expand_loss
    
    def get_frame_loss(self):
        stats = {}
        for tag in self.tagErasingStats.keys():
            stats[tag] = self.tagErasingStats[tag].get_frame_loss()
        return stats
    
    def get_knockout_stats(self):
        res = {}
        for tag in self.tagErasingStats.keys():
            res[tag] = self.tagErasingStats[tag].get_all_stats()
        return res

class KnockoutKfoldPerformance(object):
    def __init__(self, k , patient):
        self.tags = None  #neuron based
        self.tag_single = None #unique
        self.kfoldtagErasing :[KnockoutPerformance] = []
        self.k = k

    def add(self, knockoutPerformance):
        if len(self.kfoldtagErasing) == 0:
            self.tags = knockoutPerformance.tags
            self.tag_single = knockoutPerformance.tag_single
        self.kfoldtagErasing.append(knockoutPerformance)

    def get_f1_stats(self):
        tag_all = list(self.kfoldtagErasing[0].tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for kfd_performance in self.kfoldtagErasing:
            stats_temp, _ = kfd_performance.get_f1_stats()
            stats += stats_temp
        return stats/len(self.kfoldtagErasing), tag_all

    def get_acc_stats(self):
        tag_all = list(self.kfoldtagErasing[0].tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for kfd_performance in self.kfoldtagErasing:
            stats_temp, _ = kfd_performance.get_acc_stats()
            stats += stats_temp
        return stats/len(self.kfoldtagErasing), tag_all
    
    def get_loss_stats(self):
        tag_all = list(self.kfoldtagErasing[0].tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for kfd_performance in self.kfoldtagErasing:
            stats_temp, _ = kfd_performance.get_loss_stats()
            stats += stats_temp
        return stats/len(self.kfoldtagErasing), tag_all
    
    '''
    Here we use the normalized loss: (original - baseline) / num-neurons
    '''
    def get_loss_stats_normalized(self):
        tag_all = list(self.kfoldtagErasing[0].tagErasingStats.keys())
        stats = np.zeros((len(tag_all),num_characters))
        for kfd_performance in self.kfoldtagErasing:
            stats_temp, _ = kfd_performance.get_loss_stats_normalized()
            stats += stats_temp
        return stats/len(self.kfoldtagErasing), tag_all

    '''
    here we use the sum of the loss: (original - baseline)
    '''
    def get_loss_tag_expand(self):
        stats = np.zeros((len(self.tags), num_characters))
        for kfd_performance in self.kfoldtagErasing:
            stats += kfd_performance.get_loss_tag_expand()
        return stats/len(self.kfoldtagErasing)
    
        