import numpy as np
import os
from CharacterPerformance import DataSample, CharacterPerformance
from data import character_dict, frame_dir, sr_final_movie_data,num_characters, path_to_cnn_result
from performance_utilities import * 
from utilities import cal_f1_score
class KfoldPerformance(object):
    def __init__(self, k, patient_name, model_option="LSTM"):
        self.k = k
        self.all_model_performance :[ModelPerformance] = []
    
    def add(self, model_performance):
        self.all_model_performance.append(model_performance)
    
    @staticmethod
    def generate_bypath(folder_path_name):
        file_paths = os.listdir(folder_path_name) 
        model_list = []
        folder_path_name_s = folder_path_name.split("/")
        patient_name = folder_path_name_s[-1]
        model_option = folder_path_name_s[-2].split("_")[0]
        for path_fn in file_paths:
            if os.path.isdir(os.path.join(folder_path_name, path_fn)):
                path = os.path.join(folder_path_name, path_fn, "model_results.npz")
                path_s = path.split("/")
                k = path_s[-2]
                model_info = np.load(path, allow_pickle=True)
                onefold_model = ModelPerformance.generate(k, patient_name, model_option, model_info['outputs'],model_info['labels'], model_info['frame_names'])
                model_list.append(onefold_model)
        kfoldPerformance = KfoldPerformance(len(model_list), patient_name, model_option)
        kfoldPerformance.all_model_performance = model_list
        return kfoldPerformance
        
    def get_character_accuracy(self):
        accuracy = np.zeros(num_characters) 
        for i in self.all_model_performance:
            accuracy += i.get_character_accuracy()
        return accuracy*1.0/self.k 

    def get_accuracy(self):
        return np.mean(self.get_character_accuracy())

    def get_character_loss(self):
        loss = np.zeros(num_characters) 
        for i in self.all_model_performance:
            loss += i.get_character_loss()
        return loss*1.0/self.k 

    def get_loss(self):
        return np.mean(self.get_character_loss())
    
    def get_confusion_mat(self):
        return np.sum(self.get_character_confusion_mat(), axis=0)

    def get_character_confusion_mat(self):
        stats = np.zeros((num_characters, 3,3))
        for i in self.all_model_performance:
            stats += i.get_character_confusion_mat() 
        return stats
    
    def get_character_precison(self):
        stats_arr = np.zeros(num_characters)
        confusion_mat = self.get_character_confusion_mat()
        for character_idx in range(num_characters):
            character_confusion_mat = confusion_mat[character_idx,:,:]
            stats_arr[character_idx] = character_confusion_mat[0,0]/(character_confusion_mat[0,0]+character_confusion_mat[1,0])
        return stats_arr

    def get_character_confusion_mat_var(self):
        stats = np.zeros((len(self.all_model_performance), num_characters, 3,3))
        for i, model_performance in enumerate(self.all_model_performance):
            all_confusion = model_performance.get_character_confusion_mat()
            cnf = all_confusion/np.sum(all_confusion,axis=-1)[:,:,None]
            #print(cnf)
            stats[i,:,:,:] = cnf
        return np.var(stats, axis=0)

    def get_character_f1_score(self):
        stats = np.zeros(num_characters)
        for performance in self.all_model_performance:
            stats += performance.get_character_f1_score()
        return stats/self.k
    
    def get_character_coactivation(self):
        stats = []
        for performance in self.all_model_performance:
            stats.append(performance.get_character_correlation())
        return np.mean(stats, axis=0)


class ModelPerformance(object):
    def __init__(self, k, patient_name, model_option="LSTM"):
        self.k = k
        self.model_option = model_option
        self.all_character_performance:{int, CharacterPerformance} = {}
        self.labels = None
        self.probs = None 
        self.__construct()

    @staticmethod
    def generate( k, patient_name, model_option, prob_list, label_list, frame_number_list):
        model_performance = ModelPerformance(k, patient_name, model_option)
        model_performance.construct(prob_list, label_list, frame_number_list)
        return model_performance
    
    @staticmethod
    def construct_analysis(k, patient_name ,model_option, character_stats):
        model_performance = ModelPerformance(k, patient_name, model_option)
        for character_index in character_stats.keys():
            probs = character_stats[character_index]["probs"]
            labels = character_stats[character_index]["labels"]
            frames = character_stats[character_index]["frames"]
            model_performance.all_character_performance[character_index] = CharacterPerformance.construct_analysis(character_index,probs,labels,frames)
        return model_performance
    def __gt__(self, other):
        return self.k > other.k

    def __construct(self):
        self.__construct_character_performances()

    def __construct_character_performances(self):
        for i in range(num_characters):
            self.all_character_performance[i] = CharacterPerformance(i)

    def construct(self, prob_list, label_list, frame_number_list):
        for i in range(len(prob_list)):
            self.__update(prob_list[i], label_list[i],frame_number_list[i])

    def __update(self, batch_prob, batch_label, batch_frame_number):
        for batch_index in range(len(batch_prob)):
            frame_num = batch_frame_number[batch_index]
            probs = np.squeeze(batch_prob[batch_index])
            labels = np.squeeze(batch_label[batch_index])
            for character_idx in range(num_characters):
                self.all_character_performance[character_idx].update(probs[character_idx],labels[character_idx], frame_num)
    
    def get_overall_accuracy(self):
        return np.mean(self.get_character_accuracy())

    def get_character_accuracy(self):
        accuracy = np.zeros(num_characters)
        for character_idx in self.all_character_performance.keys():
            accuracy[character_idx] = self.all_character_performance[character_idx].get_accuracy()
        return accuracy

    def get_overall_loss(self):
        return np.mean(self.get_character_loss())
    
    def get_character_loss(self):
        loss = np.zeros(num_characters)
        for character_idx in self.all_character_performance.keys():
            loss[character_idx] = self.all_character_performance[character_idx].get_loss()
        return loss

    def get_len(self):
        return self.all_character_performance[0].get_len()

    def get_frame_loss(self):
        stats = np.zeros((num_characters, self.get_len()))
        for i in range(num_characters):
            stats[i,:] = self.all_character_performance[i].get_frame_loss()
        return stats

    def get_confusion_mat(self):
        return np.sum(self.get_character_confusion_mat, axis=0)                
    
    def get_character_confusion_mat(self):
        stats_arr = np.zeros((num_characters, 3, 3))
        for character_idx in range(num_characters):
            stats_arr[character_idx,:,:] = self.all_character_performance[character_idx].get_confusion_mat()
        return stats_arr

    def get_character_roc(self):
        stats = {}
        for character_idx in range(num_characters):
            stats[character_idx] = [self.all_character_performance[character_idx].roc_curve()]  
        return stats
    
    def get_all_probs(self):
        if self.probs is None:
            self.probs = []
            for character_idx in range(num_characters):
               character_prob = self.all_character_performance[character_idx].get_all_probalities()
               print(character_prob.shape)
               self.probs.append()   
            self.probs = np.concatenate(self.probs)
        return self.probs
    
    def get_all_prediction(self):
        predictions = []
        for character_idx in range(num_characters):
            character_pred = self.all_character_performance[character_idx].get_all_predictions()
            predictions.append(character_pred)   
        predictions= np.vstack(predictions)
        return predictions
    
    def get_all_labels(self):
        if self.labels is None:
            self.labels = []
            for character_idx in range(num_characters):
                self.labels.append[self.all_character_performance[character_idx].get_all_labels()]
            self.labels = np.concatenate(self.labels)
        return self.labels 

    def roc_curve(self):
        labels = self.get_all_labels()
        probs = self.get_all_probs()
        fpr, tpr, thresholds = roc_curve(labels, probs)
        return fpr, tpr, thresholds

    def get_character_f1_score(self):
        result = np.zeros(num_characters)
        stats = self.get_character_confusion_mat()
        for character_idx in range(num_characters):
            character_stats = stats[character_idx,:,:]
            tp = character_stats[0,0]
            fp = character_stats[0,1]
            fn = character_stats[1,0]
            result[character_idx] = cal_f1_score(tp,fp,fn)
        return result
    
    def get_tpfp(self):
        stats_board = np.zeros((4, 18576))
        label_board = np.zeros((4, 18576)) - 1
        for character_idx in range(num_characters):
            character_performance = self.all_character_performance[character_idx]
            for dp in character_performance.data_samples:
                stats_board[character_idx, int(dp.frame_number/4)] = dp.tp_fp_tn_fn()
                label_board[character_idx, int(dp.frame_number/4)] = dp.label
        return stats_board, label_board
    
    def get_all_stats(self):
        res = {}
        for character_index in self.all_character_performance:
            res[character_index] = self.all_character_performance[character_index].get_all_stats()
        return res
    
    def get_character_correlation(self):
        res = np.zeros((num_characters, num_characters))
        predictions = self.get_all_prediction()
        #print(predictions[0,0:10])
        for character_idx in range(num_characters):
            for character_cond in range(num_characters):
                cond_sample_point = np.where(predictions[character_cond] == 0)[0]
                cond_sample_point_sum = len(cond_sample_point)
                check_pred = len(np.where(predictions[character_idx,cond_sample_point]==0)[0])
                res[character_cond, character_idx] = check_pred/cond_sample_point_sum
        return res