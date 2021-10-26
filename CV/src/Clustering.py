from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import pickle
import numpy as np
import os
from utilities import *
import shutil
from scipy.spatial import distance
from project_setting import kmean_result_dir, filter_tracking_result_dir, stats_dir
class Cluster:
    def __init__(self, file_dir="./demo_result/"):
        super().__init__()
        self.kmeans = KMeans(n_clusters=30, random_state=3425)
        #self.kmeans = KMedoids(n_clusters=60, random_state=3425)
        self.fn_list =[]
        self.label_mapping = {}
        self.label_list = []
        self.features_each_class = {}
        self.less_face_group = load_pickle(join(file_dir,"less_face_group.pkl"))
        self.no_face_group =  load_pickle(join(file_dir,"no_face_group.pkl"))

    def construct(self, talble_fns):
        table_list = []
        for i in talble_fns:
            table_list.append(self.load_encodings_lookup(i))
        self.fn_list = list(table_list[0].keys())
        X = []
        for fn in self.fn_list:
            X.append(self.get_all_feature(fn, table_list))
        X = np.squeeze(np.array(X))
        self.kmeans.fit(X)
        self.get_label_mapping()
        
    def load_encodings_lookup(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
            print(len(lookup.keys()))
        return lookup
    
    def get_all_feature(self, fn, table_list):
        res_list = []
        for t in table_list:
            if isinstance(t[fn], list) and len(t[fn][0]) >= 128:
                res_list.append(t[fn][0][0:128])
            else:
                res_list.append(t[fn]/2)
        return np.concatenate(res_list, axis= None).reshape(1, -1) 
   
    def get_label_mapping(self):
       labels = self.kmeans.labels_
       for index in range(len(self.fn_list)):
           l = labels[index]
           file_label = int(self.fn_list[index].split("_")[-1].split(".")[0])
           if l in self.label_mapping:
               self.label_mapping[l].add(file_label)
           else:
               self.label_mapping[l] = set()
               self.label_mapping[l].add(file_label) 
    
    def export_label_mapping(self,original_dir, saved_dir, label_mapping=None):
        if label_mapping == None:
            label_mapping = self.label_mapping
        print(saved_dir)
        self.clean_folder(saved_dir)
        for k in label_mapping.keys():
            c = label_mapping[k]
            to_dir = join(saved_dir, str(k) +"_"+str(list(c)[0]))
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            for cc in c:
                from_dir = join(original_dir, str(cc))
                list_f = os.listdir(from_dir)
                for lf in list_f:
                    shutil.copy(os.path.join(from_dir, lf), os.path.join(to_dir,lf))
    
    def clean_folder(self, saved_fn):
        if not os.path.exists(saved_fn):
            os.mkdir(saved_fn)
        else:
            shutil.rmtree(saved_fn)
            os.mkdir(saved_fn)
            
    def construct_2(self, talble_fns, stage_2_folder=None):
        table_list = []
        for i in talble_fns:
            table_list.append(self.load_encodings_lookup(i))
        self.fn_list = list(table_list[0].keys())
        self.table_list = table_list
        class_feature = self.get_all_class_feature(self.fn_list, table_list) 
        class_feature_keys = set(list(class_feature.keys()))
        self.label_list = list(class_feature_keys - class_feature_keys.intersection(self.less_face_group) - class_feature_keys.intersection(self.no_face_group))
        if stage_2_folder is not None:
            all_label = set(list(map(int, os.listdir(stage_2_folder))))
            print(len(list(set(self.label_list).intersection(all_label))))
            self.label_list = list(set(self.label_list).intersection(all_label))
        X = []
        for label in self.label_list:
            X.append(class_feature[label])
        X = np.squeeze(np.array(X))
        for k in [40]:
            self.kmeans = KMeans(n_clusters=k, random_state=3425)
            self.kmeans.fit(X)
            interia = self.kmeans.inertia_
            print("k ",k, " interia ", interia)
        self.get_label_mapping_2(self.label_list)
    
    def get_all_class_feature(self,fn_list,table_list):
        res = {}
        lookup = {}
        for index in range(len(fn_list)):
            label = int(fn_list[index].split("_")[-1].split(".")[0])
            feature = self.get_all_feature(fn_list[index],table_list)
            if label in lookup:
                lookup[label] = lookup[label] + 1
                res[label] = res[label] + feature
            else:
                lookup[label] = 1
                res[label] = feature
        for label_num in lookup.keys():
            res[label_num] = res[label_num] *1.0/lookup[label_num]
        return res

    def get_label_mapping_2(self, label_list):
       labels = self.kmeans.labels_
       for index in range(len(label_list)):
           l = labels[index]
           file_label = label_list[index]
           if l in self.label_mapping:
               self.label_mapping[l].add(file_label)
           else:
               self.label_mapping[l] = set()
               self.label_mapping[l].add(file_label) 
    
    def get_feature_per_class(self,fn_list,table_list):
        res = {}
        for index in range(len(fn_list)):
            label = int(fn_list[index].split("_")[-1].split(".")[0])
            feature = self.get_all_feature(fn_list[index],table_list)
            if label in res:
                res[label].append(feature)
            else:
                res[label]= []
                res[label].append(feature)
        return res
    
    def calculate_inclass_dist(self, class_list):
        cluster_features = [] 
        for c in class_list:
            cluster_features.append(self.features_each_class[c])
        cluster_features_np = np.squeeze(np.concatenate(cluster_features), axis=1)
        if  cluster_features_np.shape[0] == 1:
            return 0, 0, 0, 0, 0
        feature_np = cluster_features_np
        dis_matrix = distance.cdist(feature_np, feature_np)
        nonzero = dis_matrix[np.nonzero(dis_matrix)]
        max_value = np.max(nonzero)
        min_value = np.min(nonzero)
        diff_value = max_value - min_value 
        var_dis = np.var(nonzero)*100
        mean_dis = np.mean(nonzero)
        return max_value, min_value, diff_value, var_dis, mean_dis
    
    def cluster_evaluation(self, threshold, stage_num = 1):
        if stage_num == 1:
            source_folder = filter_tracking_result_dir
            good_class_folder = join(kmean_result_dir, "cluster_good")
            bad_class_folder = join(kmean_result_dir, "cluster_bad")
            good_class_fn = join(kmean_result_dir, "good_class.pkl")
            bad_class_fn = join(kmean_result_dir, "bad_class.pkl")
        else:
            source_folder = filter_tracking_result_dir
            good_class_folder = join(kmean_result_dir,"cluster_good_stage"+str(stage_num))
            bad_class_folder = join(kmean_result_dir,"cluster_bad_stage"+str(stage_num))
            good_class_fn =  join(kmean_result_dir,"good_class_stage"+str(stage_num)+".pkl")
            bad_class_fn =  join(kmean_result_dir,"bad_class_stage"+str(stage_num)+".pkl")
        self.features_each_class = self.get_feature_per_class(self.fn_list, self.table_list)
        good_class = {}
        bad_class = {}
        for k in sorted(self.label_mapping.keys()):
            label_list = list(self.label_mapping[k])
            print("++++++++++")
            print(k)
            print(label_list)
            max_dis, min_dis, dif_dis,var_dis, mean_dis =  self.calculate_inclass_dist(label_list)
            print(mean_dis)
            if mean_dis >= threshold:
                bad_class[k] = label_list
            else:
                good_class[k] = label_list
        self.export_label_mapping(source_folder, good_class_folder, good_class)
        self.export_label_mapping(source_folder, bad_class_folder, bad_class)
        dump_pickle(good_class_fn, good_class)
        dump_pickle(bad_class_fn, bad_class)
    
    def standardize(self, a):
        a = (a - np.mean(a)) / np.std(a)
        return a