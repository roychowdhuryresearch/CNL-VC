import numpy as np
from sklearn.neighbors import NearestNeighbors
from utilities import *
from scipy import stats
from matplotlib import pyplot as plt
from project_setting import feature_dir,filter_tracking_result_dir, knn_result_dir, stats_dir
class KNNboostrap:
    
    def __init__(self):
        self.nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree',n_jobs=50)
        self.fn_list = []
        self.table_list = []
        self.class_features = {int:np.array}
        self.super_class_y = []
        self.super_class_feature = {int:np.array}
        self.testing_class_feature = {int:np.array}
        self.good_cluster = {int:[]}

    def construct(self, clusterfn:[], talble_fns:[]):
        good_cluster, bad_cluster = self.load_cluster_reference(clusterfn)
        self.good_cluster = good_cluster 
        table_list = []
        for i in talble_fns:
            table_list.append(load_pickle(i))
        self.fn_list = list(table_list[0].keys())
        self.table_list = table_list    
        self.class_features = self.get_all_class_feature(self.fn_list, self.table_list)
        self.super_class_feature, self.super_class_y ,self.testing_class_feature = self.split_feature(good_cluster, bad_cluster,self.class_features)
        self.train_knn(self.super_class_feature, self.super_class_y)

    def load_cluster_reference(self, clusterfn):
        good_list = []
        bad_list = []
        for c in clusterfn:
            if "good" in c:
                good_list.append(c)
            else:
                bad_list.append(c)
        return self.cluster_concate(good_list) , self.cluster_concate(bad_list)
        
    def cluster_concate(self, cluster_list):
        res:{int, set()} = {}
        for c in cluster_list:
            if len(res.keys()) == 0:
                res = load_pickle(c)
            else:
                temp = load_pickle(c)
                for key in temp.keys():
                    k = key
                    while k in res:
                        k = k + 1
                    res[k] = temp[key]
        return res


    def train_knn(self, super_class_feature, super_class_y):
        self.nbrs.fit(super_class_feature)

    def inference(self, distance_th, class_percentage, stage_num=1):
        source_folder = filter_tracking_result_dir
        if stage_num == 1:
            good_class_folder = join(knn_result_dir, "cluster_good_knn")
            bad_class_folder =  join(knn_result_dir, "cluster_bad_knn")
            good_class_fn =  join(knn_result_dir, "good_class_knn.pkl")
            bad_class_fn =  join(knn_result_dir, "bad_class_knn.pkl")
        else:
            source_folder = filter_tracking_result_dir
            good_class_folder = join(knn_result_dir,"cluster_good_knn"+str(stage_num))
            bad_class_folder = join(knn_result_dir,"cluster_bad_knn"+str(stage_num))
            good_class_fn = join(knn_result_dir,"good_class_knn_stage"+str(stage_num)+".pkl")
            bad_class_fn = join(knn_result_dir,"bad_class_knn_stage"+str(stage_num)+".pkl")
        bad = {}
        super_class_y = np.array(self.super_class_y) 
        counter = 0
        distance_median = []
        percentage_list = []
        for label in self.testing_class_feature.keys():
            one_test_feature = self.testing_class_feature[label][0]
            if len(one_test_feature.shape) == 1:
                one_test_feature = one_test_feature.reshape(1,-1)
            distances, indices =  self.nbrs.kneighbors(one_test_feature)
            distances = distances.flatten()
            distance_median.append(np.median(distances))
            if np.median(distances) > distance_th:
                bad[counter] = [label]
                counter = counter + 1
                continue
            indices = indices.flatten()
            selection = super_class_y[indices]
            class_num = stats.mode(selection)[0][0]
            per = sum(selection==class_num)*1.0/selection.shape[0]
            percentage_list.append(per)
            if per < class_percentage:
                bad[counter] = [label]
                counter = counter + 1
                continue
            self.good_cluster[class_num].append(label)
        self.export_label_mapping(source_folder, good_class_folder, self.good_cluster) 
        self.export_label_mapping_bad(source_folder, bad_class_folder, bad)  
        dump_pickle(good_class_fn, self.good_cluster)
        dump_pickle(bad_class_fn, bad)         
        self.plot_histogram(distance_median)
        self.plot_histogram(percentage_list)
    
    def split_feature(self, good_cluster, bad_cluster, class_features):
        good = {}
        bad = {}
        good_count = {}
        good_cluster_re = reverse_key_list_dict(good_cluster)
        bad_cluster_re = reverse_key_list_dict(bad_cluster)
        for label in class_features.keys():
            if label in good_cluster_re:
                super_label = good_cluster_re[label]
                if super_label in good:
                    good[super_label].append(class_features[label])
                else:
                    good[super_label] = []
                    good[super_label].append(class_features[label])
            #elif label in bad_cluster_re :
            else:
                if label in bad:
                    bad[label].append(class_features[label])
                else:
                    bad[label] = []
                    bad[label].append(class_features[label])
        super_class_feature = []
        for super_label in good.keys():
            one_supre_class_feature_list = good[super_label]
            one_supre_class_feature = np.vstack(one_supre_class_feature_list)
            good_count[super_label] = one_supre_class_feature.shape[0]
            super_class_feature.append(one_supre_class_feature)
        super_class_feature = np.vstack(super_class_feature)
        
        super_class_y = []
        for super_label in good_count:
            base = [super_label]
            super_class_y.extend(base*good_count[super_label])
            
        return super_class_feature, super_class_y, bad
        
        
    def get_all_class_feature(self,fn_list,table_list):
        res = {}
        lookup = {}
        for index in range(len(fn_list)):
            label = int(fn_list[index].split("_")[-1].split(".")[0])
            feature = self.get_all_feature(fn_list[index],table_list)
            if label in lookup:
                res[label].append(feature)
            else:
                lookup[label] = 1
                res[label] = []
                res[label].append(feature)
        for label_num in lookup.keys():
            res[label_num] = np.squeeze(np.array(res[label_num]))
        return res
    
    def get_all_feature(self, fn, table_list):
        res_list = []
        for t in table_list:
            if isinstance(t[fn], list) and len(t[fn][0]) >= 128:
                res_list.append(t[fn][0][0:128])
            else:
                res_list.append(t[fn])
        return np.concatenate(res_list, axis= None).reshape(1, -1) 
    
        
    def export_label_mapping(self,original_dir, saved_dir, label_mapping=None):
        clean_folder(saved_dir)
        for k in label_mapping.keys():
            c = label_mapping[k]
            to_dir = join(saved_dir,str(k) +"_"+str(list(c)[0]))
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            for cc in c:
                from_dir = join(original_dir, str(cc))
                list_f = os.listdir(from_dir)
                for lf in list_f:
                    shutil.copy(os.path.join(from_dir, lf), os.path.join(to_dir,lf))
    
    def export_label_mapping_bad(self,original_dir, saved_dir, label_mapping=None):
        clean_folder(saved_dir)
        for k in label_mapping.keys():
            c = label_mapping[k]
            to_dir = join(saved_dir,str(list(c)[0]))
            if not os.path.exists(to_dir):
                os.mkdir(to_dir)
            for cc in c:
                from_dir = join(original_dir , str(cc))
                list_f = os.listdir(from_dir)
                for lf in list_f:
                    shutil.copy(os.path.join(from_dir, lf), os.path.join(to_dir,lf))
    


    def plot_histogram(self, data):
        data = np.array(data)
        # fixed bin size
        bins = np.arange(0, 1, 0.1) # fixed bin size

        plt.xlim([min(data)-0.1, max(data)+0.1])

        plt.hist(data, bins=bins, alpha=0.5)

        plt.show()