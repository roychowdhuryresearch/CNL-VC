from sklearn import manifold
from sklearn.cluster import KMeans
import pickle
import numpy as np
import os
import utilities
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

class Visualize:
    def __init__(self):
        super().__init__()
        self.fn_list =[]
        self.label_mapping = {}
        self.label_list = []
        self.table_list = []
    def construct(self, talble_fns):
        table_list = []
        for i in talble_fns:
            table_list.append(self.load_encodings_lookup(i))
        self.fn_list = list(table_list[0].keys())
        class_feature = self.get_all_class_feature(self.fn_list, table_list) 
        self.label_list = list(class_feature.keys())
        self.table_list = table_list
        
    def visualize2classes(self, class1, class2):
        feature1 = self.get_one_class(self.fn_list,self.table_list,class1)[:,0:128]
        print(feature1.shape)
        label1 = np.zeros(feature1.shape[0])+class1 
        feature2 = self.get_one_class(self.fn_list,self.table_list,class2)[:,0:128]
        print(feature2.shape)
        label2 = np.zeros(feature2.shape[0])+class2 
        X = np.concatenate((feature1, feature2), axis=0)
        y = np.concatenate((label1, label2), axis=None)
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        self.plot_embedding(X_tsne,y)  
    
    def visualize_classes(self, from_class_per, to_class_per):
        for i in range(int(from_class_per*len(self.label_list)), int(to_class_per*len(self.label_list))):
            label = self.label_list[i]
            feature = self.get_one_class(self.fn_list,self.table_list,label)
            label_array = np.zeros(feature.shape[0])+label 
            if i == int(from_class_per*len(self.label_list)):
                X = feature
                y = label_array
            else:
                if len(feature.shape) == 1 :
                    feature = feature.reshape(1, -1)
                X = np.concatenate((X, feature), axis=0)
                y =  np.concatenate((y, label_array), axis=None)
        count = 0
        color_m = []
        for i in range(y.shape[0]):
            if i == 0:
                color_m.append(count)
                continue
            if y[i] != y[i-1]:
                count = count + 1 
            color_m.append(count)
            
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        X_tsne = tsne.fit_transform(X)
        self.plot_embedding(X_tsne,y, color_m) 
        
        
    def get_all_class_feature(self,fn_list,table_list):
        res = {}
        lookup = {}
        for index in range(len(fn_list)):
            label = int(fn_list[index].split("_")[-1].split(".")[0])
            feature = self.get_all_feature(fn_list[index],table_list)
            if label in lookup:
                lookup[label] = lookup[label] + 1
                res[label] = res[label] + feature[:,0:128]
            else:
                lookup[label] = 1
                res[label] = feature[:,0:128]
        for label_num in lookup.keys():
            res[label_num] = res[label_num] *1.0/lookup[label_num]
        return res

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
            res_list.append(t[fn])
        return np.concatenate(res_list, axis= None).reshape(1, -1) 
   
    def get_one_class(self,fn_list,table_list, class_num):
        res = []
        for index in range(len(fn_list)):
            label = int(fn_list[index].split("_")[-1].split(".")[0])
            if label == class_num:
                res.append(self.get_all_feature(fn_list[index],table_list))
        res = np.squeeze(np.array(res))
        return res
    
    # Scale and visualize the embedding vectors
    def plot_embedding(self, X,y, color_m = None,title=None):
        if color_m == None:
            color_m = y
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        ax = plt.subplot(111)
        for i in range(X.shape[0]):
            plt.text(X[i, 0], X[i, 1], str(int(y[i])),
                    color=plt.cm.Set2(int(color_m[i])/50.),
                    fontdict={'weight': 'bold', 'size': 9})

        '''
        if hasattr(offsetbox, 'AnnotationBbox'):
            # only print thumbnails with matplotlib > 1.0
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - shown_images) ** 2, 1)
                if np.min(dist) < 4e-3:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X[i]]]
                imagebox = offsetbox.AnnotationBbox(
                    offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                    X[i])
                ax.add_artist(imagebox)
        '''
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)