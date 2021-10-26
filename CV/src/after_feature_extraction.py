from Clustering import Cluster
from project_setting import feature_dir,filter_tracking_result_dir, kmean_result_dir, stats_dir,knn_result_dir
import os
from utilities import *
cluster = Cluster(feature_dir)
fname_list = [os.path.join(feature_dir,"face_encoding_lookup_cnn.pkl")]
cluster.construct_2(fname_list)
cluster.cluster_evaluation(0.45)
#cluster.export_label_mapping(filter_tracking_result_dir, kmean_result_dir)

from KNNboostrap import KNNboostrap
pickle_list = [join(kmean_result_dir, "good_class.pkl"), join(kmean_result_dir, "bad_class.pkl")]
fname_list = [os.path.join(feature_dir,"face_encoding_lookup_cnn.pkl")]
knn = KNNboostrap()
knn.construct(pickle_list, fname_list)
knn.inference(0.45, 0.7)


from Clustering import Cluster
cluster = Cluster(feature_dir)
fname_list = [os.path.join(feature_dir,"face_encoding_lookup_cnn.pkl"),os.path.join(feature_dir,"cloth_histogram_lookup.pkl") ]
cluster.construct_2(fname_list, join(knn_result_dir, "cluster_bad_knn"))
cluster.cluster_evaluation(0.5, stage_num=2)
cluster.export_label_mapping(filter_tracking_result_dir, join(kmean_result_dir,"group_result_stage2"))

from KNNboostrap import KNNboostrap
pickle_list = [join(kmean_result_dir, "good_class_stage2.pkl"), join(kmean_result_dir, "bad_class_stage2.pkl")]
fname_list = [os.path.join(feature_dir,"face_encoding_lookup_cnn.pkl")]
knn = KNNboostrap()
knn.construct(pickle_list, fname_list)
knn.inference(0.45, 0.7,stage_num=2)


import time
from FaceGroupingGPU import FaceGroupingGPU
start_time = time.time()
fg = FaceGroupingGPU(filter_tracking_result_dir)
fname_list = ["./demo_result/face_encoding_lookup_cnn.pkl","./demo_result/face_histogram_lookup.pkl","./demo_result/cloth_histogram_lookup.pkl","./demo_result/pic_histogram.pkl"]
fname_list = [join(feature_dir,"face_encoding_lookup_cnn.pkl"), join(feature_dir,"cloth_histogram_lookup.pkl")]
pickle_list = [join(knn_result_dir, "good_class_knn_stage2.pkl"), join(knn_result_dir, "bad_class_knn_stage2.pkl")]
fg.construct(fname_list, pickle_list)
print("--- %s seconds ---" % (time.time() - start_time))


file_dir = "/media/yipeng/Sandisk/movie_tracking/total_files_samped_4/"
image_dir = "/media/yipeng/Sandisk/movie_tracking/total_frames_samped_4/"
size_threshold = 15000
from Core import Core
core:Core = Core(file_dir, image_dir, size_threshold)
core.tracking()
core.filter_tracking_result(os.path.join(stats_dir, "scene_list.txt"), save_img=False) 
core.network_filtering(threshold=1, distance_th=200)
core.result_reassign()