import os
project_dir = "/media/yipeng/Sandisk/movie_tracking/"
working_dir = "/media/yipeng/Sandisk/movie_tracking/tracking/"
result_dir = os.path.join(working_dir, "pipeline_filtered")
stats_dir = os.path.join(working_dir, "stats_filtered")
yolo_result_dir = os.path.join(project_dir, "files")
filter_tracking_result_dir = os.path.join(result_dir, "filter_tracking_result")
feature_dir = os.path.join(result_dir, "features")
kmean_result_dir = os.path.join(result_dir, "kmean_result")
knn_result_dir = os.path.join(result_dir, "knn_result")
group_result_network_dir = os.path.join(result_dir, "group_result_network")
if not os.path.exists(kmean_result_dir):
    os.mkdir(kmean_result_dir)

if not os.path.exists(knn_result_dir):
    os.mkdir(knn_result_dir)

if not os.path.exists(group_result_network_dir):
    os.mkdir(group_result_network_dir)
