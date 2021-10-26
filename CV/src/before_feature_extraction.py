from project_setting import result_dir, stats_dir, yolo_result_dir
import os 
file_dir = "/media/yipeng/Sandisk/movie_tracking/total_files_samped_4/"
image_dir = "/media/yipeng/Sandisk/movie_tracking/total_frames_samped_4/"
size_threshold = 15000
from Core import Core
core:Core = Core(file_dir, image_dir, size_threshold)
core.tracking()
core.filter_tracking_result(os.path.join(stats_dir, "scene_list1.txt"), save_img=True) 