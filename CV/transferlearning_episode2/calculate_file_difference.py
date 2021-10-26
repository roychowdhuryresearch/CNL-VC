import os
import shutil
large_file = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/resnet_result_all"
small_file = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/resnet_result"
difference_result = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/difference"
class_dir = os.listdir(small_file) 
for c in class_dir:
    small_files = set(os.listdir(os.path.join(small_file, c)))
    large_folder = os.path.join(large_file, c)
    large_files = set(os.listdir(large_folder))
    difference_set = large_files - small_files
    for fn in difference_set:
        from_fn = os.path.join(large_folder, fn)
        to_folder = os.path.join(difference_result, c)
        if not os.path.exists(to_folder):
            os.mkdir(to_folder)
        to_fn = os.path.join(to_folder, fn)
        shutil.copyfile(from_fn, to_fn)