import numpy as np 
import sys
import os 
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from experiment_scripts import figure2b, figure2c, figure2def


project_folder = "/Users/yipengzhang/Documents/GitHub/Movie_Analysis/"
path = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD_random_label"
path = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/CNN_multi_2_KLD_random_label"
#path = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
result_folder = os.path.join(project_folder,"final_result_outputs")
if "LSTM" in path:
    model_option = "LSTM"
else:
    model_option = "CNN"
output_folder = "/media/yipeng/data/movie_2021/Movie_Analysis/final_result_outputs/CNN_shuffle_label"

######Figure 2#####
figure2def(path,output_folder)