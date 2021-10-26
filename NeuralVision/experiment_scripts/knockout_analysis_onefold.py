import sys
sys.path.append(".")
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np 
import sys
from data import project_dir ,path_to_training_data, path_to_model, patientNums, sr_final_movie_data, patient_features,\
    character_dict, path_to_matlab_generated_movie_data, num_characters, feature_time_width, model_option
from neural_correlation.KnockoutPerformance import KnockoutPerformance, KnockoutKfoldPerformance
from neural_correlation.ModelPerformance import ModelPerformance


folder = "/media/yipeng/toshiba/movie/Movie_Analysis/knockout_test_CNN_KLD"
options = ["reg", "mircowire"]
patient = "433"
option = "reg"

region_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"features_mat_regions_clean.npy")
region = np.load(region_fn, allow_pickle=True)
reg = []
for d in region:
    reg.append(d[0])
knockout_region = KnockoutPerformance.construct_analysis(folder,patient,0,"reg")
knockout_region.tags = np.array(reg)

microarr_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"channel_data.mat")
microarr_temp = loadmat(microarr_fn)["channel_reg_info"][0]
microarr = []
for m in microarr_temp:
    microarr.append(m[0][0][0])
knockout_mircowire = KnockoutPerformance.construct_analysis(folder,patient,0, "mircowire")
knockout_mircowire.set_tag(np.array(microarr))

'''
print(knockoutPerformance.get_f1_stats())
print(knockoutPerformance.get_acc_stats())
print(knockoutPerformance.get_loss_stats())
print(knockout_region.get_loss_tag_expand())
print(knockout_mircowire.get_loss_tag_expand())
'''
