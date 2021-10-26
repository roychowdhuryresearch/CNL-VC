import os
import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance

folder = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result/CNN_erasing_multi_check_2/CNN_retrain"
patient = "431"
patient_folder = os.path.join(folder,patient)
region = os.listdir(patient_folder)
res_reg = {}
for r in sorted(region):
    region_folder = os.path.join(patient_folder, r)
    region_stats = KfoldPerformance.generate_bypath(region_folder)
    res_reg[r] = region_stats.get_character_f1_score()
print(res_reg)