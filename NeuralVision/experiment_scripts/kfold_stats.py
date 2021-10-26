import numpy as np 
import sys
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_erasing_multi_regions_2/LSTM_retrain/435/LAH_LEC_LO_LPHG_LTP_RPHG"
path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD/431"
#path = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD/431"
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_erasing_multi_check1_2/CNN_retrain/431/LIP_RSTP_RSTA_RSS"
kfoldPerformance = KfoldPerformance.generate_bypath(path)

character_acc= kfoldPerformance.get_character_accuracy()
overall_acc = kfoldPerformance.get_accuracy()
confusion_mat = kfoldPerformance.get_confusion_mat()
character_confusion_mat = kfoldPerformance.get_character_confusion_mat()
character_confusion_mat_var = kfoldPerformance.get_character_confusion_mat_var()
f1_score = kfoldPerformance.get_character_f1_score()
character_precison = kfoldPerformance.get_character_precison()
print(character_acc)
print(overall_acc)
print(confusion_mat)
print(character_confusion_mat)
print(character_confusion_mat_var)
print(f1_score)
print(character_precison)