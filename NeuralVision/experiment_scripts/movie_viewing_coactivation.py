import sys
import os 
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from scipy.io import loadmat, savemat
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_erasing_multi_regions_2/LSTM_retrain/435/LAH_LEC_LO_LPHG_LTP_RPHG"
path = "/media/yipeng/data/movie/data_movie_analysis_final/LSTM_multi_2_KLD"
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_erasing_multi_check1_2/CNN_retrain/431/LIP_RSTP_RSTA_RSS"
patientNums = ['431', '433', '435', '436', '439', '441', '442', '444', '445', '452']
output_path = "/media/yipeng/data/movie/Movie_Analysis/final_result_outputs"
res = {}
for p in patientNums:
    path_p = os.path.join(path, p )
    kfoldPerformance = KfoldPerformance.generate_bypath(path_p)
    res["p_"+ p] = kfoldPerformance.get_character_coactivation()
if "LSTM" in path:
    model = "LSTM"
if "CNN" in path:
    model = "CNN"
print(res)
output_folder = os.path.join(output_path, model)
fn = os.path.join(output_folder,"figure4_si_model_learnt_association.mat")
savemat(fn,res)