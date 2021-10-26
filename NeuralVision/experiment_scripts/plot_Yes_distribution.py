import numpy as np 
import sys
import os 
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
sys.path.append("./neural_correlation")
from neural_correlation.ModelPerformance import ModelPerformance, KfoldPerformance
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_erasing_multi_regions_2/LSTM_retrain/435/LAH_LEC_LO_LPHG_LTP_RPHG"
path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
#path = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_erasing_multi_check1_2/CNN_retrain/431/LIP_RSTP_RSTA_RSS"
patientNums = ['431', '433', '435', '436', '439', '441', '442', '444', '445', '452']
res = "/media/yipeng/data/movie/Movie_Analysis/draft_result"
for p in patientNums:
    print(p)
    path_p = os.path.join(path, p)
    kfold_yes_outputs = []
    for i in range(5):
        folder_path_name = os.path.join(path_p, str(i))
        path_fn = os.path.join(folder_path_name, "model_results.npz")
        model_info = np.load(path_fn, allow_pickle=True)
        outputs = np.concatenate(model_info["outputs"])[:,:,0]
        loc = np.argmax(np.squeeze(np.concatenate(model_info["labels"])), axis=-1) == 0
        yes_output = np.exp(outputs) * loc.astype(int) 
        kfold_yes_outputs.append(yes_output)
    kfold_yes_outputs = np.concatenate(kfold_yes_outputs)
    #print(kfold_yes_outputs.max())
    plt.figure()
    fig, axs = plt.subplots(2, 2, sharey=True, tight_layout=True)
    for character in range(4):
        x = int(character/2)
        y = int(character%2)
        #print(x,"**",y)
        axs[x,y].hist(kfold_yes_outputs[:,character][np.where(kfold_yes_outputs[:,character]!=0)[0]], range=(0,1))
    plt.suptitle(p+"_testset_distribution.png")
    plt.savefig(os.path.join(res, p+"_testset_distribution.png"))