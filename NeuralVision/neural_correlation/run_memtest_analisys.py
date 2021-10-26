import sys
import numpy as np
import os
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pylab as plt
sys.path.append(".")
sys.path.insert(0, "/media/yipeng/data/movie_2021/Movie_Analysis")
from utilities import *
from data import patientNums, path_to_matlab_generated_movie_data, num_characters
#result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/CNN_multi_2_KLD_final_pooling=1"
result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD"
result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/paper_results/LSTM_multi_2_KLD"
#result_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/CNN_result/LSTM_multi_2_KLD_reverse"
#result_dir = "/media/yipeng/data/movie/Movie_Analysis/CNN_result/LSTM_70-30"
from KfoldStats import KfoldStats
k = 5


def compute_p_value(kfoldStats):
    """
    The actual plotting function of 16 plots kfolds stats 
        
    Parameters
    ----------
    kfoldStats : KfoldStats object
    """
    response_clip_comparison = {} ## 3#3#2## for node location 
    #response_clip_comp_avg_p = {}
    for clip_mode in ["clip", "response"]:
        overall_stats = kfoldStats.average_over_kfold(mode="prob", mode2="normalize", clip_mode=clip_mode)
        ### 
        response_clip_comparison[clip_mode] = {key: np.sum([val >= 0.5])*1.0/len(val) for key, val in overall_stats.items() if key[0]!=key[2]}
    
    #for clip_mode in ["clip", "response"]:
    feature_list = {key: [val for condition_key, val in val.items()] for key, val in response_clip_comparison.items()}
    feature_list_sp1 = {key: [val for condition_key, val in val.items() if condition_key[-1]=="1"] for key, val in response_clip_comparison.items()}
    feature_list_sp2 = {key: [val for condition_key, val in val.items() if condition_key[-1]=="2"]  for key, val in response_clip_comparison.items()}
    ttest,pval = ttest_1samp(np.array(feature_list['clip'])-np.array(feature_list['response']), 0)
    ttest,pval_1 = ttest_1samp(np.array(feature_list_sp1['clip'])-np.array(feature_list_sp1['response']), 0)
    ttest,pval_2 = ttest_1samp(np.array(feature_list_sp2['clip'])-np.array(feature_list_sp2['response']), 0)
    print("p test value between clip and response group among various conditions", ttest)
    

    comb_mode_predictions = {}
    plt.figure()
    for key, val in response_clip_comparison['clip'].items():
        if key[-1] == "1":
            plt.plot([1, 4], [response_clip_comparison['clip'][key],response_clip_comparison['response'][key]], "ro-")
           
        else:
            plt.plot([1, 4], [response_clip_comparison['clip'][key],response_clip_comparison['response'][key]], "bo-")
        comb_mode_predictions[key] = np.array([response_clip_comparison['clip'][key],response_clip_comparison['response'][key]])

    plt.title(f"ovarall p {pval:.2f} between clip and response, red 1 {pval_1:.2f} blue 2 {pval_2:.2f}")
    plt.savefig(os.path.join(kfoldStats.result_dir, kfoldStats.patient, "kfold_pvalue_condition.jpg"))
    dump_pickle(os.path.join(kfoldStats.result_dir, kfoldStats.patient, "kfold_pvalue_condition.pkl"), comb_mode_predictions)
    
def plot_benchmark_vs_th(x):
    x = np.array(x)
    plt.figure()
    patient_names = ["431", "435", "436", "441", "442"]
    for i in range(len(x)):
        plt.plot(x[i,:,0], x[i,:,1],"o-" ,label = patient_names[i])
    plt.legend()
    plt.title("Jack percentage of duration vs recall")
    plt.show()



def main():
    stats = []
    for patient in patientNums:
    #for patient in ["431"]:
        if patient == "439":
            continue
        kfoldStats = KfoldStats(patient, result_dir, 5)
        #kfoldStats.plot_correlation_mat2()
        #print(kfoldStats.stats_list)
        #kfoldStats.memtest_benchmark()
        #kfoldStats.memtest_vis()
        if len(kfoldStats.stats_list) > 0:
            print(patient)
            kfoldStats.memtest_benchmark()
            #kfoldStats.memtest_benchmark_search()
            #print(kfoldStats.memtest_benchmark())
            #stats.append(kfoldStats.memtest_benchmark_search())
            #kfoldStats.memtest_vis()
            #kfoldStats.memtest_vis_all()
            #kfoldStats.draft_function_get_yes_distribution()
            #kfoldStats.avg_prob_on_response_edge()
            #kfoldStats.print_prediction_percentage()
            #response_clip_comparison = {} ## 3#3#2:
            #KfoldStats.memtest_benchmark()
            #for clip_mode in ["clip", "all", "response"]:
            #    kfoldStats.average_over_kfold(mode="cv_label", clip_mode=clip_mode)
            #    kfoldStats.average_over_kfold(mode="prob", clip_mode=clip_mode)
            #print(kfoldStats.exclusive_coactivation())
            #print("+_++++_)_+_+_+_+_+_")
            #print(kfoldStats.exclusive_coactivation_count())
            #compute_p_value(kfoldStats)
                #print(overall_stats)
    #print("final_recall is ", np.mean(recall))    
    #plot_benchmark_vs_th(stats)
    
if __name__ == '__main__':
    main()
