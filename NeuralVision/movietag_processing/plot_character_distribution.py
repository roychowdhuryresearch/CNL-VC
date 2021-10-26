from EpisodeStats import EpisodeStates
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np 
path = path = "/media/yipeng/data/movie/Movie_Analysis/movietag_processing/result/compare_movie_annotation_allresearchers_v2_40m_act_24_S06E01_30fps_ft1_original.mat"
frame_fir = "/home/yipeng/Desktop/video_process/frames/"
cvlabel_dir = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_resnet"
episodeStates = EpisodeStates(path, 0, cvlabel_dir)
res_cv, res_annotation = episodeStates.get_character_distribution()

def plot_tpfp(res_cv, res_annotation,direct = None, fn= 'fp_tp.jpg'):
    plt.figure(figsize=(12, 10))
    for character_index in range(len(res_cv)):
        cv_label = res_cv[character_index]
        annotation = res_annotation[character_index]
        plt.subplot(len(res_cv),1,character_index + 1)
        plt.scatter(range(len(cv_label)), cv_label, s=5, c='tab:blue', label="cv",alpha=1)
        plt.scatter(range(len(cv_label)), annotation*2, s=5, c='tab:orange',label="annotation",alpha=1)
        plt.title("person index: " + str(character_index) +  " distribution")
        plt.legend()
        plt.ylim(0.5, 2.5)
        if character_index is not 3:
            plt.xticks([], [])
    plt.xlabel('Time (s)')
    plt.show()
    #plt.savefig(os.path.join(direct, fn))
    #plt.close('all')

plot_tpfp(res_cv, res_annotation) 