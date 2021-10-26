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
#bench_mark = episodeStates.benchmark_cv_labels()
#print(bench_mark)
character_index = 3 
#episodeStates.get_all_fn_character(character_index)
#episodeStates.get_all_frame_fp(character_index)

bench_mark = episodeStates.benchmark_cv_labels()

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
all_confusion = np.zeros((2,2)).astype(int)
y_tick = ["yes_p", "no_p"]
x_tick = ["yes", "no"]
for character_index in range(len(bench_mark)):
    one_person_confusion = bench_mark[character_index]
    k = one_person_confusion
    all_confusion = all_confusion+one_person_confusion
    ax = sns.heatmap(one_person_confusion.astype(int),annot=True, fmt="d", ax=axs[int(character_index/3), character_index%3])
    ax.set_yticks(np.arange(2),y_tick)
    ax.set_xticks(np.arange(2),x_tick)
    ax.set_yticklabels(y_tick)
    ax.set_xticklabels(x_tick)
    ax.set(title = str(character_index) + " accuracy " + str(1.0*(k[0,0] + k[1,1])/k.sum()))
ax = sns.heatmap(all_confusion.astype(int), ax=axs[1,1],annot=True, fmt="d")
k = all_confusion
ax.set(title = 'all'+ " accuracy " + str(1.0*(k[0,0] + k[1,1])/k.sum()))
ax.set_yticks(np.arange(2),y_tick)
ax.set_xticks(np.arange(2),x_tick)
ax.set_yticklabels(y_tick)
ax.set_xticklabels(x_tick)
plt.show()
