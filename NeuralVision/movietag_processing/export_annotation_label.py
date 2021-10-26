from EpisodeStats import EpisodeStates
import pickle
path = "/media/yipeng/toshiba/movie/Movie_Analysis/movietag_processing/result/compare_movie_annotation_allresearchers_v2_40m_act_24_S06E02_30fps_ft1.mat"
#path = "/media/yipeng/data/movie/Movie_Analysis/movietag_processing/result/compare_movie_annotation_allresearchers_v2_40m_act_24_S06E01_30fps_ft1_original.mat"
#path = "/media/yipeng/data/movie/Movie_Analysis/movietag_processing/cv_label_updated.mat"
frame_fir = "/home/yipeng/Desktop/video_process/frames/"
cvlabel_dir = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_resnet"
episodeStates = EpisodeStates(path, 1, cvlabel_dir)
#episodeStates.print_scene()
def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)
res = episodeStates.export_annotation_label()
for i in res.keys():
    dump_pickle("/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_annotation_2/"+str(i)+".pkl", res[i])
