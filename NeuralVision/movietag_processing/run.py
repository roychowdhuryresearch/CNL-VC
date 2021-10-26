from EpisodeStats import EpisodeStates
import pickle
path = "/media/yipeng/toshiba/movie/Movie_Analysis/movietag_processing/result/compare_movie_annotation_allresearchers_v2_40m_act_24_S06E01_30fps_ft1.mat"
#path = "/media/yipeng/toshiba/movie/Movie_Analysis/movietag_processing/cv_label_updated.mat"
frame_fir = "/home/yipeng/Desktop/video_process/frames/"
cvlabel_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/Character_TimeStamp_resnet"
episodeStates = EpisodeStates(path, 1, cvlabel_dir)
#episodeStates.print_scene()
def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)
'''
res = episodeStates.character_conditional_prob(mode="sum_first")
res_fn = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD/character_conditional_prob.pkl"
dump_pickle(res_fn,res)

for i in range(4):
    for j in range(4):
        key = str(i) + "|" + str(j)
        print(key, round(res[key], 3))
'''
res2 = episodeStates.character_association()
print(res2)
#res_fn = "/media/yipeng/toshiba/movie/Movie_Analysis/CNN_result_zero/CNN_multi_2_KLD/character_association.pkl"
#dump_pickle(res_fn,res2)

for i in range(4):
    for j in range(4):
        key = str(i) + "|" + str(j)
        print(key, round(res2[key], 3))
def parse_key(char, cond_char):
    return str(char) + "|"+ str(cond_char)

def one_step(prob):
    res = {}
    for char in range(4):
        for cond_char in range(4):
            ori_key = parse_key(char, cond_char)
            if char == cond_char:
                res[ori_key] = prob[ori_key]  
                continue  
            #print("we car calcuating, ", parse_key(char,cond_char))
            walk_prob = 0.0
            for walk_char in range(4):      
                if walk_char == char:
                    continue
                walk_cond = parse_key(walk_char,cond_char)
                char_walk = parse_key(char, walk_char)
                #print(char_walk,prob[char_walk], walk_cond,prob[walk_cond])
                walk_prob = walk_prob + 1.0*prob[walk_cond] * prob[char_walk]
                #print(walk_prob)
            final_key = parse_key(char,cond_char)
            res[final_key] = walk_prob 
    print(res)

one_step(res2)
'''
#episodeStates.plot_scene(frame_fir, "/media/yipeng/toshiba/movie/Movie_Analysis/movietag_processing/result")
'''