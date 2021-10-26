import os
import pickle
def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

result_dir = "/media/yipeng/Sandisk/movie_tracking/tracking/transferlearning/resnet_result_all"
save_dir = "/media/yipeng/Sandisk/Movie_Analysis/Character_TimeStamp_resnet"
classes = os.listdir(result_dir)
for c in classes:
    one_character_set = set()
    frames = os.listdir(os.path.join(result_dir, c))
    for fn in frames:
        frame_number = int(fn.strip().split("_")[1])
        one_character_set.add(frame_number)
    saved_dir = os.path.join(save_dir, c + ".pkl")
    dump_pickle(saved_dir, one_character_set)

