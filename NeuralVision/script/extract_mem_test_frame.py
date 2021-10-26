import sys
sys.path.append(".")
sys.path.append("./neural_correlation")
from data import patientNums, path_to_matlab_generated_movie_data, num_characters
import os 
import os 
from neural_correlation.utilities import *
import scipy as sp
import scipy.io as sio
import shutil

ep1_folder = "/home/yipeng/Desktop/video_process/frames/"
ep2_folder = "/media/yipeng/Sandisk/movie_tracking/total_frames_samped_4"
def find_closed_4_factor(num):
    num_4 = round(num/4)
    num = num_4*4
    return num

def move_frame(current_frame, label, ep_folder, save_dir):
    from_fn = os.path.join(ep_folder, "frame_" + str(current_frame) + ".jpg")
    to_fn = os.path.join(save_dir, "frame_" + str(current_frame)+"_" +str(label) + ".jpg")
    #print(from_fn, to_fn)
    shutil.copyfile(from_fn, to_fn)

def extract_frame(exp_file, save_dir):
    for i in range(exp_file.shape[0]):
        one_line = exp_file[i,:]
        label = one_line[1]
        movie_frame_start = one_line[5] * 30
        movie_frame_end = one_line[6] * 30 
        #print(movie_frame_start)
        movie_frame_start = int(find_closed_4_factor(movie_frame_start))
        movie_frame_end = int(find_closed_4_factor(movie_frame_end))
        #print(movie_frame_start, movie_frame_end)
        for sample_stp in range(movie_frame_start, movie_frame_end, 4):
            current_frame = sample_stp
            if label == 1: ### first movie
                move_frame(current_frame,label ,ep1_folder, save_dir)
            else:
                move_frame(current_frame,label ,ep2_folder, save_dir)
   


def main():
    save_folder = "/media/yipeng/toshiba/movie/Movie_Analysis/memtest_vis"
    clean_folder(save_folder)
    for patient in patientNums:
        patient_folder = os.path.join(path_to_matlab_generated_movie_data, patient)
        file_list = os.listdir(patient_folder)
        #ts_offeset = sio.loadmat("/media/yipeng/toshiba/movie/Movie_Analysis/data/442/units.mat")["units"]
        save_dir = os.path.join(save_folder, patient)
        print(save_dir)
        clean_folder(save_dir)
        for folder in file_list:
            if "memtest1" in folder:
                men_test_folder = os.path.join(patient_folder, folder)
                exp_fn = "psychophysics/" + "EXP_" +  patient+ "memtest1.mat"
                exp_file = sio.loadmat(os.path.join(men_test_folder, exp_fn))['TRIAL_ID']
                extract_frame(exp_file, save_dir)
                #print("trigger clip, start,ent ttl", triggle_file_start, triggle_file_end)

        
if __name__ == '__main__':

    main()
