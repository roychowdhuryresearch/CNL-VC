import os
import numpy as np
from scipy.io import loadmat, savemat
import pickle
def dump_pickle(saved_fn, variable):
    with open(saved_fn, 'wb') as ff: 
        pickle.dump(variable, ff)

path_to_matlab_generated_movie_data = "/media/yipeng/data/movie_2021/Movie_Analysis/data"
patients = ['431', '433', '435', '436', '439', '441', '444', '445', '452']
output_dir = "/media/yipeng/data/movie_2021/Movie_Analysis/final_result_inputs"
for patient in patients:
    output_folder = os.path.join(output_dir,patient)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    region_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"features_mat_regions_clean.npy")
    region = np.load(region_fn, allow_pickle=True)
    reg = []
    for d in region:
        reg.append(d[0])
    reg_np = np.array(reg)

    microarr_fn = os.path.join(path_to_matlab_generated_movie_data,patient,"channel_data.mat")
    microarr_temp = loadmat(microarr_fn)["channel_reg_info"][0]
    microarr = []
    for m in microarr_temp:
        microarr.append(m[0][0][0])
    microarr_np = np.array(microarr)
    dump_pickle(os.path.join(output_folder,"regions.pkl"),reg_np)
    dump_pickle(os.path.join(output_folder,"microwire.pkl"),microarr_np)