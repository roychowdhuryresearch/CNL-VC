import os
import pickle

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        print(fn)
    return lookup
project_dir = '/data/Projects/Movie/Movie_Analysis-Repo'
model_type = 'LSTM'


preprocess_filename = '/data/Projects/Movie/Movie_Analysis_Repo/final_result_outputs/LSTM/figure3_preprocess.pkl'
output_folder = '/data/Projects/Movie/Movie_Analysis_Repo/final_result_outputs/LSTM'

''' example knockout results'''
patient = '445'
figure3a(preprocess_filename, patient, output_folder)

'''
ALL patient region knockout test
'''
figure3b(preprocess_filename, output_folder)

'''
patient mircowire knockout test
'''
patient = '445'
figure3c(preprocess_filename, patient, output_folder)


'''
patient sum_mircowire - region
'''
figure3e(preprocess_filename, patient, output_folder)

'''
ALL patient sum_mircowire - region
'''
figure3f(preprocess_filename, output_folder)
