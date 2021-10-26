from utilities import * 
import matplotlib
import matplotlib.pyplot as plt
frame_filename = "./final_result_per_frame.pkl" 
neural_signal_fn = "Neuron_dictionary.pkl"
workdir = "./signal_vis/"
# th 1.18 # 30 
#group_list = [716,272,276,340,352,1431,1080,1036,705] #senator
#group_list = [442,47,71,104,138,1202,882,793,1897,1509,793,1556,1168,871,782,443,435,430,428,154,131] #fat guy
group_list = [156,859,1193,857] #yellow lady
#group_list = [261,242,2387,612,611,600,566,263,199] # blue bad mood lady
#group_list = [602] # blue straight hair lady
#group_list = [277,278,315,351,354,711,328,315] # black guy with senator
#group_list = [296,348,358,362,1801,1810,2143,1417,1084,717,700,673,362,663,659,358] # CIA
#group_list = [1565,1213,519,500,487,102] # president
group_set = set(group_list)

neural_signals= load_pickle(neural_signal_fn)
neural_signal_result = {}
