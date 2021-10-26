from utilities import * 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
frame_filename = "./final_result_per_frame.pkl" 
neural_signal_fn = "Neuron_dictionary.pkl"
workdir = "./time_stamp/"
#os.mkdir(workdir)
group_list6 = [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480, 1821,544 ] 
group_list5 =  [954, 2027, 2151,765,6482,4175,3104,3070,1433, 1610, 2996,2148,1985,1075,1037,961,931,929]
group_list4 =  [193,204,212,216,373,383,395,414,428,435,841, 854,1273, 1281, 1283, 1289, 130, 1319,1927, 1938, 2109, 1032, 2366, 2404, 3186, 3438, 3667, 858, 834 ,4071,4515,4013,3007,2597,1891,1814,1805,1307,1294,1270,1263,793,741, 443, 404, 392, 389, 206] 
group_list1 =  [3567,266,270,334, 995, 1004, 1362, 1413, 1992, 2076, 2092, 2424, 2458, 2496, 2690, 331,343,5854,485,558,869,873,889,1616,4612,2497,1922,1023,894,562, 1964, 545, 541, 297]
group_list3 = [8994, 132,520, 861, 886, 2284, 3334, 527]#yellow lady
# president
group_list10 = [501, 1200, 1557,1571,1035]
# fat guy
group_list7 = [2209, 46,70,153,503,524,495,494, 506,1156, 1573, 518,144,  499, 143]
# new guy
group_list8 = [3169,8786,5555,3404,3401,2070, 468, 787, 796, 799, 1533, 1535,1250,1247,457,1021,1008]
# mossi
group_list9 = [725,4753,6417,5838,5340,3684,2612,2358,2110,1280,1039,835,736,726,187, 449,218, 171,215, 199, 211, 173]
group_list = group_list10
group_set = set(group_list) 


neural_signals= load_pickle(neural_signal_fn)
filtered_frame_trk = load_pickle(frame_filename)
filtered_frame = {}
for k in filtered_frame_trk.keys():
    tracker = filtered_frame_trk[k]
    filtered_frame[k] = []
    if len(tracker) == 0:
        continue 
    for t in tracker:
        filtered_frame[k].append(int(t[-1]))

result = set()
for k in filtered_frame.keys():
    tracker = filtered_frame[k]
    if len(tracker) == 0:
        continue 
    for group_num in filtered_frame[k]:
        if group_num in group_set:
            time = int(k.split(".")[0].split("_")[1])
            result.add(time)

dump_pickle(workdir + str(group_list[0]) + ".pkl", sorted(list(result)))
