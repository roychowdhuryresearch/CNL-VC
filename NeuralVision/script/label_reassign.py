## this file is to reassign the grouped movie analysis result
import sys
sys.path.append(".")
from neural_correlation.utilities import * 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
from data import path_to_results
frame_filename = "result/final_result_per_frame.pkl" 
neural_signal_fn = "Character_TimeStamp/Neuron_dictionary.pkl"
#os.mkdir(workdir)
#Jack
group_list1 =  [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480, 1821,544 ] 
#terro
group_list2 =  [954, 2027, 2151,765,6482,4175,3104,3070,1433, 1610, 2996,2148,1985,1075,1037,961,931,929]
#chole
group_list3 =  [193,204,212,216,373,383,395,414,428,435,841, 854,1273, 1281, 1283, 1289, 130, 1319,1927, 1938, 2109, 1032, 2366, 2404, 3186, 3438, 3667, 858, 834 ,4071,4515,4013,3007,2597,1891,1814,1805,1307,1294,1270,1263,793,741, 443, 404, 392, 389, 206] 
#bill
group_list4 =  [3567,266,270,334, 995, 1004, 1362, 1413, 1992, 2076, 2092, 2424, 2458, 2496, 2690, 331,343,5854,485,558,869,873,889,1616,4612,2497,1922,1023,894,562, 1964, 545, 541, 297]
group_l = [set(group_list1), set(group_list2), set(group_list3), set(group_list4)]
group_reassigned = [sorted(group_list1)[0], sorted(group_list2)[0], sorted(group_list3)[0], sorted(group_list4)[0]]
print(group_reassigned)
filtered_frame_trk = load_pickle(frame_filename)

for k in filtered_frame_trk.keys():
    tracker = filtered_frame_trk[k]
    if len(tracker) == 0:
        continue 
    for t in tracker:
        for index_g in range(len(group_l)): 
            if int(t[-1]) in group_l[index_g]:
                #print(group_reassigned[index_g])
                t[-1] = group_reassigned[index_g]

dump_pickle( "result/final_result_per_frame_reassigned.pkl", filtered_frame_trk)

