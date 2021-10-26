from utilities import * 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import shutil
import cv2
import numpy as np
import glob

frame_filename = "./final_result_per_frame.pkl" 
workdir = "./movie_segment_frames/"
all_frame_dir = "/home/yipeng/Desktop/video_process/frames/"
#clean_folder(workdir)
#bill
group_list1 = [3567,266,270,334, 995, 1004, 1362, 1413, 1992, 2076, 2092, 2424, 2458, 2496, 2690, 331,343,5854,485,558,869,873,889,1616,4612,2497,1922,1023,894,562, 1964, 545, 541, 297]
#modification by Yiming, added a new group
group_list2 = [193,204,212,216,373,383,395,414,428,435,841,854,1273,1281,1283,1289,130,1319,1927,1938,2109,1032,2366,2404,3186,3438,3667,858,834,4071,4515,4013,3007,2597,1891,1814,1805,
1307,1294,1270,1263,793,741,443,404,392,389,206]
#end of modification
group_list5 =  [954, 2027, 2151,765,6482,4175,3104,3070,1433, 1610, 2996,2148,1985,1075,1037,961,931,929]
group_list6 = [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480, 1821,544 ] 
group_list = group_list2
group_set = set(group_list)

#modification by Yiming, added folders for each group
workdir = workdir + str("group2")+"/" 
#end of modification

filtered_frame_trk = load_pickle(frame_filename)
apperance_fn = set()
for k in filtered_frame_trk.keys():
    tracker = filtered_frame_trk[k]
    if len(tracker) == 0:
        continue 
    for t in tracker:
        if int(t[-1]) in group_set:
            original_fn = k[0:-4]
            apperance_fn.add(original_fn)

for a in apperance_fn:
    shutil.copy(all_frame_dir+a, workdir+a)

img_array = []
for filename in sort_filename(apperance_fn):
    img = cv2.imread(workdir + filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter(str(group_list[0])+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), 29.9, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
