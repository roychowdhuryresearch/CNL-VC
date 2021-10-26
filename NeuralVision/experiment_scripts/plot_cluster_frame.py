import numpy as np
import os
import sys
import matplotlib.pylab as plt

clusters_dir = "/media/yipeng/data/movie/Movie_Analysis/train_data_clean/test"
cluster_info_list = []
#number_frame_list = []
for folder in os.listdir(clusters_dir):
    frame_dir = os.path.join(clusters_dir, folder)
    cluster_info_list.append((frame_dir, len(os.listdir(frame_dir))))
    #print()
sorted_info_list = sorted(cluster_info_list, key=lambda x: x[1])[::-1]
print(sorted_info_list)
# #colormap = plt.cm.get_cmap('viridis', len(sorted_info_list))
# colormap = plt.cm.get_cmap('viridis', 5)
# plt.figure(figsize=(20, 5))
# for cnt, (cluster_dir, num_frames) in enumerate(sorted_info_list[:5]):
#     f_names = os.listdir(cluster_dir)
#     frame_name = [int(item.split("_")[1])/4 for item in f_names]
#     plt.scatter(frame_name, [cnt*10]*num_frames, s=5, c=[colormap(cnt+1)]*num_frames)
# min_value_list = np.arange(0, 30*60*45/4, 30*60*5/4) 
# # 
# # 
# plt.xticks(min_value_list, ([f'{item} min' for item in range(0, 45, 5)]))
# plt.show()
#plt.savefig("cluster_frame_number1.jpg") 



'''
import matplotlib.pyplot as plt 
fig, gnt = plt.subplots() 
gnt.set_ylim(0, 50)
gnt.set_xlim(0, 160) 
gnt.set_xlabel('Minutes in movie') 
gnt.set_ylabel('Cluster') 
gnt.set_yticks([15, 25, 35]) 
gnt.set_yticklabels(['1', '2', '3']) 
gnt.grid(True) 
gnt.broken_barh([(10, 50), (100, 20), (130, 10)], (20, 9), facecolors =('tab:red')) 
gnt.broken_barh([(110, 10), (150, 10)], (10, 9), facecolors ='tab:blue') 
gnt.broken_barh([(40, 50)], (30, 9), facecolors =('tab:orange')) 
plt.show()
'''

def gen_intervals(data_arr):
    #print(np.diff(data_arr))
    index_jump = np.where(np.diff(data_arr)!=1)[0]
    if len(index_jump) == 0:
        return [[data_arr[0], data_arr[1]]]
    #print(index_jump, data_arr[index_jump],data_arr[index_jump+1], data_arr[index_jump-1])
    step0 = data_arr[0]
    #steps = np.array([step0] + [data_arr[item] for item in index_jump]+ [data_arr[-1]]).reshape(-1,2)
    steps = [[data_arr[0], data_arr[index_jump[0]]]]
    for i in range(len(index_jump)-1):
        steps.append([data_arr[index_jump[i]+1], data_arr[index_jump[i+1]]]) 
    steps.append([data_arr[index_jump[-1]+1], data_arr[-1]])
    return steps

def filling_gaps(data, resolution):
    intervals = gen_intervals(data)
    if len(intervals) < 1:
        return np.array(intervals)
    index = 1
    res = []
    res.append(intervals[0])
    while index < len(intervals):
        i = intervals[index]
        if i[0] - res[-1][1] > resolution:
            res[-1][1] = res[-1][1]-res[-1][0]
            res.append(i)
        else:
            res[-1][1] = i[-1]
        index += 1
    res[-1][1] = res[-1][1]-res[-1][0]
    return res


all_intervals = []
for cnt, (cluster_dir, num_frames) in enumerate(sorted_info_list):
    #print(cluster_dir)
    f_names = os.listdir(cluster_dir)
    frame_name = [int(item.split("_")[1])/4 for item in f_names]
    res_frame_name = filling_gaps(sorted(frame_name), 50)
    all_intervals.append(res_frame_name)

import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
font = {'family' : 'normal',
        'size'   : 25}
matplotlib.rc('font', **font)

fig, gnt = plt.subplots(figsize=(15,12)) 
gnt.set_xlabel('Minutes in movie')
min_value_list = np.arange(0, 30*60*45/4, 30*60*5/4) 
plt.xticks(min_value_list, ([f'{item} min' for item in range(0, 45, 5)]))
gnt.set_ylabel('Cluster') 
num_cluster = 6 
kk = np.array(list(range(num_cluster)))*10 + 5
last5 = []
#print(all_intervals)
for a in all_intervals[5:]:
    for aa in a:
        print(aa)
        last5.append(aa)
res_interval = all_intervals[0:5]
res_interval.append(last5)
print("res_interval")
print(res_interval)

print("kk end")
gnt.set_yticks(kk)
y_labels = []
#for k in kk:
#    y_labels.append(str(k/5))
y_labels = [str(item+1) for item in list(range(len(all_intervals)))]
print(y_labels)
gnt.set_yticklabels( y_labels)
gnt.grid(True)
mycmp = plt.get_cmap("tab10") 
for i in range(num_cluster):
    gnt.broken_barh(res_interval[i], (10*i, 9), facecolors =mycmp(i)) 
#plt.show()
plt.savefig("./jack_cluster.jpg")