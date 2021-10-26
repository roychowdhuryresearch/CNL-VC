import numpy as np
import os
import sys
import matplotlib.pylab as plt
import pickle
import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False
font = {'family' : 'normal',
        'size'   : 14}
matplotlib.rc('font', **font)

import matplotlib.pylab as plt

def load_pickle(fn):
    if not os.path.exists(fn):
        print(fn , " notexist")
        return
    with open(fn, "rb") as f:
        lookup = pickle.load(f)
        print(fn)
    return lookup


def gen_intervals(data_arr):
    #print(np.diff(data_arr))
    index_jump = np.where(np.diff(data_arr)!=1)[0]
    #print(index_jump)
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
    print(res)
    return res
all_intervals = []
char_occ_percent = []
num_character = 9
ch_fn_dir = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_resnet"
#for cnt, (character_info, num_frames) in enumerate(sorted_info_list[0:5]):
for cnt in range(num_character):#, (character_info, num_frames) in enumerate(sorted_info_list[0:5]):
    #print(cluster_dir)
    f_names = os.path.join(ch_fn_dir,  f"{cnt}.pkl")
    org_frame_name = load_pickle(f_names)
    #print("frame_name",type(org_frame_name))#, frame_name)
    frame_name = [int(frame_name_i)/4  for frame_name_i in org_frame_name]
    #frame_name = [int(item.split("_")[1])/4 for item in f_names]
    res_frame_name = filling_gaps(sorted(frame_name), 50)
    char_occ_percent.append(len(frame_name))
    all_intervals.append(res_frame_name)


fig, gnt = plt.subplots(constrained_layout=True) 
gnt.set_xlabel('Minutes in movie')
min_value_list = np.arange(0, 30*60*45/4, 30*60*5/4) 

plt.xticks(min_value_list, ([f'{item} min' for item in range(0, 45, 5)]))
plt.title("Distribution of the Characters’ Presence in Time during Movie")

gnt.grid(True)
mycmp = plt.get_cmap("tab10") 

print(matplotlib.__version__)
for i in range(len(all_intervals)):
    gnt.broken_barh(all_intervals[i], (10*i, 9), facecolors =mycmp(i)) 
gnt.set_ylim(-1, 10*num_character )
#gnt.set_ylabel('Character') 
gnt.set_yticks([5 + 10*i for i in range(num_character)]) 
gnt.set_yticklabels(['Jack Bauer','Bill Buchanan','Chloe O\'Brian','Abu Fayed', 'Karen Hayes', \
    'Wayne Palmer', 'Tom Lennox','Milo Pressman','Morris O\'Brian']) 
#gnt.set_yticklabels([str(i) for i in range(1, num_character + 1)]) 

#ax2.set_yticks([2 + 3*i for i in range(num_character)])
#ax2.set_yticks([5 + 10*i for i in range(num_character)])  
ax2 = gnt.twinx()
ax2.set_ylim(-1, 10*num_character)
ax2.set_ylabel('Coverage') 
ax2.set_yticks([5 + 10*i for i in range(num_character)],) 
#gnt.secondary_yaxis('right')
ax2.set_yticklabels([f"{100*char_occ_percent[i-1]/(30*60*45/4):.1f}%" for i in range(1,num_character + 1)])
fig.tight_layout() 
plt.show()
