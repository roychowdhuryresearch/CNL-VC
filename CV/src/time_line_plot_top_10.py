from utilities import * 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
frame_filename = "./final_result_per_frame.pkl" 
neural_signal_fn = "Neuron_dictionary.pkl"
workdir = "./signal_vis_top_10/"
#clean_folder(workdir)
#bill
group_list1 =  [3567,266,270,334, 995, 1004, 1362, 1413, 1992, 2076, 2092, 2424, 2458, 2496, 2690, 331,343,5854,485,558,869,873,889,1616,4612,2497,1922,1023,894,562, 1964, 545, 541, 297]
group_list3 = [8994, 132,520, 861, 886, 2284, 3334, 527]#yellow lady
# chloe
group_list4 =  [193,204,212,216,373,383,395,414,428,435,841, 854,1273, 1281, 1283, 1289, 130, 1319,1927, 1938, 2109, 1032, 2366, 2404, 3186, 3438, 3667, 858, 834 ,4071,4515,4013,3007,2597,1891,1814,1805,1307,1294,1270,1263,793,741, 443, 404, 392, 389, 206] 
#torrest 
group_list5 =  [954, 2027, 2151,765,6482,4175,3104,3070,1433, 1610, 2996,2148,1985,1075,1037,961,931,929]
# ladyM
#group_list =  [390, 696, 781,785, 789, 792, 797, 814, 817, 832, 840, 1002, 1025, 1309, 1327, 1512, 1777, 1857, 1859, 1892, 1894, 2100, 5090, 7178 2424, 2458, 2496, 2690] 

#group_list = [277,278,315,351,354,711,328,315] # black guy with senator
# Jack 
group_list6 = [744, 936, 1048, 1058, 1973, 2008, 3048, 3101, 560, 1635, 3971, 1617, 4294, 5867, 6931, 908 ,896,901,7406,352,5252,4790,3782,3710, 970, 916,958,3191,2141,2013,1998,1556,1448,1408,992,950,930,546, 543, 480, 1821,544 ] 
#group_list = [501, 1200, 1557,1571] # president
group_list = [group_list1] 
group_dict = {}
for g in group_list:
    group_dict[g[0]] = set(g)

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

group_time_line_result = {}
sorted_frame_name = sort_filename(list(filtered_frame.keys()))
for g_key in group_dict.keys():
    group_set = group_dict[g_key]
    character_appear_seconds = set()
    for frame_name in sorted_frame_name :
        group_numbers = filtered_frame[frame_name]
        frame_number = int(frame_name.split("rame_")[1].split(".")[0])
        frame_second = int(frame_number/30)
        for g in group_numbers:
            if g in group_set:
                if frame_second in character_appear_seconds:
                    continue
                character_appear_seconds.add(frame_second)
                character_appear_seconds.add(frame_second+1)
                character_appear_seconds.add(frame_second-1)
    group_time_line_result[g_key] = character_appear_seconds 

result = np.zeros((len(neural_signals.keys()),len(list(sorted(group_time_line_result.keys())))))
group_nums = sorted(list(group_time_line_result.keys()))
for g_idx in range(len(group_nums)):
    k = group_nums[g_idx]
    character_appear_seconds = group_time_line_result[k]
    sorted_time = np.array(sorted(list(character_appear_seconds)))
    for neural_index in sorted(neural_signals.keys()):
        time_line_result = neural_signals[neural_index][0:2520]
        one_channel = time_line_result[sorted_time]
        result[neural_index-1, g_idx] = np.mean(one_channel.flatten())

most_active_each_group = {}
for g_idx in range(result.shape[1]):
    arr = result[:,g_idx]
    neural_index = arr.argsort()[-10:][::-1]
    print(arr[neural_index])
    most_active_each_group[group_nums[g_idx]] = neural_index

base = 20
#print(most_active_each_group)
for g_idx in range(len(group_nums)):
    #print(g_idx)
    k = group_nums[g_idx]
    character_appear_seconds = group_time_line_result[k]
    sorted_time = np.array(sorted(list(character_appear_seconds)))
    save_dir = workdir + str(k) + "/"
    os.mkdir(save_dir)
    for neural_index in most_active_each_group[group_nums[g_idx]].tolist():
        print(neural_index)
        plt.close("all")
        plt.figure(figsize=(40,4))
        time_line_result = neural_signals[neural_index+1][0:2520]
        padded_sorted_time = time_line_result * 0
        padded_sorted_time[sorted_time] = base
        plt.plot(padded_sorted_time,label=k) 
        plt.plot(time_line_result,label='firing rate')
        plt.xlabel('time line')
        plt.ylabel('neural firing rate')
        plt.title("No. "+str(neural_index) + " firing rate")
        plt.xticks(np.arange(0, len(time_line_result), step=50))
        plt.ylim(0, 25)
        plt.legend(title='legend')
        plt.savefig(save_dir + str(neural_index) + ".jpg")




'''
for neural_index in neural_signal_result.keys():
    time_line_result = neural_signal_result[neural_index]
    if sum(time_line_result) == 0:
        continue
    plt.close("all")
    ind = range(len(character_appear_seconds))
    sorted_time = sorted(list(character_appear_seconds))
    #print(character_appear_seconds)
    plt.bar(ind, time_line_result)
    plt.xlabel('character appears in seconds')
    plt.ylabel('neural firing rate')
    plt.title("No. "+str(neural_index) + " fires in " + str(len(sorted_time))+ " second")
    plt.xticks([0,len(character_appear_seconds)],[sorted_time[0],sorted_time[-1]])
    plt.ylim(0, 100)
    plt.savefig(save_dir + str(neural_index) + ".jpg")
'''




