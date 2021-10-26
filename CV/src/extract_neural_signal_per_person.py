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

## construct_signal_result:
for i in range(1,139):
    neural_signal_result[i] = []

filtered_frame_trk = load_pickle(frame_filename)
filtered_frame = {}
for k in filtered_frame_trk.keys():
    tracker = filtered_frame_trk[k]
    filtered_frame[k] = []
    if len(tracker) == 0:
        continue 
    for t in tracker:
        filtered_frame[k].append(int(t[-1]))


sorted_frame_name = sort_filename(list(filtered_frame.keys()))
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
            for neural_index in neural_signal_result.keys():
                neural_signal_result[neural_index].append(neural_signals[neural_index][frame_second])
                
save_dir = workdir + str(group_list[0]) + "/"
clean_folder(save_dir)
for neural_index in neural_signal_result.keys():
    time_line_result = neural_signal_result[neural_index]
    if sum(time_line_result) == 0:
        continue
    plt.close("all")
    ind = range(len(character_appear_seconds))
    sorted_time = sorted(list(character_appear_seconds))
    plt.figure(figsize=(50,10))
    #print(character_appear_seconds)
    plt.bar(ind, time_line_result)
    plt.xlabel('character appears in seconds')
    plt.ylabel('neural firing rate')
    plt.title("No. "+str(neural_index) + " fires in " + str(len(sorted_time))+ " second")
    plt.xticks([0,len(character_appear_seconds)],[sorted_time[0],sorted_time[-1]])
    plt.ylim(0, 100)
    plt.savefig(save_dir + str(neural_index) + ".jpg")





