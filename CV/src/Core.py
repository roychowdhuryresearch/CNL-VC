import os
from Record import Record
from Sort import Sort
import random
from PIL import Image , ImageDraw, ImageColor
from FaceGrouping import FaceGrouping
from utilities import *
import dlib
from Network import Network
import numpy as np
from project_setting import result_dir, stats_dir, filter_tracking_result_dir, knn_result_dir, group_result_network_dir

class Core:
    def __init__(self, file_dir, image_dir, size_threshold):
        self.file_dir = file_dir
        self.image_dir = image_dir
        self.record_dict:{str, Record} = {}
        self.sort_fn:[] = []
        self.tracker = Sort(use_dlib= False)
        self.size_threshold = size_threshold
        self.process()
        self.tracking_result = {}
        self.filtered_tracking_result = {}
        self.number_of_object = 0
        self.scene_switch_label_mapping = set()

        self.faceGrouping = None
        self.group_matching = {}
        self.only_has_face = set() 
        
        self.label_map = []
        self.label_graph = None
        self.output_label_map = {}
        self.final_tracking_result = {}

    def process(self):
        self.number_of_object = 0
        file_list = os.listdir(self.file_dir)
        self.sorted_fn = sort_filename(os.listdir(self.image_dir))
        for fn in file_list:
            self.record_dict[fn] = Record.create_record(self.file_dir + fn, self.size_threshold)
    
    def tracking(self):
        self.tracker = Sort(use_dlib=False)
        self.tracking_result = {}
        self.number_of_object = 0 
        for image_fn in self.sorted_fn:
            fn = image_fn+".txt"
            #print(image_fn)
            #if 
            #img = dlib.load_rgb_image(self.image_dir + image_fn)
            if fn in self.record_dict and len(self.record_dict[fn].good_rect_list) > 0:
                detections = self.record_dict[fn].generate_detections()
                #print(detections)
                #trackers:np.array = self.tracker.update(detections, img)
                trackers:np.array = self.tracker.update(detections)
                #print(trackers)
                self.number_of_object = trackers[:,-1].max()
                self.tracking_result[fn] = trackers
            else:
                detections = []
                #trackers:np.array = self.tracker.update(detections,img)
                trackers:np.array = self.tracker.update(detections)
                self.tracking_result[fn] = []
    
    def filter_tracking_result(self,scene_list_fn,save_img = False):
        scene_dict:{} = read_scene_list(os.path.join(stats_dir,"scene_list.txt"))
        save_dir = filter_tracking_result_dir
        if save_img:
            clean_folder(save_dir)
        label_count = {}
        for k in scene_dict.keys():
            start = k
            end = scene_dict[k]
            local_set = {}
            for i in range(start, end):
                txt_fn = "frame_"+str(i)+".jpg.txt"
                if txt_fn in self.tracking_result:
                    trackers  = self.tracking_result[txt_fn].copy()
                    if isinstance(trackers, list):
                        continue
                    image_fn = txt_fn.split(".")[0]
                    if save_img:
                        image = Image.open(join(self.image_dir,image_fn + ".jpg"))
                    for index in range(len(trackers[:,-1].tolist())):
                        l = trackers[index, -1] 
                        l = int(l)     
                        if l in label_count and l not in local_set: ## in global bot not in local
                            local_set[l] = int(label_count[l] + self.number_of_object)
                            count = str(local_set[l])
                            folder_name = join(save_dir, count)
                            if save_img:
                                os.mkdir(folder_name) 
                        elif l in label_count and l in local_set:
                            count = str(local_set[l])
                            folder_name = join(save_dir, count)
                        elif l not in label_count and l in local_set:
                            count = str(local_set[l])
                            folder_name =  join(save_dir, count) 
                        else:
                            local_set[l] = int(l)
                            count = str(local_set[l])
                            folder_name =  join(save_dir, count)
                            if save_img:
                                if not os.path.exists(folder_name):
                                    os.mkdir(folder_name)
                        trackers[index, -1] = int(count)
                        self.filtered_tracking_result[txt_fn] = trackers
                        t = trackers[index,:]
                        if save_img:
                            im_c = image.crop((t[0], t[1], t[2], t[3]))
                            im_c.save(join(folder_name,image_fn+"_"+count+".jpg"))
            label_count = self.update_label_count(label_count,local_set)
        self.find_switch_mapping(scene_dict)
        self.export_scene_switch_label_mapping(os.path.join(stats_dir, "scene_switch_label_mapping.txt"))

    def update_label_count(self, label_count ,local_set):
        for l in local_set.keys():
            label_count[l] = local_set[l]
        return label_count 
    
    def find_switch_mapping(self, scene_dict):
        max_frame = len(self.record_dict.keys())*4
        for k in scene_dict:
            end_point = int(scene_dict[k])
            i = end_point -1
            txt_fn = "frame_"+str(i)+".jpg.txt"
            while True:
                if txt_fn not in self.filtered_tracking_result and i >= k:
                    i = i - 1
                    txt_fn = "frame_"+str(i)+".jpg.txt"
                else:
                    break 
            if i < k:
                continue
            previous_frame = i 
            previous_tracker = self.filtered_tracking_result["frame_"+str(i)+".jpg.txt"]
            #print("previous: frame_"+str(i)+".jpg.txt" )
            i = end_point
            txt_fn = "frame_"+str(i)+".jpg.txt"
            while True:
                if txt_fn not in self.filtered_tracking_result and i <= max_frame:
                    i = i + 1
                    txt_fn = "frame_"+str(i)+'.jpg.txt'
                else:
                    break 
            if i > max_frame:
                continue
            end_frame = i 
            if end_frame - previous_frame > 16:
                continue
            #print("Later: frame_"+str(i)+".jpg.txt" )
            later_tracker = self.filtered_tracking_result["frame_"+str(i)+".jpg.txt"]
            #print(previous_tracker)
            #print(later_tracker)
            for j in previous_tracker[:,-1].tolist():
                #print(previous_tracker[: -1])
                for k in later_tracker[:, -1].tolist():
                    self.scene_switch_label_mapping.add(str(int(j)) + "#"+str(int(k)))
    
    def network_filtering(self, threshold = 4, distance_th=50 ):
        nw = Network()
        nw.construct(os.path.join(stats_dir, "group_matching.txt"))
        nw.filter_by_threshold(threshold)
        nw.filter_by_frame(self.filtered_tracking_result)
        print(self.number_of_object)
        nw.filter_by_distance(self.number_of_object,distance_th)
        nw.filter_by_bad_group(os.path.join(knn_result_dir,"bad_class_knn_stage2.pkl"))
        nw.filter_by_scene_switch(os.path.join(stats_dir,"scene_switch_label_mapping.txt"))
        self.label_graph= nw.filtered_graph
        self.label_map = nw.get_label_map()
        print(self.label_map)
        nw.export_group_result(filter_tracking_result_dir,group_result_network_dir)

    def result_reassign(self):
        count = 0
        face_set = set(np.loadtxt(os.path.join(stats_dir,"face.txt")))
        for fn in self.filtered_tracking_result:
            trackers = self.filtered_tracking_result[fn]
            self.final_tracking_result[fn] = []
            if len(trackers) == 0:      
                continue
            for t in trackers:
                if t[-1] in face_set:
                    tt = t.copy()
                    tt[-1] = int(self.in_group(tt[-1], self.label_map))
                    if tt[-1] not in self.output_label_map:
                        self.output_label_map[tt[-1]] = count
                        count = count + 1
                    self.final_tracking_result[fn].append(tt)
        dump_pickle(join(stats_dir,"final_result_per_frame.pkl"), self.final_tracking_result)
        self.draw_final_result(face_set)

    def in_group(self, label ,label_map):
        for l in label_map.keys():
            if label in label_map[l]:
                return l
        return label

    def draw_final_result(self, faceset):
        save_dir = join(result_dir,"final_result")
        clean_folder(save_dir)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(int(len(self.output_label_map.keys())) + 1)]
        for fn in self.final_tracking_result.keys():
            trackers = self.final_tracking_result[fn]
            image_fn = fn.strip().split(".tx")[0]
            #print(image_fn)
            image = cv2.imread(self.image_dir + image_fn) 
            if len(trackers) == 0:
                cv2.imwrite(join(save_dir,image_fn) , image)        
                continue
            for t in trackers:
                color = colors[self.output_label_map[int(t[-1])]]
                plot_one_box(t, image, color= color, label=str(int(t[-1])), line_thickness=None)
            cv2.imwrite(join(save_dir,image_fn) , image)

    '''
    def filter_tracking(self,scene_list_fn):
        scene_dict:{} = read_scene_list("./scene_list.txt")
        label_set = set()
        for k in scene_dict.keys():
            start = k
            end = scene_dict[k]
            temp_set = set()
            for i in range(start, end):
                image_fn = "frame_"+str(i)+".jpg.txt"
                if image_fn in self.tracking_result:
                    trackers = self.tracking_result[image_fn]
                    if isinstance(trackers, list):
                        continue
                    for l in trackers[:,-1].tolist():
                        temp_set.add(l)            
            if len(label_set.intersection(temp_set)) == 0:
                label_set = label_set.union(temp_set)
                for i in range(start, end):
                    image_fn = "frame_"+str(i)+".jpg.txt"
                    if image_fn in self.tracking_result:
                        self.filtered_tracking_result[image_fn] = self.tracking_result[image_fn]
    
    def face_grouping(self):
        self.faceGrouping = FaceGrouping("./filtered_tracking_group/")
        self.group_matching = self.faceGrouping.group_maping
        group_matching_temp = {}
        for k in list(self.group_matching.keys()):
            key = int(k)
            value = int(self.group_matching[k])
            if abs(key - value) > 90:
                continue
            group_matching_temp[key] = value

        for fn in self.filtered_tracking_result:
            trackers = self.filtered_tracking_result[fn]
            temp_set = set()
            if isinstance(trackers, list):
                continue
            for t in trackers[-1]:
                temp_set.add(int(t))
            for k in list(group_matching_temp.keys()):
                if k in temp_set and group_matching_temp[k] in temp_set:
                    group_matching_temp.pop(k, None)
        with open("./group_matching.txt", "w") as f:
            for k in group_matching_temp.keys():
                line = str(k) + "\t" + str(group_matching_temp[k]) + "\n"
                f.write(line)
    '''
    def draw_tracking_result(self):
        save_dir = os.path.join(result_dir," tracking_sample_demo")
        clean_folder(save_dir)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(int(self.number_of_object) + 1)]
        for fn in self.tracking_result.keys():
            trackers = self.tracking_result[fn]
            image_fn = fn.strip().split(".tx")[0]
            #print(image_fn)
            image = cv2.imread(self.image_dir + image_fn) 
            if isinstance(trackers, list):
                cv2.imwrite(save_dir + image_fn , image)        
                continue
            for t in trackers:
                color = colors[int(t[-1])]
                plot_one_box(t, image, color= color, label=str(int(t[-1])), line_thickness=None)
            cv2.imwrite(save_dir + image_fn , image)

    def draw_filtered_tracking_result(self):
        save_dir = os.path.join(result_dir,"filtered_tracking_sample/")
        clean_folder(save_dir)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(int(100000) + 1)]
        for fn in self.filtered_tracking_result.keys():
            trackers = self.filtered_tracking_result[fn]
            image_fn = fn.strip().split(".tx")[0]
            #print(image_fn)
            image = cv2.imread(self.image_dir + image_fn) 
            if isinstance(trackers, list):
                cv2.imwrite(save_dir + image_fn , image)        
                continue
            for t in trackers:
                print(int(t[-1]))
                plot_one_box(t, image, color=colors[int(t[-1])], label=str(int(t[-1])), line_thickness=None)
            cv2.imwrite(save_dir + image_fn , image)


    def export_filtered_tracking(self):
        save_dir = os.path.join(result_dir,"filtered_tracking_group")
        clean_folder(save_dir)
        for key in self.filtered_tracking_result.keys():
            trackers = self.filtered_tracking_result[key]
            if isinstance(trackers, list) and len(trackers) == 0:
                continue
            image_fn = key.split(".")[0]
            image = Image.open(self.image_dir + image_fn + ".jpg")
            for t in trackers:
                label = str(int(t[-1]))
                group_dir = save_dir + label + "/"
                if not os.path.exists(group_dir):
                    os.mkdir(group_dir)
                im_c = image.crop((t[0], t[1], t[2], t[3]))
                im_c.save(group_dir+image_fn+"_"+label+".jpg")

    def export_confused_tracking(self):
        save_dir = os.path.join(result_dir,"confused_tracking_sample")  
        clean_folder(save_dir)
        set_1 = set(self.filtered_tracking_result.keys())
        set_2 = set(self.tracking_result.keys())
        confused_fns = set_2 - set_1 
        for key in confused_fns:
            image_fn = key.split(".")[0]
            image = Image.open(self.image_dir + image_fn + ".jpg")                 
            trackers = self.tracking_result[key]
            if isinstance(trackers, list) and len(trackers) == 0:
                continue
            for t in trackers:
                label = str(int(t[-1]))
                group_dir = save_dir + label + "/"
                if not os.path.exists(group_dir):
                    os.mkdir(group_dir)
                im_c = image.crop((t[0], t[1], t[2], t[3]))
                im_c.save(group_dir+image_fn+"_"+label+".jpg")
    
    def export_scene_switch_label_mapping(self, saved_fn):
        with open(saved_fn, "w") as f:
            for i in self.scene_switch_label_mapping:
                i_s = i.split("#")
                f.write(i_s[0] +"\t"+i_s[1] + "\n")
            