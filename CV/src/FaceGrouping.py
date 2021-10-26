import face_recognition
import os
import pickle
import numpy as np
import cv2
class FaceGrouping:
    
    def __init__(self, face_folder_dir):
        self.face_folder_dir = face_folder_dir
        self.sub_folder = os.listdir(face_folder_dir) 
        self.found_group:{str,[]} = {}
        self.encodings_lookup = {}
        self.image_look_up = {}
        self.distance_map ={}
        self.histogram_lookup = {}
        self.find_face()
        self.group_face()
        self.export_face_group()
        self.export_group_mapping()
        
    def find_face(self):
        counter = 0
        for f in self.sub_folder:
            group_folder = self.face_folder_dir + f
            imf = sorted(os.listdir(group_folder))
            if len(imf) <= 3:
                continue
            print("current progress ", f)
            for i in imf:
                counter = counter + 1
                image = face_recognition.load_image_file(group_folder+"/"+i)
                self.image_look_up[i] = image
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) != 0:
                    group_num = f
                    if group_num in self.found_group:
                        self.found_group[group_num].append(i)
                    else:
                        self.found_group[group_num] = []
                        self.found_group[group_num].append(i)
                    self.face_histogram(i,image,face_locations)
        with open('image_look_up.pkl', 'wb') as ff: 
            pickle.dump(self.image_look_up, ff)

    def group_face(self):
        face_group = sorted(list(self.found_group.keys()))
        for i in range(len(face_group)):
            group_num = face_group[i]
            for j in range(i+1, len(face_group)):
                group_num_cmp =face_group[j]
                if group_num == group_num_cmp:
                    continue 
                self.distance_map[str(group_num)+"##"+str(group_num_cmp)] = self.distance(group_num, group_num_cmp)
        with open('distance.pkl', 'wb') as ff: 
            pickle.dump(self.distance_map, ff)
        with open('encodings_lookup.pkl', 'wb') as ff: 
            pickle.dump(self.encodings_lookup, ff)
        return 
    def distance(self, group_num, group_num_cmp):
        encodings = []
        dis = []
        for image_fn in self.found_group[group_num]:
            if image_fn in self.encodings_lookup:
                my_encoding = self.encodings_lookup[image_fn]
            else:
                if image_fn in self.image_look_up:
                    image = self.image_look_up[image_fn]
                else:
                    image = face_recognition.load_image_file(self.face_folder_dir+group_num+"/"+image_fn)
                my_encoding = face_recognition.face_encodings(image)[0]
                self.encodings_lookup[image_fn] = my_encoding
            my_histogram = self.histogram_lookup[image_fn]
            my_encoding = np.concatenate((my_encoding, my_histogram), axis=None)
            encodings.append(my_encoding)
        encodings = np.array(encodings)
        for cmp_image_fn in self.found_group[group_num_cmp]:
            if cmp_image_fn in self.encodings_lookup:
                unknown_face_encoding = self.encodings_lookup[cmp_image_fn]
            else:  
                if cmp_image_fn is self.image_look_up:
                    image_cmp = self.image_look_up[cmp_image_fn]  
                else:
                    image_cmp = face_recognition.load_image_file(self.face_folder_dir+group_num_cmp+"/"+cmp_image_fn)
                unknown_face_encoding = face_recognition.face_encodings(image_cmp)[0]
                unknow_face_histogram = self.histogram_lookup[cmp_image_fn]
                unknown_face_encoding = np.concatenate((unknown_face_encoding, unknow_face_histogram), axis=None)
            results = face_recognition.face_distance(encodings, unknown_face_encoding)
            dis.append(results.mean())
        return sum(dis) * 1.0/ len(dis)

    def export_face_group(self):
        found = self.found_group.keys()
        with open("./face.txt", "w") as f:
            for ff in found:
                f.write(ff +  "\n")

    def export_group_mapping(self):
        with open("./group_matching.txt", "w") as f:
            for k in self.distance_map.keys():
                k_s = k.split("##")
                line = str(k_s[0]) + "\t" + str(k_s[1]) + "\t" + str(self.distance_map[k]) + "\n"
                f.write(line)
    
    def face_histogram(self, fn ,pil_image, face_location):
        total_length = 126
        opencv_image = self.convert_to_opencv(pil_image)
        location = face_location[0]
        shape = opencv_image.shape[:2]
        x1, x2, y1, y2 = self.expand_rect(shape, location)
        mask = np.zeros(shape, np.uint8)
        mask[x1:x2, y1:y2] = 255
        mask_size = (x2-x1)*(y2-y1)
        res = self.color_histogram(opencv_image,mask, total_length, mask_size)
        self.histogram_lookup[fn] = res

    def convert_to_opencv(self, pil_image):
        # Convert RGB to BGR 
        return pil_image[:, :, ::-1].copy() 

    def expand_rect(self,shape, location, p = 0.2):
        #print(location)
        #print(shape)
        y1 = max(location[3] - shape[1] * p/2, 0)
        y2 = min(location[1] + shape[1] * p/2, shape[1])
        x1 = max(location[0] - shape[0] * p/2, 0)
        x2 = min(location[2] + shape[0] * p/2, shape[0])
        return int(x1), int(x2), int(y1), int(y2)

    def color_histogram(self, opencv_image, mask, total_length, mask_size):
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        res = []
        lens = int(total_length/3)
        chans = cv2.split(opencv_image)
        # loop over the image channels
        for chan in chans:
            # create a histogram for the current channel and
            # concatenate the resulting histograms for each
            # channel
            hist = cv2.calcHist([chan], [0], None, [lens], [0, 256])
            normalized_hist = hist *1.0/mask_size
            res.append(normalized_hist)
        return np.concatenate((np.concatenate((res[0], res[1]), axis=None), res[2]), axis=None)
