import face_recognition
import os
import cv2
from PIL import Image
import pickle
import numpy as np
from utilities import *
from ColorLayoutComputer import ColorLayoutComputer
from project_setting import feature_dir
class FeatureExtractor:
    def __init__(self, face_folder_dir, mode):
        self.face_folder_dir = face_folder_dir
        self.sub_folder = os.listdir(face_folder_dir) 
        self.found_group:{str,[]} = {}
        self.encodings_lookup = {}
        self.histogram_lookup = {}
        self.cloth_histogram_lookup = {}
        self.face_encoding_lookup_cnn = {}
        self.less_face_group = set()
        self.no_face_group = set()
        self.save_dir = feature_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.mode = mode
        self.cld = ColorLayoutComputer()
        self.find_face_feature()
        self.export_less_face_group()
        self.export_no_face_group()
        
    def find_face_feature(self, allowence_percentage = 0.3):
        for f in self.sub_folder:
            group_folder = os.path.join(self.face_folder_dir, f)
            imf = sorted(os.listdir(group_folder))
            if len(imf) <= 3:
                continue
            print("current progress ", f)
            counter = 0
            for i in imf:
                image = face_recognition.load_image_file(os.path.join(group_folder,i))
                face_locations = face_recognition.face_locations(image,number_of_times_to_upsample=1,model="cnn")
                if len(face_locations) != 0:
                    counter = counter + 1
                    self.feature_extract(i,image,face_locations)
            if counter == 0:
                self.no_face_group.add(int(f))
                continue
            if counter < len(f)*allowence_percentage or counter <= 2: 
                self.less_face_group.add(int(f))
                
        if self.mode == "face_histogram":
            saved_fn_1 = "face_histogram_lookup.pkl"
            saved_fn_2 = "cloth_histogram_lookup.pkl"
            dump_pickle(join(self.save_dir,saved_fn_1), self.histogram_lookup)
            dump_pickle(join(self.save_dir,saved_fn_2), self.cloth_histogram_lookup)
        elif self.mode == "face_histogram_he":
            saved_fn_1 = "face_histogram_lookup_he.pkl"
            saved_fn_2 = "cloth_histogram_lookup_he.pkl"
            dump_pickle(join(self.save_dir,saved_fn_1), self.histogram_lookup)
            dump_pickle(join(self.save_dir,saved_fn_2), self.cloth_histogram_lookup)
        elif self.mode == "face_encodding":
            saved_fn = "face_encoding_lookup_cnn.pkl"
            dump_pickle(join(self.save_dir,saved_fn), self.face_encoding_lookup_cnn)
        elif self.mode == "face_encodding_he":
            saved_fn = "face_encoding_lookup_cnn_he.pkl"
            dump_pickle(join(self.save_dir, saved_fn), self.face_encoding_lookup_cnn)
        elif self.mode == "pic_histogram":
            saved_fn = "pic_histogram.pkl"
            dump_pickle(join(self.save_dir, saved_fn), self.histogram_lookup)
        elif self.mode == "CLD":
            saved_fn = "CLD.pkl"
            dump_pickle(join(self.save_dir,saved_fn), self.histogram_lookup)


    def feature_extract(self, image_fn, pil_image, face_location):
        if self.mode == "face_histogram":
            opencv_image = self.convert_to_opencv(pil_image)
            self.face_histogram(image_fn, opencv_image, face_location)
        elif self.mode == "face_histogram_he":
            opencv_image = self.his_equl_Color(self.convert_to_opencv(pil_image))
            self.face_histogram(image_fn, opencv_image, face_location)
        elif self.mode == "face_encodding":
            self.face_encoding_lookup_cnn[image_fn] = face_recognition.face_encodings(pil_image, known_face_locations=face_location,num_jitters=1)
        elif self.mode == "face_encodding_he":
            pil_image = self.his_equl_Color(pil_image)
            self.face_encoding_lookup_cnn[image_fn] = face_recognition.face_encodings(pil_image, known_face_locations=face_location,num_jitters=1)
        elif self.mode == "pic_histogram":
            opencv_image = self.convert_to_opencv(pil_image)
            self.histogram(image_fn, opencv_image)
        elif self.mode == "CLD":
            opencv_image = self.convert_to_opencv(pil_image)
            feature = self.standardize(self.cld.compute(opencv_image))
            self.histogram_lookup[image_fn] = feature
    
            
    def convert_to_opencv(self, pil_image):
        # Convert RGB to BGR 
        return pil_image[:, :, ::-1].copy() 
    
    def face_histogram(self, fn ,opencv_image, face_location):
        histogram_lookup = {}
        cloth_histogram_lookup = {}
        total_length = 63
        location = face_location[0]
        shape = opencv_image.shape[:2]
        x1, x2, y1, y2 = self.expand_rect(shape, location)
        head_patch = opencv_image[x1:x2, y1:y2] 
        mask_size = (x2-x1)*(y2-y1)
        res = self.color_histogram(head_patch, None, total_length, mask_size)
        self.histogram_lookup[fn] = res
        
        x3, x4, y3, y4 = self.shift_rect(shape, location)
        cloth_patch = opencv_image[x3:x4, y3:y4] 
        mask_size = (x4-x3)*(y4-y3)
        res_cloth = self.color_histogram(cloth_patch, None, total_length, mask_size)
        self.cloth_histogram_lookup[fn] = res_cloth

    def histogram(self, fn ,opencv_image):
        total_length = 63
        shape = opencv_image.shape[:2]
        mask_size = shape[0]*shape[1]
        res = self.color_histogram(opencv_image, None, total_length, mask_size)
        self.histogram_lookup[fn] = res


    def expand_rect(self,shape, location, p = 0.2):
        #print(location)
        #print(shape)
        y1 = max(location[3] - shape[1] * p/2, 0)
        y2 = min(location[1] + shape[1] * p/2, shape[1])
        x1 = max(location[0] - shape[0] * p/2, 0)
        x2 = min(location[2] + shape[0] * p/2, shape[0])
        return int(x1), int(x2), int(y1), int(y2)
    
    
    def his_equl_Color(self,img):
        ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
        channels=cv2.split(ycrcb)
        cv2.equalizeHist(channels[0],channels[0])
        cv2.merge(channels,ycrcb)
        cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
        return img
    
    def shift_rect(self, shape, location, p = 0.5):
        x1, x2, y1, y2 = self.expand_rect(shape, location, p = 0.2)
        y1 = y1
        y2 = y2
        x1 = max(x2 -  shape[0] * p/2, 0)
        x2 = min(x2 + shape[0] * p/2, shape[0])
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
            normalized_hist = hist *1.0 / mask_size
            res.append(normalized_hist)
        return np.concatenate((np.concatenate((res[0], res[1]), axis=None), res[2]), axis=None)


    def standardize(self, a):
        a = (a - np.mean(a)) / np.std(a)
        return a

    def export_less_face_group(self):
        print("num of less facel,  " ,len(self.less_face_group))
        dump_pickle(join(self.save_dir ,"less_face_group.pkl"), self.less_face_group)
        
    def export_no_face_group(self):
        print("num of no facel,  " ,len(self.no_face_group))
        dump_pickle(join(self.save_dir, "no_face_group.pkl"), self.no_face_group)