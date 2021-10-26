import numpy as np 
import sys
sys.path.append(".")
import numpy as np
import copy
class Clip:
    def __init__(self, frame_start, frame_end):
        ## frame number is left right BOTH included
        self.frame_start = frame_start # in term of frame_number
        self.frame_end = frame_end #  in term of frame_number
        self.scene = -1
        self.frame_set = set(range(frame_start, frame_end + 1))
        self.character_checkbox = {}
        self.cvlabel_checkbox = {}
    def __str__(self):
        return str(self.frame_start) + " " +str(self.frame_end) + " " + str(self.scene)
    
    def __eq__(self,other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return self.frame_start < other.frame_start

    def set_scene(self, scene):
        self.scene = scene

    def get_character_checkbox(self):
        character_dic = {0:11, 1:3, 2:6, 3:1}
        res = {}
        for i in character_dic:
            if self.character_checkbox[character_dic[i]] >= 1 :
                start = (int(self.frame_start/4) + 1) *4 
                end = (int(self.frame_end/4) -1 ) *4
                res[i] = set(range(start, end, 4))
            else:
                res[i] = set()
        return res

    def update_character_checkbox(self, character_checkbox):
        self.character_checkbox = character_checkbox

    def cvlabel_filtering(self, threshold = 2):
        clip_length = len(self.frame_set)/4
        res = copy.deepcopy(self.cvlabel_checkbox)
        line = [str(clip_length)]
        for character_index in sorted(self.cvlabel_checkbox.keys()):
            line.append(str(character_index)+ "_" + str(len(self.cvlabel_checkbox[character_index])))
            if clip_length < 10:
                if len(self.cvlabel_checkbox[character_index]) < 3:
                    del res[character_index]
            else:
                if len(self.cvlabel_checkbox[character_index]) * 1.0 /clip_length < 0.4 :
                    del res[character_index]
        print(" ".join(line))
        self.cvlabel_checkbox = res

    def cv_accuracy(self):
        character_dic = {0:11, 1:3, 2:6, 3:1}
        res = np.zeros((len(character_dic), 2, 2))
        for k in character_dic.keys():
            if k in self.cvlabel_checkbox: # and len(character_dic[k]) > th:  
                if self.character_checkbox[character_dic[k]] >= 1:
                    res[k, 0, 0] = 1 #len(self.cvlabel_checkbox[k])  prediction YES label YES TP
                else:
                    res[k, 0, 1] = 1 #len(self.cvlabel_checkbox[k])  Prediction Yes label NO FP
            else:
                if self.character_checkbox[character_dic[k]] >= 1:
                    res[k, 1, 0] = 1 #len(self.cvlabel_checkbox[k]) Predition NO, Label NO  TN
                else:
                    res[k, 1, 1] = 1 #len(self.cvlabel_checkbox[k]) Predition No Label YES  FN
        return res
    
    def contains_character(self, character_index):
        character_dic = {0:11, 1:3, 2:6, 3:1}
        return self.character_checkbox[character_dic[character_index]] >= 1 

    def update_cvlabel(self, character, frame):
        if character not in self.cvlabel_checkbox:
            self.cvlabel_checkbox[character] = [] 
        self.cvlabel_checkbox[character].append(frame)

    def within(self, start, end):
        left_bound = start <= self.frame_start
        right_bound = end >= self.frame_end
        if self.frame_end <= start or self.frame_start >= end :
            return False
        elif left_bound and right_bound:
            return True 
        else:
            print("Error start ", start, " end ", end, "self start ", self.frame_start, "self end ", self.frame_end)

    def contains(self, frame):
        return self.frame_start <= frame and self.frame_end >= frame 

    def get_width(self):
        return self.frame_end - self.frame_start + 1

    def get_overall_character(self):
        num_character = 4
        res = np.zeros(4)
        for character_index in range(num_character):
            if self.contains_character(character_index):
                res[character_index] = 1
        return res

    def get_overall_cvlabel(self):
        num_character = 4
        res = np.zeros(4)
        for character_index in range(num_character):
            if character_index in self.cvlabel_checkbox:
                if len(self.cvlabel_checkbox[character_index])>0:
                    res[character_index] = 1
        return res