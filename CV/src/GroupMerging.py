import numpy as np
from collections import defaultdict
class GroupMerging:
    def __init__(self, has_face_fn, group_mapping_fn):
        self.faces_group = set()
        self.group_mapping = {}
        self.load(group_mapping_fn)
        self.load_face_group(has_face_fn)
        self.merge_map = {}
        self.final_merge_map = {}
        self.group()
        self.append_one_group()

    def load(self, group_mapping_fn):
        with open(group_mapping_fn , "r") as f:
            lines = f.readlines()
        for line in lines:
            line_s = line.strip().split("\t")
            self.group_mapping[line_s[0]] = line_s[1]

    def load_face_group(self, has_face_fn):
        with open(has_face_fn, "r") as f:
            lines = f.readlines()
        for line in lines:
            self.faces_group.add(line.strip())

    def group(self):
        v = defaultdict(list)
        for key, value in sorted(self.group_mapping.items()):
            v[value].append(key)
        self.merge_group(v)
    
    def merge_group(self,dic):
        keys = list(dic.keys())
        for k in keys:
            value = dic[k]
            for kk in keys:
                if kk not in dic:
                    continue
                if kk == k:
                    continue
                if k in dic[kk]:
                    for vv in value:
                        dic[kk].append(vv)
                    dic.pop(k, None)
        self.merge_map = dic

    def append_one_group(self):
        keys = list(self.merge_map.keys())
        all_mapping_group = set()
        for index in range(len(self.merge_map.keys())):
            key = keys[index]
            all_mapping_group.union(set(self.merge_map[key]))
            self.final_merge_map[index] = self.merge_map[key]
        ex_group = self.faces_group - all_mapping_group
        count = len(keys)
        for e in ex_group:
            temp = set()
            temp.add(e)
            self.final_merge_map[count] = temp
            count = count + 1 