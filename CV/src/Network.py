import networkx as nx 
import matplotlib.pyplot as plt
from utilities import *
import shutil
import os
class Network:
    def __init__(self):
        super().__init__()
        self.graph = nx.Graph()
        self.filtered_graph = None
        self.connected = []
    def construct(self, fn, max_duration = 999999):
        checked = set()
        checked_single = set()
        with open(fn ,"r") as f:
            lines = f.readlines() 
        for line in lines:
            line_s = line.strip().split("\t")
            from_node = line_s[0]
            end_node = line_s[1]
            if from_node == end_node:
                continue
            check_label = line_s[0] + "#" + line_s[1] 
            if check_label in checked:
                continue 
            else:
                checked.add(check_label)
            if from_node not in checked_single :
                self.graph.add_edge(from_node, from_node, weight=float(0.001))
                checked_single.add(from_node)
            if  end_node not in checked_single:
                self.graph.add_edge(end_node, end_node, weight=float(0.001))
                checked_single.add(end_node)
            self.graph.add_edge(from_node, end_node,weight=float(line_s[2]))
    
    def filter_by_frame(self, frame_dict):
        if self.filtered_graph == None:
            self.filtered_graph = self.graph.copy()
        for fn in frame_dict:
            trackers = frame_dict[fn]
            temp_set = set()
            if isinstance(trackers, list):
                continue
            for t in list(trackers[:,-1]):
                temp_set.add(t)
            for i in temp_set:
                for j in temp_set:
                    if i == j:
                        continue
                    if self.filtered_graph.has_edge(i, j):
                        self.filtered_graph.remove_edge(i, j)
    def filter_by_distance(self, max_distance, threshold):
        if self.filtered_graph == None:
            self.filtered_graph = self.graph.copy()
        edge_list = list(self.filtered_graph.edges()).copy()
        for e in edge_list:
            if "_" in e[0] or "_" in e[1]:
                continue 
            if abs(int(e[0]) - int(e[1]))%max_distance > threshold :
                self.filtered_graph.remove_edge(e[0], e[1])

    def filter_by_scene_switch(self,scene_switch_fn):
        if self.filtered_graph == None:
            self.filtered_graph = self.graph.copy()
        with open(scene_switch_fn, "r") as f:
            lines = f.readlines()
        for line in lines:
            line_s = line.split("\t")
            if self.filtered_graph.has_edge(line_s[0], line_s[1]):
                self.filtered_graph.remove_edge(line_s[0], line_s[1])
    def filter_by_bad_group(self, bad_group_fn):
        if self.filtered_graph == None:
            self.filtered_graph = self.graph.copy()
        bad_group = self.load_pickle(bad_group_fn)
        for value in bad_group.values():
            for v in value:
                for v1 in value:
                    if self.filtered_graph.has_edge(str(v), str(v1)):
                        self.filtered_graph.remove_edge(str(v), str(v1))        


    def filter_by_threshold(self, threshold):
        if self.filtered_graph == None:
            self.filtered_graph = self.graph.copy() 
        #print([(u, v) for (u, v, d) in self.filtered_graph.edges(data=True) if d['weight']< threshold])
        self.filtered_graph = nx.Graph([(u, v, d) for (u, v, d) in self.filtered_graph.edges(data=True) if d['weight']< threshold])
        
    def export_network(self, option = "raw"):
        if option == "filtered":
            if self.filtered_graph == None:
                print("Filtered Graph is None")
            self.draw_network(self.filtered_graph)
        elif option == "raw":
            self.draw_network(self.graph)
            
    def draw_network(self, G):
        pos=nx.spring_layout(G) # positions for all nodes
        # nodes
        nx.draw_networkx_nodes(G,pos,node_size=100)

        # edges
        nx.draw_networkx_edges(G,pos,width=2)

        # labels
        nx.draw_networkx_labels(G,pos,font_size=5,font_family='sans-serif')

        plt.axis('off')
        plt.savefig("weighted_graph.png") # save as png
        plt.show()   
    
    def parse_label_map(self, connected): 
        res = []
        for c in connected:
            temp = []
            for cc in c:
                if "_" in cc:
                    for ccc in cc.split("_"):
                        temp.append(int(ccc))
                else:
                    temp.append(int(cc))         
            res.append(temp)
        return res
    
    def export_group_result(self, original_dir, saved_dir):
        clean_folder(saved_dir)
        connected =  list(nx.connected_components(self.filtered_graph))
        #print(connected)
        connected = self.parse_label_map(connected)
        #print(connected)
        self.connected = connected
        for c in connected:
            to_dir = join(saved_dir, str(list(c)[0]))
            os.mkdir(to_dir)
            for cc in c:                                        
                from_dir = join(original_dir , str(cc))
                list_f = os.listdir(from_dir)
                for lf in list_f:
                    shutil.copy(os.path.join(from_dir, lf), os.path.join(to_dir,lf))
    
    def get_label_map(self):
        res = {}
        connected =  list(nx.connected_components(self.filtered_graph))
        connected = self.parse_label_map(connected)
        for l in connected:
            ls = list(l)[0]
            res[ls] = set(l)
        return res

    def load_pickle(self, fn):
        if not os.path.exists(fn):
            return
        with open(fn, "rb") as f:
            lookup = pickle.load(f)
            print(fn,"is" ,len(lookup.keys()))
        return lookup