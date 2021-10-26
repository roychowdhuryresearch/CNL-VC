class Scene:
    def __init__(self, scene):
        self.scene = scene
        self.clip_list = []
        self.length = -1
        self.character_occurance = {}
    
    def __lt__(self, other):
        return self.scene < other.scene

    def set_clip_list(self, clip_list):
        self.clip_list = clip_list

    def append(self, clip):
        self.clip_list.append(clip)
    
    def stats_cal(self):
        self.calculate_length()
        self.calculate_occurance_cv()

    def calculate_length(self):
        for c in self.clip_list:
            self.length += c.get_width()

    def calculate_occurance_cv(self):
        for c in self.clip_list:
            for character_index in c.cvlabel_checkbox.keys():
                if character_index not in self.character_occurance:
                    self.character_occurance[character_index] = 0
                self.character_occurance[character_index] += len(c.cvlabel_checkbox[character_index])
        #print(self.character_occurance)

    def appear_enough(self, character_index, threshold=0):
        if threshold == 0:
            return character_index in self.character_occurance
        elif character_index in self.character_occurance:
            return 4*self.character_occurance[character_index] > max(threshold*self.length, 30)
        return False
        


    