from Rect import *

class Record:
    def __init__(self, source):
        self.source = source # image fn 
        self.rect_list = [] # list of all rects in the one image
        self.good_rect_list: [Rect]= [] # list of wanted rects in the one image
    
    @staticmethod
    def create_record(source, size_threshold):
        record = Record(source)
        with open(source, "r") as f:
            lines = f.readlines()
        for line in lines:
            line_s = line.strip().split(" ")
            x0 = int(line_s[0])
            y0 = int(line_s[1])
            x1 = int(line_s[2])
            y1 = int(line_s[3])
            marker = int(line_s[4])
            confidence  = float(line_s[5])
            w = x1 - x0
            h = y1 - y0
            rect = Rect(marker, confidence, x0, y0, w, h)
            record.rect_list.append(rect) 
            if marker != 0 or w * h < size_threshold or 1.0*w/h>2 or 1.0*h/w>2 or confidence < 0.5 or h < 150 or w < 150:
                continue
            record.good_rect_list.append(rect) 
        return record
    
    def generate_detections(self):
        res = []
        for r in self.good_rect_list:
            res.append(r.get_list())
        return np.array(res)
