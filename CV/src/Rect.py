import math
import numpy as np
class Rect:
    def __init__(self, marker, dis, x, y, w, h):
        self.marker = marker
        self.dis = dis
        self.x = x
        self.y = y 
        self.w = w
        self.h = h 
        self.match = False
    def __str__(self):
        return str(self.marker) + " "+ str(self.x)+ ":" + str(self.y) +":" + str(self.w)+":"+ str(self.h)
    
    def __hash__(self):
        return hash(str(self))    

    def __eq__(self,other):
        return str(self) == str(other)
    
    def __gt__(self, other):
        return self.dis > other.dis

    def get_info(self):
        return self.x, self.y, self.w, self.h, self.marker
    
    def get_list(self):
        return np.array([self.x, self.y, self.x + self.w, self.y + self.h, self.dis])

    def get_location(self):
        return self.x, self.y, self.w, self.h
    
    def get_confidence(self):
        return self.dis
    
    def include(self, other_rect, margin = 0):
        '''
        check if two rect can be merged used in merged rect
        '''
        if other_rect == None:
            return False
        x, y, w, h = other_rect.get_location()
        c1 = self.has_point( x + w/2, y + h/2, margin)
        c2 = other_rect.has_point( self.x +  self.w/2 , self.y +  self.h/2 , margin)
        return c1 and c2 
    
    def has_point(self, point_x,point_y, margin):
        '''
        if the rect contain the pont(x, y) with the margin
        '''
        dx = point_x - self.x
        dy = point_y - self.y
        c1 = dx > (margin * self.w)
        c2 = (self.w *(1 - margin)) > dx 
        c3 = dy > (self.h * margin) 
        c4 = (self.h * (1 - margin)) > dy
        return c1 and c2 and c3 and c4 
    
    def distance_from(self, x, y):
        return math.sqrt((self.x + self.w * 0.5 - x)**2 + (self.y + self.h * 0.5 - y)**2)
      
    def diff(self, arect):
        aw = arect.w
        ah = arect.h

        sw = (self.w + aw)
        sh = (self.h + ah)

        dx = (self.x - arect.x + (self.w - aw).to_f / 2).abs.to_f / sw
        dy = (self.y - arect.y + (self.h - ah).to_f / 2).abs.to_f / sh
        dw = (math.log(self.w) - math.log(aw)).abs.to_f

        return dx + dy + dw
      
    def scale(self, k): # for average calculation
        atype = self.marker
        adis = self.dis
        ax = int(self.x * k)
        ay = int(self.y * k)
        aw = int(self.w * k)
        ah = int(self.h * k)
        return Rect(atype, adis, ax, ay, aw, ah)
    

    def add(self, other): # for average calculation
        atype = self.marker
        adis = 0
        ax = self.x + other.x
        ay = self.y + other.y
        aw = self.w + other.w
        ah = self.h + other.h
        return Rect(atype, adis, ax, ay, aw, ah)
    

    def subtract(self, other): # for difference calculation
        atype = self.marker
        adis = 0
        ax = self.x - other.x
        ay = self.y - other.y
        aw = self.w - other.w
        ah = self.h - other.h
        return Rect(atype, adis, ax, ay, aw, ah)

    def divide(self, n):
        return Rect(self.marker, self.dis, int(self.x / n), int(self.y / n), int(self.w / n), int(self.h / n))
    
