import time
start_time = time.time()
from FeatureExtractor import * 
FeatureExtractor("/home/yipeng/Desktop/movie_tracking/tracking/filter_tracking_result/")
print("--- %s seconds ---" % (time.time() - start_time))
