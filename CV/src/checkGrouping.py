import time
from FaceGroupingGPU import *
start_time = time.time()
FaceGroupingGPU("./filter_tracking_result/")
print("--- %s seconds ---" % (time.time() - start_time))
