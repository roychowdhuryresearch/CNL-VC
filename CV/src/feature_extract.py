from project_setting import result_dir, stats_dir, filter_tracking_result_dir
from FeatureExtractor import FeatureExtractor as FE
modes = ["face_histogram", "face_histogram_he","face_encodding","face_encodding_he","pic_histogram"]
for mode in modes:
    fe = FE(filter_tracking_result_dir, mode= mode)