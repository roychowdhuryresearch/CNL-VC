import os

project_dir = '/media/yipeng/data/movie/Movie_Analysis/'
path_to_matlab_generated_movie_data = os.path.join(project_dir, 'data')
#episode1_cv_label_fn = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_annotation"
episode1_cv_label_fn = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_resnet"
episode2_cv_label_fn = "/media/yipeng/data/movie/Movie_Analysis/time_stamp_2"
#episode2_cv_label_fn = "/media/yipeng/data/movie/Movie_Analysis/Character_TimeStamp_annotation_2"

fps_movie = 30 
episodes = [1, 2]
patientNums = ['431', '433', '435', '436', '439', '441', '442', '444', '445', '452']
time_offset = 2520
num_characters = 4

frame_sample_frequency = 4