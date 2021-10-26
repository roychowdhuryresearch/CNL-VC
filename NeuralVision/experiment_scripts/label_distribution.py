import numpy as np 
import matplotlib.pyplot as plt
labels = np.load("/media/yipeng/data/movie/Movie_Analysis/training_data_1_zero/431/label.npy")
print(labels.shape)
for character_index in range(4):
    character_label = np.argmax(labels[:,character_index,:], axis=1)
    print(character_label.sum())
    len_len = len(character_label)
    print(sum(character_label==0)*1.0/len_len,sum(character_label==1)*1.0/len_len,sum(character_label==2)*1.0/len_len )