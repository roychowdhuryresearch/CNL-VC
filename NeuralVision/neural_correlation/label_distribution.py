import sys
sys.path.append(".")
from data import num_characters
from utilities import *

def get_character_correlation():
    label_dir = "/media/yipeng/toshiba/movie/Movie_Analysis/Character_TimeStamp_resnet"
    character_labels = {}
    for character_index in range(num_characters):
        character_labels[character_index] = set(load_pickle(os.path.join(label_dir , str(character_index) +".pkl")))
    res = np.zeros((num_characters, num_characters))
    for character_index_cond in sorted(character_labels.keys()):
        one_character_label_cond = character_labels[character_index_cond]
        for frame_number in sorted(one_character_label_cond):
            for character_index in sorted(character_labels.keys()):
                one_character_label = character_labels[character_index]
                if frame_number in one_character_label:
                    res[character_index, character_index_cond] += 1
    
    for character_index in range(num_characters):
        for character_index_cond in range(num_characters):
            res[character_index,character_index_cond] /= len(character_labels[character_index_cond])
    print("   condition on character")
    print("   0     1     2     3   ")
    for character_index in range(num_characters):
        line = str(character_index) + " "  
        for character_index_cond in range(num_characters):
            line += " %0.3f" % res[character_index,character_index_cond]
        print(line)
get_character_correlation()