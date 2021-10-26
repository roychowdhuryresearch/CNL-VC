import random
import numpy as np
class RandomShuffle(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        return 
        
    def __call__(self, sample):
        image, region_tag = sample['image'], sample['region_tag']
        index = list(range(image.shape[1]))
        np.random.shuffle(index)
        image = image[:,index,:]
        region_tag = region_tag[index,:]
        return {'image': image, 'region_tag': region_tag}
