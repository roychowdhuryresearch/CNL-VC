import random
import numpy as np
class RandomDrop(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self):
        return 

    def __call__(self, sample):
        image, region_tag, indices = sample['image'], sample['region_tag'], sample['indices']
        #print(indices)
        index = self.shuffle_by_part(indices)
        #print(index)
        image = image[:,index,:]
        #region_tag = region_tag[index,:]
        return {'image': image, 'region_tag': region_tag}
    
    def shuffle_by_part(self, indices):
        index = []
        classname, idx = np.unique(indices , return_inverse=True)
        for c in classname:
            loc = np.where(indices == c)[0]
            np.random.shuffle(loc)
            index.append(loc)
            #print(loc.min(), loc.max())
        return np.concatenate(index)