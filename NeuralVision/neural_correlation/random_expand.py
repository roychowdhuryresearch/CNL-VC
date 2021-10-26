import random
import numpy as np
class RandomExpand(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, neurons_per_region):
        self.neurons_per_region = neurons_per_region
        return 

    def __call__(self, sample):
        image, region_tag, indices = sample['image'], sample['region_tag'], sample['indices']
        #print(indices)
        image, region_tag = self.append_blank(image, region_tag, indices)
        return {'image': image, 'region_tag': region_tag}
    
    def append_blank(self, image, region_tag,indices):
        classname, idx = np.unique(indices , return_inverse=True)
        index = []
        for c in sorted(classname):
            loc = np.where(indices == c)[0]
            diff = len(loc) - self.neurons_per_region[c]
            if diff == 0:
                random.shuffle(loc)
                index.append(loc)
                continue
            zeros_index = np.random.choice(loc, abs(diff))
            #print("c is ", c)
            if diff > 0: ## random blank
                #print("erasing")
                image[:,zeros_index,:] = 0
                #print(image[:,zeros_index,:].sum())
            else:
                #print("appending")
                image = np.insert(image, zeros_index, 0, axis = 1)
                region_tag = np.insert(region_tag, zeros_index, region_tag[zeros_index[0]], axis=0)
                indices = np.insert(indices, zeros_index, c)
                #print(image[:,zeros_index,:].sum())
            #print(image.shape)
            #loc = np.where(indices == c)[0]
            #random.shuffle(loc)
            #index.append(loc)
        #image = image[:,np.concatenate(index),:]
        return image, region_tag