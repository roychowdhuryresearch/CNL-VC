import random
import numpy as np
class ImageExpand(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, append_loc):
        self.append_loc = append_loc
        return 

    def __call__(self, sample):
        image, region_tag, indices = sample['image'], sample['region_tag'], sample['indices']
        #print(indices)
        image, region_tag = self.append_blank(image, region_tag, indices)
        return {'image': image, 'region_tag': region_tag}
    
    def append_blank(self, image, region_tag,indices):
        classname, idx = np.unique(indices , return_inverse=True)
        res = []
        res_tag = []
        for c in sorted(classname):
            diff, num ,exact_loc = self.append_loc[c] 
            diff = diff[0][0]
            num = num[0][0]
            loc = np.where(indices == c)[0]
            temp = image[:,loc,:]
            temp_tag = region_tag[loc, :]
            if diff == 0:
                res.append(temp)
                res_tag.append(temp_tag)
                continue
            #print("c is ", c)
            if diff < 0: ## random blank
                #print("erasing")
                temp[:,exact_loc,:] = 0
                res.append(temp)
                res_tag.append(temp_tag)
                #print(image[:,zeros_index,:].sum())
            else:
                r = np.zeros((1, num, temp.shape[-1]))
                r[:,exact_loc,:] = temp
                t = np.zeros((num, temp_tag.shape[-1]))
                t[:] = temp_tag[0]
                res.append(r)
                res_tag.append(t)  
                #print(image[:,zeros_index,:].sum())
            #print(image.shape)
            #loc = np.where(indices == c)[0]
            #random.shuffle(loc)
            #index.append(loc)
        image = np.concatenate(res, axis=1)
        region_tag = np.concatenate(res_tag)
        return image, region_tag