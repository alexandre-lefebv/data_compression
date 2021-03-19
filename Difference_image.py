import numpy as np

from tqdm import tqdm, trange

def Difference_image_compute(data):
    """ Compute the images successive difference Ik+1 - Ik, I0 remain unchanged
    Args:
        data (array(nb_frames,_)) : All the images of the video.
    Return:
        modified_data (array(nb_frames,_)) : All the difference images of the
            video.
    """
    
    nb_frames  = data.shape[0]

    modified_data = np.zeros_like(data)
    modified_data[0] = data[0]
    for k in trange(1,nb_frames,desc="Computing difference images"):
        modified_data[k] = data[k]-data[k-1]
    
    return modified_data


def Difference_image_reverse(modified_data):
    """ Reverse-compute the images successive difference Ik+1 - Ik, I0 remain
    unchanged.
    Args:
        modified_data (array(nb_frames,_)) : All the difference images of the
            video.
    Return:
        data (array(nb_frames,_)) : All the images of the video.
    """
    
    nb_frames  = modified_data.shape[0]

    data = np.array(modified_data)
    for k in trange(1,nb_frames,desc="Reversing difference images"):
        data[k] += data[k-1]
    
    return data