import numpy as np

width = 176
height = 144
frame_size = width*height + 2*width//2*height//2

def Difference_image_compute(data,nb_diff_frame='all'):
    """ Compute the images successive difference Ik+1 - Ik, I0 remain unchanged.
    If a lossy coding is used, the frames I_(k*nb_diff_frame) remain unchanged.
    
    Args:
        data (array(nb_frames,_)) : All the frames of the video.
        nb_diff_frame (int) : one frame every <nb_diff_frame> the difference is
            not computed. This improves the quality when there is a quantization
            but reduce the compression rate .
    Return:
        modified_data (array(nb_frames,_)) : All the difference images of the
            video.
    """
    
    nb_frames = data.size//frame_size
    if nb_diff_frame=='all':
        nb_diff_frame = nb_frames
    
    reshaped_data = np.array(data).reshape(nb_frames,frame_size)
    modified_data = np.array(reshaped_data)
    modified_data[1:] -= modified_data[:-1]
    for keep_frame_id in range(nb_diff_frame, nb_frames, nb_diff_frame):
        modified_data[keep_frame_id] = reshaped_data[keep_frame_id]
    
    return modified_data


def Difference_image_reverse(modified_data,nb_diff_frame='all'):
    """ Reverse-compute the images successive difference Ik+1 - Ik, I0 remain
    unchanged. If a lossy coding is used, the frames I_(k*nb_diff_frame) remain
    unchanged.
    Args:
        modified_data (array(nb_frames,_)) : All the difference images of the
            video.
        nb_diff_frame (int) : one frame every <nb_diff_frame> the difference is
            not computed. This improves the quality when there is a quantization
            but reduce the compression rate .
    Return:
        data (array(nb_frames,_)) : All the frames of the video.
    """
    
    nb_frames = modified_data.size//frame_size
    if nb_diff_frame=='all':
        nb_diff_frame = nb_frames
    
    reshaped_data = np.array(modified_data).reshape(nb_frames,frame_size)
    data = np.array(reshaped_data)
    for frame_id in range(nb_frames):
        if frame_id%nb_diff_frame != 0:
            data[frame_id] += data[frame_id-1]
    
    data[data<0]=0
    data[data>255]=255
    return data