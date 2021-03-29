import numpy as np
from DCT import DCT_compute, DCT_inverse


width = 176
height = 144
frame_size = width*height + 2*width//2*height//2
nb_dct_bloc = height//8*width//8+2*height//16*width//16
sep1 = height*width
sep2 = sep1+height//2*width//2

big_diam = (np.array(np.nonzero([[0,0,1,0,0],
                                 [0,1,0,1,0],
                                 [1,0,0,0,1],
                                 [0,1,0,1,0],
                                 [0,0,1,0,0]])).T-2)*2

small_diam = (np.array(np.nonzero([[0,1,0],
                                   [1,0,1],
                                   [0,1,0]])).T-1)*2

dist_sq = lambda a,b : ((a-b)**2).max()


def Motion_estimate_compute_1frame(ref1_frame,ref2_frame,target_frame,block_size):
    """ Match block of 16x16 of the image to be estimated with a block in a
    reference image.
    
    Args:
        ref1_frame, ref2_frame (array(frame_size)) : Frames on witch to look
            for the patches.
        target_frame (array(block_size)) : Frame to be estimated.
    Return:
        P_frame (array) : contatenation of flatenned
            vect_field (array(frame_size,3)) : motion vectors for each blocks.
                The first channel is 0 or 1 (selected reference frame), the
                second and third channels are the motion vector coordinates.
            dct_error (array(frame_size)) : Error image after applying the
                patches.
    """

    ref_Y, ref_U, ref_V = [], [], []
    for ref_frame in [ref1_frame,ref2_frame]:
        ref_Y.append(np.array(ref_frame[    :sep1]).reshape(height,width))
        ref_U.append(np.array(ref_frame[sep1:sep2]).reshape(height//2,width//2))
        ref_V.append(np.array(ref_frame[sep2:    ]).reshape(height//2,width//2))
    
    tar_Y = target_frame[    :sep1].reshape(height,width)
    tar_U = target_frame[sep1:sep2].reshape(height//2,width//2)
    tar_V = target_frame[sep2:    ].reshape(height//2,width//2)
    
    err_Y = np.array(tar_Y)
    err_U = np.array(tar_U)
    err_V = np.array(tar_V)
    vect_field = np.zeros((height//block_size,width//block_size,3),dtype=int)
    
    for X in range(0,height//block_size):
        for Y in range(0,width//block_size):
            xa, xz = X*block_size,(X+1)*block_size
            ya, yz = Y*block_size,(Y+1)*block_size
            # Find the motion vector for the block XY
                     
            ref,vx,vy = Motion_estimate_compute_P_1block(ref_Y[0],ref_Y[1],
                                                         tar_Y[xa:xz,ya:yz],
                                                         [xa,ya])
            
            vect_field[X,Y,:] = np.array([ref,vx,vy])
            
            pxa, pxz = xa+vx,xz+vx
            pya, pyz = ya+vy,yz+vy
            
            patch_Y = ref_Y[ref][pxa:pxz,pya:pyz]
            patch_U = ref_U[ref][pxa//2:pxz//2,pya//2:pyz//2]
            patch_V = ref_V[ref][pxa//2:pxz//2,pya//2:pyz//2]
            
            err_Y[xa:xz,ya:yz]             -= patch_Y
            err_U[xa//2:xz//2,ya//2:yz//2] -= patch_U
            err_V[xa//2:xz//2,ya//2:yz//2] -= patch_V
            
    frame_error = np.concatenate((err_Y.flatten(),
                                  err_U.flatten(),
                                  err_V.flatten()))
    dct_error = DCT_compute(frame_error,offset=0,Q='opti') # Error -> mean = 0
                                                           #       -> offset =0
    
    P_frame = np.concatenate((vect_field.flatten(),dct_error.flatten()))
    
    return P_frame


def Motion_estimate_reverse_1frame(ref0_frame,ref1_frame,P_frame,block_size):
    """ Reconstruct the frame that was estimated by the P_frame (see the doc of
    Motion_estimate_compute_P for details about the parameters. The return of
    this function is the 'target_frame')"""
                                            
    nb_blocks = width//block_size*height//block_size
                                                            
    vect_field = np.array(P_frame[:nb_blocks*3])
    vect_field = vect_field.reshape((height//block_size,width//block_size,3))
    
    frame_error = DCT_inverse(np.array(P_frame[nb_blocks*3:]),offset=0)
    tar_Y = frame_error[    :sep1].reshape(height,width)
    tar_U = frame_error[sep1:sep2].reshape(height//2,width//2)
    tar_V = frame_error[sep2:    ].reshape(height//2,width//2)
                                   
    ref_frame = [ref0_frame,ref1_frame]
    
    for X in range(0,height//block_size):
        for Y in range(0,width//block_size):
            xa, xz = X*block_size,(X+1)*block_size
            ya, yz = Y*block_size,(Y+1)*block_size
                                                            
            ref,vx,vy = vect_field[X,Y,:]
            
            pxa, pxz = xa+vx,xz+vx
            pya, pyz = ya+vy,yz+vy
            
            patch_Y = ref_Y[ref][pxa:pxz,pya:pyz]
            patch_U = ref_U[ref][pxa//2:pxz//2,pya//2:pyz//2]
            patch_V = ref_V[ref][pxa//2:pxz//2,pya//2:pyz//2]
            
            tar_Y[xa:xz,ya:yz]             += patch_Y
            tar_U[xa//2:xz//2,ya//2:yz//2] += patch_U
            tar_V[xa//2:xz//2,ya//2:yz//2] += patch_V

    target_frame = np.concatenate((tar_Y.flatten(),
                                   tar_U.flatten(),
                                   tar_V.flatten()))
    return target_frame
    
                                                            
def Motion_estimate_compute_P_1block(ref0_frame, ref1_frame, target_block, xy0):
    """ Return a vector that give the (non-optimal) patch matching the targeted
    block and the selected reference frame.
    """

    block_size = len(target_block)
    x0, y0 = xy0
                                                        
    err1=dist_sq(target_block,ref0_frame[x0:x0+block_size,y0:y0+block_size])
    err2=dist_sq(target_block,ref1_frame[x0:x0+block_size,y0:y0+block_size])
    min_err = min(err1,err2)
    min_err_ref= (min_err==err2)
    vx,vy = 0,0

    diam_coor_list = [np.zeros((8,2),dtype=int),np.zeros((4,2),dtype=int)]
    diam_list      = [big_diam,small_diam]
    for diam_id in range(2):
        # Big then small diamond
        diam_coor    = diam_coor_list[diam_id]
        diam         = diam_list[diam_id]
        prev_min_err = min_err+1
        while prev_min_err>min_err:
            prev_min_err = min_err

            # Filter valid diamond coordinates
            diam_coor[:,0] = diam[:,0]+x0+vx
            diam_coor[:,1] = diam[:,1]+y0+vy
            in_range = (diam_coor[:,0]>=0)*(diam_coor[:,1]>=0)
            in_range*= (diam_coor[:,0]<(height-block_size+1))
            in_range*= (diam_coor[:,1]<(width -block_size+1))
            keep_diam_coor = diam_coor[in_range,:]
            
            for pxa,pya in keep_diam_coor:
                pxz,pyz = pxa+block_size,pya+block_size
                err0 = dist_sq(target_block,ref0_frame[pxa:pxz,pya:pyz])
                err1 = dist_sq(target_block,ref1_frame[pxa:pxz,pya:pyz])
                if err0<min_err:
                    min_err = err0
                    min_err_ref = 0
                    vx,vy = pxa-x0,pya-y0
                if err1<min_err:
                    min_err = err1
                    min_err_ref = 1
                    vx,vy = pxa-x0,pya-y0
    
    return min_err_ref,vx,vy


def Motion_estimate_inverse_1frame(symbols_stream, frame_start_id, frame_type, block_size):
    """ Read the symbol stream until a complete frame is found, return the end
    index of the compressed frame.
    Args:
        symbols_stream (array) : compressed data symbols.
        frame_start_id (int) : start index of the frame.
        frame_type ('I' or 'P') : frame type, needed to know how to decode the
            symbol sequence.
    Returns:
        frame_end_id (int) : end index of the frame.
    """
    n = symbols_stream.size
    frame_end_id = frame_start_id

    if frame_type == 'P':
        frame_end_id += width//block_size*height//block_size*3 # motion vectors
        frame_end_id += 2
        for _ in range(nb_dct_bloc):
            frame_end_id += 1
            while frame_end_id < n and symbols_stream[frame_end_id-1] != 0:
                frame_end_id += 2
        return frame_end_id

    if frame_type == 'I':
        frame_end_id += 2
        for _ in range(nb_dct_bloc):
            frame_end_id += 1
            while frame_end_id < n and symbols_stream[frame_end_id-1] != 0:
                frame_end_id += 2
        return frame_end_id

    else :
        print("Motion_estimate_inverse_1frame error : invalide format "\
              "specified.")


def Motion_estimate_compute(data,block_size=16):
    """ Encode a list of frames using alternatively reference frames encoded
    with the DCT and frames encoded using 2 reference images and motion
    estimate (IPIPIPII -> IIPIPIP(-1)I).
    Args:
        data (array) : frames to compress
    Return:
        symbols_stream (array) : compressed data symbols
    """

    nb_blocks = width//block_size*height//block_size
    nb_frames = data.size//frame_size
    frames = np.array(data).reshape(nb_frames,frame_size)
    symbols_stream = [DCT_compute(frames[0],offset=128)]
    print(symbols_stream[-1].shape)

    for frame_index in range(1,nb_frames-1,2):
        # I
        symbols_stream.append(DCT_compute(frames[frame_index+1],offset=128))
        print(symbols_stream[-1].shape)
        # P
        P_frame = Motion_estimate_compute_1frame(frames[frame_index-1],
                                                 frames[frame_index+1],
                                                 frames[frame_index],
                                                 block_size=block_size)
        
        print(P_frame[-1].shape)
        symbols_stream.append(P_frame)

    # Extra I if there is an odd number of frames
    if nb_frames%2 == 0:
        symbols_stream.append(np.array([-1]))
        symbols_stream.append(DCT_compute(frames[-1],offset=128))
        print(symbols_stream[-1].shape)
    symbols_stream = np.concatenate(symbols_stream)
    
    print(symbols_stream[17870:17890])
    return symbols_stream


def Motion_estimate_inverse(symbols_stream, block_size=16):
    """ Decode a list of symbols using alternatively reference frames encoded
    with the DCT and frames encoded using 2 reference images and motion
    estimate.
    Args:
        symbols_stream (array) : compressed data symbols.
    Return:
        data (array) : frames.
    """
    print("####")
    print(symbols_stream[17870:17890])
    data = []
    nb_dct_bloc = height//8*width//8+2*height//16*width//16
    n = symbols_stream.size

    block_start_id = 0
    block_end_id   = block_start_id+1
    frame_end_id = Motion_estimate_inverse_1frame(symbols_stream,0,
                                                  'I', block_size)
    
    print(symbols_stream[0:frame_end_id].shape)
    I_tm1 = DCT_inverse(symbols_stream[0:frame_end_id],offset=128)
    data.append(np.array(I_tm1))
    frame_start_id = frame_end_id
    
    while frame_start_id != n:
        # I
        frame_end_id = Motion_estimate_inverse_1frame(symbols_stream,
                                                      frame_start_id,
                                                      'I', block_size)
        print(symbols_stream[frame_start_id:frame_end_id].shape)
        I_tp1 = DCT_inverse(symbols_stream[frame_start_id:frame_end_id],
                            offset=128)
        frame_start_id = frame_end_id

        # P
        if frame_start_id != n:
            frame_end_id = Motion_estimate_inverse_1frame(symbols_stream,
                                                          frame_start_id,
                                                          'P', block_size)
            P = symbols_stream[frame_start_id:frame_end_id]
            print(symbols_stream[frame_start_id:frame_end_id].shape)
            frame_start_id = frame_end_id
            P_frame = Motion_estimate_reverse_1frame(I_tm1,I_tp1,P,block_size)
            data.append(np.array(P_frame))

        data.append(np.array(I_tp1))

        I_tm1 = np.array(I_tp1)



    return data


