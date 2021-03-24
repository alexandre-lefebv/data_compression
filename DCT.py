import numpy as np
from numpy import pi, cos

from tqdm import tqdm, trange

# Format for qcif images
width = 176
height = 144

# Base for the DCT
vect_uv = np.zeros((8,8,8,8))
for u in range(8):
    for v in range(8):
        cux = np.cos((2*np.arange(0,8,1)+1)*u*np.pi/16)
        cvy = np.cos((2*np.arange(0,8,1)+1)*v*np.pi/16)
        cvyy,cuxx = np.meshgrid(cvy,cux)
        vect_uv [u,v,:,:] = cvyy*cuxx
vect_uv /= 4
vect_uv[0,:,:,:] /= np.sqrt(2)
vect_uv[:,0,:,:] /= np.sqrt(2)

# Psychovisual matrices
P_lumi = np.array( [[16,11,10,16,24, 40, 51, 61],
                    [12,12,14,19,26, 58, 60, 55],
                    [14,13,16,24,40, 57, 69, 56],
                    [14,17,22,29,51, 87, 80, 62],
                    [18,22,37,56,68, 109,103,77],
                    [24,36,55,64,81, 104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])

P_chromi= np.array([[17,18,24,47,99,99,99,99],
                    [18,21,26,66,99,99,99,99],
                    [24,26,56,99,99,99,99,99],
                    [47,66,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99],
                    [99,99,99,99,99,99,99,99]])/6

# Reading order
diag_reading =  np.array([[0, 1, 5, 6, 14,15,27,28],
                          [2, 4, 7, 13,16,26,29,42],
                          [3, 8, 12,17,25,30,41,43],
                          [9, 11,18,24,31,40,44,53],
                          [10,19,23,32,39,45,52,54],
                          [20,22,33,38,46,51,55,60],
                          [21,34,37,47,50,56,59,61],
                          [35,36,48,49,57,58,62,63]]).flatten()
reading_order = np.argsort(diag_reading)


def DCT_compute(data_xy,Q='opti'):
    """ Compute the DCT of all the images of the video, apply a psychovisual
    matrix, read the coeff along the diagonals.

    Args:
        data_xy (array(nb_frames,-1))
        Q (int or 'opti', default='opti'): % of the coefficients of the
            psychovisual matrix to use for quantification, default is the
            smallest value of Q that allows to quantify on -255,+255. If the
            value passed is not high enought to squeeze the quantized values in
            the range -255,+255 the optimal value will e used  instead and a
            warning message is displayed.
    Return:
        data_uv (array): The first value is an int,
            the optimal Q for quantization. Q = <value>/128*100 % of the base
            matrix.
    """
    if type(Q) == int:
        Q/=100
    frame_size = width*height+2*width//2*height//2
    nb_frames = data_xy.size//frame_size
    sep1 = height*width
    sep2 = sep1+height//2*width//2

    if data_xy.size % frame_size != 0:
        print("DCT_compute error: invalide shape. Expected (nb_frames,"\
              "frame_size) with frame_size = "+str(frame_size)+", got "\
              +str(data_xy.shape))
    data_xy = np.array(data_xy).reshape(nb_frames,frame_size)
    #data_xy-= 128 # DCT is meant for values in range -128,+128.

    # Compute the DCT
    optimal_Q = 0
    list_block_uv = []
    for k in trange(nb_frames,desc="DCT"):
        y = data_xy[k,    :sep1].reshape(height,width)
        u = data_xy[k,sep1:sep2].reshape(height//2,width//2)
        v = data_xy[k,sep2:   ].reshape(height//2,width//2)

        # Process y
        for X in range(0,height,8):
            for Y in range(0,width,8):
                block_xy = y[X:X+8,Y:Y+8]
                block_uv = DCT_compute_1block(block_xy)
                list_block_uv.append(block_uv)

                temp_Q = abs(block_uv/P_lumi).max()/255
                if temp_Q>optimal_Q:
                    optimal_Q=temp_Q

        # Process u and v
        for img in [u,v]:
            for X in range(0,height//2,8):
                for Y in range(0,width//2,8):
                    block_xy = img[X:X+8,Y:Y+8]
                    block_uv = DCT_compute_1block(block_xy)
                    list_block_uv.append(block_uv)

                    temp_Q = abs(block_uv/P_chromi).max()/255
                    if temp_Q>optimal_Q:
                        optimal_Q=temp_Q
    
    # Quantization coeffs
    if Q == 'opti':
        Q=optimal_Q
        print("DCT: optimal Q as been set to ",optimal_Q*100,"% of the "\
              "psychovisual matrices values. ")
    if Q<optimal_Q:
        print("DCT error: quantization will not fit all the values in "\
              "[-255-255]. Q should be higher than ",optimal_Q*100,"%. Min"\
              "value selected instead.")
        Q=optimal_Q
    if Q>1.99:
        print("DCT error: maximum Q value is 199")
        Q = 1.99

    list_encoded_uv = [np.array([int(np.ceil(Q*128))])]

    # Quantization, diagonal reading and runlength encode
    current_block_index = 0
    for dummy0 in range(nb_frames):
        # Process y
        for dummy_1 in range(height//8*width//8):
            block = list_block_uv[current_block_index]
            quantized_block = np.array(np.round(block/(P_lumi*Q)),dtype=int)
            quantized_vector= quantized_block.flatten()[reading_order]
            encoded_uv = Runlength_encode(quantized_vector)
            
            current_block_index += 1
            list_encoded_uv.append(encoded_uv)

        # Process u and v
        for dummy_2 in ['u','v']:
            for dummy_3 in range(height//16*width//16):
                block = list_block_uv[current_block_index]
                quantized_block = np.array(np.round(block/(P_chromi*Q)),dtype=int)
                quantized_vector= quantized_block.flatten()[reading_order]
                encoded_uv = Runlength_encode(quantized_vector)

                current_block_index += 1
                list_encoded_uv.append(encoded_uv)

    data_uv = np.concatenate(list_encoded_uv)

    return data_uv


def DCT_inverse(data_uv):
    """ Reverse-compute the DCT, see the doc of 'DCT_compute' to see wich
    operations are reversed.

    Args:
        data_uv (array(nb_frames,-1))
    Return:
        data_xy (array(nb_frames,-1))
    """
    block_per_frame = height//8*width//8+2*height//16*width//16
    block_in_y      = height//8*width//8
    frame_size      = width*height+2*width//2*height//2
    
    data_uv = np.array(data_uv).flatten()
    
    block_start_id   = 1
    block_end_id     = block_start_id+1
    Q = data_uv[0]/128

    list_block_xy    = []
    current_block_index = 0
    pbar = tqdm(total=data_uv.size,desc="DCT inverse")
    block_end_id_prev = 0
    # Runlengh decode, reconstruct block, reverse quantification, reverse DCT
    while block_end_id < data_uv.size:
        block_end_id += 2
        if data_uv[block_end_id-1]==0:
            pbar.update(block_end_id-block_end_id_prev)
            block_end_id_prev = block_end_id
            encoded_uv = data_uv[block_start_id:block_end_id]
            quantized_vector     = Runlength_decode(encoded_uv)
            quantized_block   = quantized_vector[diag_reading].reshape((8,8))
            
            if current_block_index%block_per_frame < block_in_y:
                block_uv = quantized_block*P_lumi*Q
            else:
                block_uv = quantized_block*P_chromi*Q
            
            block_xy = DCT_inverse_1block(block_uv)
            list_block_xy.append(block_xy)

            block_start_id = block_end_id
            block_end_id  += 1
            current_block_index+=1
    pbar.close()

    # Check that there is no error with the number of blocks
    if (current_block_index)%block_per_frame != 0:
        print("Error DCT_inverse: inconsistente number of 8x8 blocks, last "\
              "frame countain "+str((current_block_index+1)%block_per_frame)+\
              " blocks")

    # Reconstruct the frames
    nb_blocks = current_block_index
    nb_frames = nb_blocks//block_per_frame
    data_xy   = np.zeros((nb_frames,frame_size),dtype=int)

    for k in range(nb_frames):
        frame_block = list_block_xy[k*block_per_frame:(k+1)*block_per_frame]

        y_blocks = np.array(frame_block[:block_in_y])
        y_blocks.shape = (height//8,width//8,8,8)
        data_xy[k,:height*width] = y_blocks.swapaxes(1,2).flatten()

        u_blocks = np.array(frame_block[block_in_y:block_in_y+block_in_y//4])
        u_blocks.shape = (height//8//2,width//8//2,8,8)
        u_blocks = u_blocks.swapaxes(1,2).flatten()
        data_xy[k,height*width:height*width+height*width//4] = u_blocks

        v_blocks = np.array(frame_block[-block_in_y//4:])
        v_blocks.shape = (height//8//2,width//8//2,8,8)
        data_xy[k,-height*width//4:] = v_blocks.swapaxes(1,2).flatten()

    #data_xy += 128
    data_xy[data_xy<0]=0
    data_xy[data_xy>255]=255

    return data_xy

def DCT_compute_1block(block_xy):
    """ Compute the DCT for one bloc. """

    block_uv = np.zeros((8,8),dtype=int)
    for u in range(8):
        for v in range(8):
            block_uv[u,v] = np.sum(block_xy*vect_uv[u,v,:,:])

    return block_uv

def DCT_inverse_1block(block_uv):
    """ Reverse-compute the DCT for one block. """

    block_xy = np.zeros((8,8),dtype=float)
    for x in range(8):
        for y in range(8):
            block_xy[x,y] = int(np.round(np.sum(block_uv*vect_uv[:,:,x,y])))

    return block_xy

def Runlength_encode(data):
    """ Encode pairs of (nb_zeros, non_zero_value) and add a (0,0) when there
    are no non zeros values left. """
    if data.size != 64:
        print("Runlength_encode error: expected data.size==64, got ",data.size)
    res = [data[0]]
    zeros_counter = 0
    for value in data[1:]:
        if value==0:
            zeros_counter+=1
        else:
            res.append(zeros_counter)
            res.append(value)
            zeros_counter =0

    # End of non-zeros values
    res.append(0)
    res.append(0)

    return np.array(res)

def Runlength_decode(data):
    """ Decode pairs of (nb_zeros, non_zero_value) with an initial coeff """
    if data.size%2 != 1:
        print("Runlength_decode error: expected data.size%2==1")
    res = np.zeros(64,dtype=int)
    res[0]=data[0]
    current_pos = 1
    for k in range(1,data.size-2,2):
        nb_zeros,value = data[k],data[k+1]
        current_pos+=nb_zeros
        res[current_pos]=value
        current_pos+=1
    return np.array(res)