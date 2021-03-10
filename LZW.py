import numpy as np
from readImage import Video,height,width


def LZW_encode(data):
    """ Encode a list of symbols, return a stream of bit. Data is expected
    to be an array_like of uint8.
    Bit stream format :
    | (1) uint<8> : codewords length, in number of additional bits.
    |               -> codewords length = 8 + <value>
    | (n) uint<k> : <n> codewords of size <k> bits.
    """
    
    if data.max()>255 or data.min()<0:
        print("LZW_encode error: data value outside of range [0,255] ")
    
    current_codeword = 255
    omega = (data[0],)
    dictionary = {(k,):format(k,"b") for k in range(256)}
    words_list = []

    # Fill the dictionary, list the words
    for K in data[1:]:
        new_omega = omega+(K,)
        
        if not new_omega in dictionary:
            if len(new_omega)==1:
                dictionary[new_omega]=format(new_omega[0],"b")
            else:
                current_codeword+=1
                dictionary[new_omega]=format(current_codeword,"b")
            
            words_list.append(dictionary[omega])
            omega = (K,)
        else:
            omega = new_omega
    words_list.append(dictionary[omega])
    
    # Complete the codewords so that they all have the same length
    nb_bits = int(np.log2(len(dictionary)-1))+1
    bitstream = format(nb_bits-8,"b").zfill(8)
    for word in words_list:
        bitstream += word.zfill(nb_bits)
    
    return bitstream


def LZW_decode(bitstream):
    """ Decode a stream of bits, return a list of symbols. bitstream is
    expected to be a string composed of '0' and '1'.
    Bit stream format :
    | (1) uint<8> : codewords length, in number of additional bits.
    |               -> codewords length = 8 + <value>
    | (n) uint<k> : <n> codewords of size <k> bits.
    """
    codeword_sz = int(bitstream[:8], base=2)+8
    reverse_dict = {format(k,"b").zfill(codeword_sz):[k] for k in range(0,256)}
    symbols_list = list(reverse_dict[bitstream[8:8+codeword_sz]])
    current_codeword = 255
    omega_prev = reverse_dict[bitstream[8:8+codeword_sz]]
    
    # Fill the dictionary and decode
    for k in range(8+codeword_sz,len(bitstream),codeword_sz):
        codeword = bitstream[k:k+codeword_sz]
        
        # Decode
        if not (codeword in reverse_dict):
            omega = omega_prev+omega_prev[0:1]
        else:
            omega = list(reverse_dict[codeword])
        symbols_list += omega
        
        # Update the dictionary
        current_codeword+=1
        reverse_dict[format(current_codeword,"b").zfill(codeword_sz)]=omega_prev+omega[0:1]
        
        omega_prev = list(omega)
    
    return symbols_list