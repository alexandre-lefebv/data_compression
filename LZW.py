import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange


def LZW_encode(data):
    """ Encode a list of symbols, return a stream of bit. Data is expected
    to be an array_like of int.
    Bit stream format :
    | (1) sign+uint8 : min symbol
    | (1) uint9 : range between min and max symbol in data (data.max-data.min+1)
    | (1) uint<8> : codewords length in number of bits
    | (n) uint<k> : <n> codewords of size <k> bits.
    """
    
    bitstream = ''
    data = np.array(data)
    
    if data.max()>255 or data.min()<-255:
        print("LZW_encode error: data value outside of range [-255,255] ")
        print(data.min(),data.max())
    
    m = data.min()
    nb_symbols = data.max()-m+1
    bitstream+=format(m<0,"b")                  # Sign of m
    bitstream+=format(abs(m),"b").zfill(8)      # 0,+255
    bitstream+=format(nb_symbols,"b").zfill(9)  # 0,511

    current_codeword = nb_symbols
    omega = (data[0]-m,)
    dictionary = {(k,):format(k,"b") for k in range(nb_symbols)}
    words_list = []

    #################
    #width = 176
    #height = 144
    #frame_id = 0
    #symbol_counter = 1
    #frame_size = width*height + 2*width//2*height//2
    #lengths = np.zeros(data.size//frame_size)
    #################
    
    # Fill the dictionary, list the words
    for k in tqdm(data[1:],desc="Building the dictionary"):
        K = k-m
        new_omega = omega+(K,)
        
        if not new_omega in dictionary:
            if len(new_omega)==1:
                dictionary[new_omega]=format(new_omega[0],"b")
            else:
                current_codeword+=1
                dictionary[new_omega]=format(current_codeword,"b")
            
            words_list.append(dictionary[omega])
            
            ###
            #symbol_counter += len(omega)
            #lengths[frame_id] += 1
            #frame_id += symbol_counter//frame_size
            #symbol_counter %= frame_size
            ###
            
            omega = (K,)
        else:
            omega = new_omega
    words_list.append(dictionary[omega])
    
    # Complete the codewords so that they all have the same length
    codeword_sz = int(np.log2(len(dictionary)-1))+1
    bitstream += format(codeword_sz,"b").zfill(8)
    for word in tqdm(words_list,desc="Encoding"):
        bitstream += word.zfill(codeword_sz)
    
    
    #plt.plot(lengths*codeword_sz)
    
    return bitstream


def LZW_decode(bitstream):
    """ Decode a stream of bits, return a list of symbols. bitstream is
    expected to be a string composed of '0' and '1'.
    Bit stream format :
    | (1) sign+uint8 : min symbol
    | (1) uint9 : range between min and max symbol in data (data.max-data.min+1)
    | (1) uint<8> : codewords length in number of bits
    | (n) uint<k> : <n> codewords of size <k> bits.
    """
    sign = int(bitstream[0], base=2)
    m =    int(bitstream[1:9], base=2)*((-1)**sign)
    nb_symbols = int(bitstream[9:18], base=2)
    codeword_sz = int(bitstream[18:26], base=2)
    reverse_dict = {format(k,"b").zfill(codeword_sz):[k+m] for k in range(nb_symbols)}
    symbols_list = list(reverse_dict[bitstream[26:26+codeword_sz]])
    current_codeword = nb_symbols
    omega_prev = reverse_dict[bitstream[26:26+codeword_sz]]
    # Fill the dictionary and decode
    for k in trange(26+codeword_sz,len(bitstream),codeword_sz,desc="Decoding"):
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
    
    return np.array(symbols_list)