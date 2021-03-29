import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange


def Huffman_code(min_symbol, proba_table, tree_only=False):
    """ Compute the Huffman code table for the given distribution """
    n=len(proba_table)
    
    # Compute the tree
    groups_symbols = list(range(min_symbol,min_symbol+n))
    groups_proba   = [proba_table[k] for k in range(0,n)]
    for _ in range(n-1):
        id_a = np.argmin(groups_proba)
        symbol_a = groups_symbols.pop(id_a)
        proba_a  = groups_proba.pop(id_a)
        id_b = np.argmin(groups_proba)
        symbol_b = groups_symbols.pop(id_b)
        proba_b  = groups_proba.pop(id_b)
        groups_symbols.append((symbol_a,symbol_b))
        groups_proba.append(proba_a+proba_b)
    tree = groups_symbols[0]
    
    if tree_only:
        return tree
        
    # Go trough the tree to get the code table
    code_table = ['']*n
    
    def visit_node(prefix,node):
        if type(node)==int: # Leaf
            code_table[node-min_symbol] = prefix
        else: # type == tupl, node
            visit_node(prefix+'0',node[0])
            visit_node(prefix+'1',node[1])
            
    visit_node('',tree)
    
    return code_table


def Huffman_encode(data):
    """ Encode a list of symbols, return a stream of bit. Data is expected
    to be an array_like of int.
    Bit stream format :
    | (1) sign+uint8 : min symbol
    | (1) uint9 : proba table len <n>
    | (n) uint8 : proba of each symbol (proba is equal to <value>/255)
    | (remain) bit : code elements
    """
    
    bitstream = ''
    data = np.array(data)
    
    if data.max()>255 or data.min()<-255:
        print("Huffman_encode error: data value outside of range [-255,255] ")
        print(data.min(),data.max())
    
    m = data.min()
    nb_symbols = data.max()-m+1
    bitstream+=format(m<0,"b")                  # Sign of m
    bitstream+=format(abs(m),"b").zfill(8)      # 0,+255
    bitstream+=format(nb_symbols,"b").zfill(9)  # 0,511
    
    # Compute the proba table
    proba_table,dummy=np.histogram(data,bins=np.arange(m-0.5,m+nb_symbols+0.5),density=True)
    proba_table = np.array(np.round(proba_table/proba_table.max()*255),dtype=int)
    for proba in proba_table:
        bitstream+=format(proba,"b").zfill(8)
    
    # Compute the code table
    code_table = Huffman_code(m, proba_table)
    
    # Convert symbols into a stream of bits
    for symbol in tqdm(data,desc="Encoding"):
        bitstream+=code_table[symbol-m]
    
    #################
    width = 176
    height = 144
    frame_size = width*height + 2*width//2*height//2
    lengths = np.zeros(data.size)
    for k,symbol in enumerate(data):
        lengths[k] = len(code_table[symbol-m])
    lengths = lengths.reshape(-1,frame_size).sum(axis=1)
    plt.plot(lengths)
    #################
    
    return bitstream


def Huffman_decode(bitstream):
    """ Decode a stream of bits, return a list of symbols. bitstream is
    expected to be a string composed of '0' and '1'.
    Bit stream format :
    | (1) sign+uint8 : min symbol
    | (1) uint9 : proba table len <n>
    | (n) uint8 : proba of each symbol (proba is equal to <value>/255)
    | (remain) bit : code elements
    """
    
    sign = int(bitstream[0], base=2)
    m    = int(bitstream[1:9], base=2)*((-1)**sign)
    nb_symbols = int(bitstream[9:18], base=2)

    # Decode the proba table
    proba_table = [int(bitstream[18+k*8:18+(k+1)*8], base=2)
                   for k in range(nb_symbols)]

    # Compute the code table
    Huffman_tree = Huffman_code(m, proba_table, tree_only=True)
    
    # Convert the stream of bits into symbols
    symbols_list = []
    position = 18+nb_symbols*8
    node = Huffman_tree
    for position in trange(18+nb_symbols*8,len(bitstream),desc="Decoding"):
        # Go down in the tree for each code word, proceed bit by bit
        node = node[int(bitstream[position])]
        if type(node)==int: # Leaf
            symbols_list.append(node)
            node = Huffman_tree
        position += 1

    return symbols_list