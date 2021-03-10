import numpy as np
from readImage import Video,height,width


def Huffman_code(proba_table, tree_only=False):
    """ Compute the Huffman code table for the given distribution """
    n=len(proba_table)
    
    # Compute the tree
    groups_symbols = [k for k in range(0,n)]
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
    code_table = ['0']*n
    
    def visit_node(prefix,node):
        if type(node)==int: # Leaf
            code_table[node] = prefix
        else: # type == tupl, node
            visit_node(prefix+'0',node[0])
            visit_node(prefix+'1',node[1])
            
    visit_node('',tree)
    
    return code_table


def Huffman_encode(data):
    """ Encode a list of symbols, return a stream of bit. Data is expected
    to be an array_like of uint8.
    Bit stream format :
    | (1) uint8 : proba table len <n>
    | (n) uint8 : proba of each symbol (proba is equal to <value>/255)
    | (remain) bit : code elements
    """
    
    bitstream = ''
    data = np.array(data)
    
    if data.max()>255 or data.min()<0:
        print("Huffman_encode error: data value outside of range [0,255] ")
    
    n = data.max()+1
    bitstream+=format(n-1,"b").zfill(8)
    
    # Compute the proba table
    proba_table,_=np.histogram(data,bins=np.arange(-0.5,n+0.5),density=True)
    proba_table = np.array(np.round(proba_table/proba_table.max()*255),dtype=int)
    for proba in proba_table:
        bitstream+=format(proba,"b").zfill(8)
    
    # Compute the code table
    code_table = Huffman_code(proba_table)
    
    # Convert symbols into a stream of bits
    for symbol in data:
        bitstream+=code_table[symbol]
    
    return bitstream


def Huffman_decode(bitstream):
    """ Decode a stream of bits, return a list of symbols. bitstream is
    expected to be a string composed of '0' and '1'.
    Bit stream format :
    | (1) uint8 : proba table len <n>
    | (n) uint8 : proba of each symbol (proba is equal to <value>/255)
    | (remain) bit : code elements
    """
    
    n = int(bitstream[:8], base=2)+1
    
    # Decode the proba table
    proba_table = [int(bitstream[k*8:(k+1)*8], base=2) for k in range(1,n+1)]
        
    # Compute the code table
    Huffman_tree = Huffman_code(proba_table, tree_only=True)
    
    # Convert the stream of bits into symbols
    symbols_list = []
    position = (n+1)*8
    node = Huffman_tree
    while position < len(bitstream):
        node = node[int(bitstream[position])]
        if type(node)==int: # Leaf
            symbols_list.append(node)
            node = Huffman_tree
        position += 1

    return symbols_list