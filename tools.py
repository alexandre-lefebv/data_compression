import numpy as np

from tqdm import tqdm, trange

def benchmark(original,processed,compressed):
    nb_symbols = original.size
    nb_bit = len(compressed)
    print("MSE: ", np.mean((original-processed)**2))
    print("B/symb: ", nb_bit/nb_symbols)
    print("Original size (MiB): ", nb_symbols/1024**2)
    print("Compressed size (%): ", nb_bit/8/nb_symbols*100)