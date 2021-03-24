import numpy as np

def benchmark(original,compressed):
    nb_symbols = original.size
    nb_bit = len(compressed)
    print("B/symb: ", nb_bit/nb_symbols)
    print("Original size (MiB): ", nb_symbols/1024**2)
    print("Compressed size (%): ", nb_bit/8/nb_symbols*100)