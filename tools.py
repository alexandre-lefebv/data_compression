import numpy as np
import matplotlib.pyplot as plt

width = 176
height = 144
frame_size = width*height + 2*width//2*height//2

def compression_benchmark(original, compressed):
    nb_symbols = original.size
    nb_bit = len(compressed)
    print("B/symb: ", nb_bit/nb_symbols)
    print("Original size (MiB): ", nb_symbols/1024**2)
    print("Compressed size (%): ", nb_bit/8/nb_symbols*100)
    
def quality_benchmark(original, decompressed):
    nb_frames = original.size//frame_size
    original_frames = original.reshape(nb_frames,frame_size)
    decompressed_frames = decompressed.reshape(nb_frames,frame_size)
    mse_per_frame = np.mean((original_frames-decompressed_frames)**2,axis=1)
    plt.figure()
    plt.title("MSE per frame")
    plt.xlabel("Frame index")
    plt.ylabel("MSE")
    plt.plot(range(nb_frames), mse_per_frame)
    plt.show()