import numpy as np

# Format for qcif images
width = 176
height = 144


class Video:
    """ Class for reading qcif video image by image 

    Example:
    |   video = Video(fname)
    |
    |   if not video.EOF:
    |       compY, compU, compV = video.read_img()  # Get 
    |       plt.figure(figsize=(20,10))
    |       plt.subplot(1,2,1)
    |       plt.imshow(compY)
    |       plt.subplot(2,4,3)
    |       plt.imshow(compU)
    |       plt.subplot(2,4,7)
    |       plt.imshow(compV)
    |       plt.show()
    """
    
    def __init__(self,fname):
        """ fname (string) : file name of the video to read """
        self.fname = fname
        self.fd    = open(fname,mode='r+b')
        self.file_size = self.fd.seek(0,2)
        self.fd.seek(0,0)
        self.EOF = self.file_size == self.fd.tell()

    def read_img(self):
        """ Get the 3 components of the next image """
        compY = np.frombuffer(self.fd.read(width*height)      , dtype=np.uint8).reshape((height,width))
        compU = np.frombuffer(self.fd.read(width//2*height//2), dtype=np.uint8).reshape((height//2,width//2))
        compV = np.frombuffer(self.fd.read(width//2*height//2), dtype=np.uint8).reshape((height//2,width//2))

        compY = np.array(compY,dtype=int)
        compU = np.array(compU,dtype=int)
        compV = np.array(compV,dtype=int)

        self.EOF = self.fd.tell() == self.file_size
        return compY, compU, compV
    
    def restart(self):
        """ Go back to the start of the file (first image) """
        self.fd.seek(0)
    
    def close():
        """ Close file when work is done """
        self.fd.close()

def read_qcif(fname,nb_frame='all'):
    """ Read a qcif video and store it in a numpy array.
    Args:
        fname (string) : file name of the video to read.
        nb_frame (int or 'all') : number of frames to read, default is all the
            frames.
    Return:
        frames (array(nb_frame,frame_size)) : y,u and v concatenated for each
            frame. The frame size is fixed by the qcif standard.
    """
    video = Video(fname)
    frames = []
    if nb_frame == 'all':
        nb_frame = 10000
    for _ in range(nb_frame):
        if not video.EOF: # and n<2:
            y,u,v = video.read_img()
            frames.append(np.concatenate((y.flatten(),u.flatten(),v.flatten())))
    frames = np.array(frames)
    return frames

def save_qcif(fname,data):
    """ Save a qcif video stored in a numpy array.
    Args:
        fname (string) : file name of the video to read.
        nb_frame (int or 'all') : number of frames to read, default is all the
            frames.
    """
    m,M = data.min(),data.max()
    if m<0 or M>255:
        print("save_qcif warning: invalid values range, can not convert"\
              "data without loss. Expected range is 0|255, got "+str(m)+"|"\
              +str(M)+".")

    res = np.array(data,dtype=np.uint8)
    res.tofile(fname)