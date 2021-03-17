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
