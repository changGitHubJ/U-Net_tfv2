import numpy as np

class MyLoadData:
    IMG_SIZE = 0
    OUTPUT_SIZE = 0

    def __init__(self, IMG_SIZE, channels):
        self.IMG_SIZE = IMG_SIZE
        self.channels = channels

    def readImages(self, filename, DATA_SIZE):
        images = np.zeros((DATA_SIZE, self.IMG_SIZE, self.IMG_SIZE), dtype=np.float)
        fileImg = open(filename)
        for k in range(DATA_SIZE):
            line = fileImg.readline()
            if(not line):
                break
            val = line.split(',')
            for i in range(self.IMG_SIZE):
                for j in range(self.IMG_SIZE):
                    images[k, i, j] = float(val[self.IMG_SIZE*i + j + 1])
        return images/127.5 - 1
    
    def readLabels(self, filename, DATA_SIZE):
        images = np.zeros((DATA_SIZE, self.IMG_SIZE, self.IMG_SIZE, self.channels), dtype=np.float)
        fileImg = open(filename)
        for k in range(DATA_SIZE):
            line = fileImg.readline()
            if(not line):
                break
            val = line.split(',')
            for i in range(self.IMG_SIZE):
                for j in range(self.IMG_SIZE):
                    for n in range(self.channels):
                        images[k, i, j, n] = float(val[self.channels*(self.IMG_SIZE*i + j) + n + 1])
        return images