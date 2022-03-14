import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from matplotlib import cm

import common as c
import load_data as data

dir = "u-net"

def readImages(filename, channel):
    images = np.zeros((c.TEST_DATA_SIZE*c.CATEGORY, c.IMG_SIZE, c.IMG_SIZE, channel))
    fileImg = open(filename)
    for k in range(c.TEST_DATA_SIZE*channel):
        line = fileImg.readline()
        if(not line):
            break
        val = line.split(',')
        for i in range(c.IMG_SIZE):
            for j in range(c.IMG_SIZE):
                for n in range(channel):
                    images[k, i, j, n] = float(val[channel*(c.IMG_SIZE*i + j) + n + 1])
    return images

def main(data):
    with tf.device("/gpu:0"):
        filepath = [dir + "/data/testImage256_100.txt"]
        print("reading images...")
        test_image = data.readImages(filepath[0], c.TEST_DATA_SIZE*c.CATEGORY)
        print("loading model...")
        loaded = tf.saved_model.load(dir + "/output")
        infer = loaded.signatures["serving_default"]
        for i in range(c.TEST_DATA_SIZE*c.CATEGORY):
            x = test_image[i, :, :].reshape([1, c.IMG_SIZE, c.IMG_SIZE, 1]).astype(np.float32)
            inferred = infer(tf.convert_to_tensor(x))["conv2d_22"].numpy()
            plt.figure(figsize=[15, 6])
            plt.subplot(2, 6, 1)
            fig = plt.imshow(test_image[i, :, :].reshape([c.IMG_SIZE, c.IMG_SIZE]))
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            for j in range(c.CATEGORY):
                plt.subplot(2, 6, c.CATEGORY + j + 1)
                fig = plt.imshow(inferred[0, :, :, j].reshape([c.IMG_SIZE, c.IMG_SIZE]), vmin=0.0, vmax=1.0)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)  
            plt.show()
            # plt.savefig("%d.png"%i)

if __name__=='__main__':
    data = data.MyLoadData(c.IMG_SIZE, c.CATEGORY)
    main(data)