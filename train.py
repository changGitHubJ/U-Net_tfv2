import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow as tf
import time

from PIL import Image

import common as c
import load_data as data
import model

# Parameter
dir = "./u-net"
training_epochs = 200
batch_size = 12

def main(data, model, category, channel):
    print("Reading images...")
    x_train = data.readImages(dir + '/data/trainImage256_%d.txt'%category, c.TRAIN_DATA_SIZE*channel)
    x_test = data.readImages(dir + '/data/testImage256_%d.txt'%category, c.TEST_DATA_SIZE*channel)
    y_train = data.readLabels(dir + '/data/trainLabel256_%d.txt'%category, c.TRAIN_DATA_SIZE*channel)
    y_test = data.readLabels(dir + '/data/testLabel256_%d.txt'%category, c.TEST_DATA_SIZE*channel)
    
    print("Creating model...")
    model.create_model()

    print("Now training...")
    history = model.training(x_train, y_train, x_test, y_test)
    accuracy = history.train_acc
    loss = history.train_loss
    val_accuracy = history.val_acc
    val_loss = history.val_loss
    time = history.time
    
    # if not os.path.exists(dir + "/output"):
    #     os.mkdir(dir + "/output")
    model.save(dir + '/output') #/model_%d.h5'%category)
    
    with open(dir + "/output/training_log_%d.txt"%category, "w") as f:
        for i in range(training_epochs):
            line = "%f,%f,%f,%f,%f\n"%(loss[i], accuracy[i], val_loss[i], val_accuracy[i], time[i])
            f.write(line)

if __name__=='__main__':
    # args = sys.argv
    # CATEGORY = int(args[1])
    # channel = int(args[2])
    CATEGORY = 100
    channel = 6

    data = data.MyLoadData(c.IMG_SIZE, channel)
    model = model.MyModel((c.IMG_SIZE, c.IMG_SIZE, 1), channel, batch_size, training_epochs)
    main(data, model, CATEGORY, channel)