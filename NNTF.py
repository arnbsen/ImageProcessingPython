import tflearn
from supportLibrary import *
import numpy as np
from scipy import misc


def netInit():
    print("Initialising the net......")
    net = tflearn.input_data(shape=[None, 18])
    net = tflearn.fully_connected(net, 36, activation='sigmoid')
    net = tflearn.fully_connected(net, 18, activation='sigmoid')
    net = tflearn.fully_connected(net, 9, activation='sigmoid')
    net = tflearn.fully_connected(net, 5, activation='sigmoid')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', loss='binary_crossentropy', learning_rate=0.006)
    model = tflearn.DNN(net)
    return model


def trainData(model, data, labels):
    model.fit(data, labels, show_metric=True, n_epoch=10000)
    return model


im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell1small.png'
in2 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell2small.png'
l = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/gtsmall.png'

print("Training with two examples")
im2 = in2  # +'0882.png'
lb = l  # + '0882.png'

(data, labels, r) = prepareDataInter(im1, im2, lb)
model = netInit()
model = trainData(model, data, labels)
writeImage(model, data, r)
