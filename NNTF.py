import tflearn
from tensorflow import DType
import numpy as np
from scipy import misc
import math


def convertImageToBinary(lbl):
    binImage = []
    binImageInv = []
    for i in range(0, len(lbl)):
        tmp = []
        tmp2 = []
        for j in range(0, len(lbl[0])):
            if lbl[i][j] == 0:
                tmp.append(0)
                tmp2.append(1)
            else:
                tmp.append(1)
                tmp2.append(0)
        binImage.append(tmp)
        binImageInv.append(tmp2)
    return binImage, binImageInv


def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            try:
                window.append(arr[i + h - 1][j + w - 1])
            except IndexError:
                print("i = ", i, " j = ", j)
    return window


def prepareData(im1, im2, lb):
    # dif = abs(misc.imread(im1).astype(float)/255.0 - misc.imread(im2).astype(float)/255.0)
    img1 = (misc.imread(im1).astype(float) / 255.0).tolist()
    img2 = (misc.imread(im2).astype(float) / 255.0).tolist()
    # dif = np.pad(dif, pad_width=10, mode='reflect').tolist()
    lbl = misc.imread(lb).tolist()
    (binImage, binImageInv) = convertImageToBinary(lbl)
    binImage = np.array(binImage).astype(float).ravel().tolist()
    binImageInv = np.array(binImageInv).astype(float).ravel().tolist()
    data = []
    labels = []
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            data = data + [[img1[i][j], img2[i][j]]]
            labels = labels + [np.array([binImage[i + j], binImageInv[i + j]])]
    r = (len(img1), len(img1[0]))
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, r


def prepareClusterData(im1, lb):
    print("Preparing Data. Please Wait...")
    dif = np.pad(misc.imread(im1).astype(float) / 255.0, mode='reflect', pad_width=(2, 2)).tolist()
    print(len(dif), len(dif[0]))
    binImage = (misc.imread(lb).astype(float) / 255.0).ravel()
    binImageInv = abs(np.full(binImage.shape, 1.0) - binImage).tolist()
    data = []
    labels = []
    for i in range(2, len(dif) - 2):
        for j in range(2, len(dif[0]) - 2):
            data = data + [windowCreator(i, j, 3, 3, dif)]
            labels = labels + [[binImage[i + j], binImageInv[i + j]]]
    r = (len(dif), len(dif[0]))
    return data, labels, r


def prepareDataAlternateNorm(im1, im2, lb):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).ravel().tolist()
    min1 = min(img1)
    max1 = max(img1)
    img2 = misc.imread(im1).ravel().tolist()
    min2 = min(img2)
    max2 = max(img2)
    binImage = (misc.imread(lb).astype(float) / 255.0).ravel()
    binImageInv = abs(np.full(binImage.shape, 1.0) - binImage).tolist()
    img1 = np.array(img1)
    img2 = np.array(img2)
    # Normalisation of data
    img1 = ((img1 - np.full(img1.shape, min1)) / (max1 - min1))
    img2 = ((img2 - np.full(img2.shape, min2)) / (max2 - min2))
    img1 = img1.tolist()
    img2 = img2.tolist()
    data = []
    labels = []
    for i in range(len(img1)):
        data = data + [[abs(img2[i] - img1[i])]]
        labels = labels + [np.array([binImage[i], binImageInv[i]])]
    data = np.array(data)
    labels = np.array(labels)
    return data, labels, r


def prepareDataInter(im1, im2, lb):
    img1 = misc.imread(im1).astype(float)
    r = img1.shape
    img2 = misc.imread(im2).astype(float)
    img1 = np.pad(img1, pad_width=(1, 1), mode='reflect').tolist()
    img2 = np.pad(img2, pad_width=(1, 1), mode='reflect').tolist()
    binImage = misc.imread(lb).astype(int)
    binImageInv = abs(np.full(binImage.shape, 255) - binImage).tolist()
    binImage = binImage.tolist()
    data = []
    labels = []
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            data = [[img1[i - 1][j - 1], img2[i - 1][j - 1], img1[i - 1][j], img2[i - 1][j], img1[i - 1][j + 1],
                     img2[i - 1][j + 1], img1[i][j - 1], img2[i][j - 1], img1[i][j], img2[i][j],
                     img1[i][j], img2[i][j + 1], img1[i  + 1][j - 1], img2[i + 1][j - 1], img1[i + 1][j], img2[i + 1][j],
                     img1[i + 1][j + 1], img2[i + 1][j + 1]]]
            labels = [[binImage[i][j] , binImageInv[i][j]]]
    return data, labels, r

def netInit():
    print("Initialising the net......")
    net = tflearn.input_data(shape=[None, 18])
    net = tflearn.fully_connected(net, 18, activation='relu')
    net = tflearn.fully_connected(net, 15, activation='relu')
    net = tflearn.fully_connected(net, 10, activation='relu')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.00001, loss='categorical_crossentropy')
    model = tflearn.DNN(net)
    return model


def trainData(model, data, labels):
    model.fit(data, labels, show_metric=True, n_epoch=10000)
    return model


def writeImage(model, data, r):
    output = model.predict(data)
    output = output.tolist()
    temp = []
    for i in range(1, r[0]+1):
        itemp = []
        for j in range(1, r[1]+1):
            if output[i + j][0] < output[i + j][1]:
                itemp = itemp + [0]
            else:
                itemp = itemp + [255]
        temp = temp + [itemp]
    temp = np.array(temp).astype(np.uint8)
    misc.imsave('disaster.png', temp)


im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell1small.png'
in2 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell2small.png'
l = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/gtsmall.png'

print("Training with two examples")
im2 = in2  # +'0882.png'
lb = l  # + '0882.png'

(data, labels, r) = prepareDataAlternateNorm(im1, im2, lb)
model = netInit()
model = trainData(model, data, labels)
writeImage(model, data, r)
