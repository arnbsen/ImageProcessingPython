# A second Approach to Neural Network (Has output Probabilities)
# Performs Stochastic Gradient Descent
# Neural Layer Structure (Input) -> 5 -> 3 -> (Output)
import statistics
import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
import math

import random


# Creating a simple neuron based on Threshold
def neuron(x, w, th):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    # print(total)
    if total > th:
        return 1
    else:
        return 0


# Creating a neuron based on the ReLu (Rectified Linear) function
def neuronReLu(x, w):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    return max(0, total)


def ReLuDer(x):
    if x > 0:
        return 1
    else:
        return 0


# Creating a neuron having Sigmoid function
def neuronSigmoid(x, w):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    return 1.0 / (1.0 + math.exp(-total))


def SigmoidDer(x):
    return x * (1.0 - x)


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


def assignRandomWeight(l, b):
    wv = []
    for i in range(l):
        wvi = []
        for j in range(b):
            wvi.append(random.uniform(0, 1))
        wv.append(wvi)
    return wv


def twoDimImread(im1, im2, lb):
    img1 = misc.imread(im1).astype(int)/255
    img2 = misc.imread(im2).astype(int)/255
    lbl = misc.imread(lb).astype(int).tolist()
    r = img1.shape
    img1 = np.pad(img1, mode='reflect', pad_width=2).tolist()
    img2 = np.pad(img2, mode='reflect', pad_width=2).tolist()
    return img1, img2, lbl, r


def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            window.append(arr[i + h - 1][j + w - 1])
    return window

def assignIntWeights():
    wInpL1 = assignRandomWeight(5, 1)
    wL1ToL2 = assignRandomWeight(10, 5)
    wL2toL3 = assignRandomWeight(5, 10)
    wL3toL4 = assignRandomWeight(3, 5)
    wL4toOut = assignRandomWeight(1, 3)
    return wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut

def pearsonCorrelationCoeff(img1, img2, r):
    for i in range(r[0]):
        for j in range(r[1]):

