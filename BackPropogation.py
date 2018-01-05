# Back Propagation using Stochastic Gradient Descent
# Starting with 3 hidden layers of size (in) -> 5 -> 3 -> 1 -> (out)
# This library is for for change detection only


import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
import math
from PIL import Image
import os
import random


# Creating a simple neuron based on Threshold
def neuron(x, w, th):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    print(total)
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

def convertImageToBinary(img1):
    binImage = []
    for i in range(0, len(img1)):
        tmp = []
        for j in range(0, len(img1[0])):
            if img1[i][j] == 0:
                tmp.append(0)
            else:
                tmp.append(1)
        binImage.append(tmp)
    return binImage


def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            window.append(arr[i + h - 1][j + w - 1])
    return window


def assignRandomWeight(l, b):
    wv = []
    for i in range(l):
        wvi = []
        for j in range(b):
            wvi.append(random.uniform(0, 1))
        wv.append(wvi)
    return wv


def twoDimImread(im1, im2, lb):
    img1 = misc.imread(im1).astype(int)
    img2 = misc.imread(im2).astype(int)
    lbl = misc.imread(lb).astype(int).tolist()
    dif = abs(img1 - img2)/255
    r = dif.shape
    dif = np.pad(dif, mode='reflect', pad_width=2)
    dif = dif.tolist()
    return (dif, lbl, r)


def assignIntWeights():
    wInpL1 = assignRandomWeight(3, 9)
    wL1ToL2 = assignRandomWeight(2, 3)
    wL2toOut = assignRandomWeight(1, 2)
    return (wInpL1, wL1ToL2, wL2toOut)


def BackPropagationOutput(dif, r, weightVector):
    (wInpL1, wL1ToL2, wL2toOut) = weightVector
    oInpL1 = []
    # Computing Output of the neuron directly connected to Inputs
    for i in range(1, r[0] + 1):
        oTemp = []
        for j in range(1, r[1] + 1):
            retWin = windowCreator(i, j, 3, 3, dif)
            temp = [neuronReLu(retWin, wInpL1[0])] + [neuronReLu(retWin, wInpL1[1])] + [
                neuronReLu(retWin, wInpL1[2])]
            oTemp.append(temp)
        oInpL1.append(oTemp)
    # Computing Layer 1 to Layer 2
    oL1toL2 = []
    for i in range(len(oInpL1)):
        oTemp = []
        for j in range(len(oInpL1[0])):
            temp = [neuronReLu(oInpL1[i][j], wL1ToL2[0])] + [neuronReLu(oInpL1[i][j], wL1ToL2[1])]
            oTemp.append(temp)
        oL1toL2.append(oTemp)
    # Computing Layer 2 to Output
    oL2toOut = []
    for i in range(len(oL1toL2)):
        oTemp = []
        for j in range(len(oL1toL2[0])):
            oTemp.append(neuronReLu(oL1toL2[i][j], wL2toOut[0]))
        oL2toOut.append(oTemp)
    weightVector = (wInpL1, wL1ToL2, wL2toOut)

    return (weightVector, oInpL1, oL1toL2, oL2toOut, r)


def BackPropagationSinglePoint(i, j, dif, lbl, weightVector, oInpL1, oL1toL2, oL2toOut, l_rate):
    (wInpL1, wL1ToL2, wL2toOut) = weightVector
    # Computing error term for the pixel[i][j]
    # d for the output Layer
    dOut = (lbl[i][j] - ReLuDer(oL2toOut[i][j]))  # * oL2toOut[i][j]*(1 - oL2toOut[i][j])
    # d for the second layer
    # print(dOut)
    dL2 = []
    for k in range(len(wL2toOut[0])):
        s = 0
        for h in range(len(wL2toOut)):
            s = s + wL2toOut[h][k] * dOut
        dL2.append(s * ReLuDer(oL1toL2[i][j][k]))
    # print(dL2)
    # d for the first layer
    dL1 = []
    for k in range(len(wL1ToL2[0])):
        s = 0
        for h in range(len(wL1ToL2)):
            s = s + wL1ToL2[h][k]*dL2[h]
        d = s * ReLuDer(oInpL1[i][j][k])
        dL1.append(d)
    # print(dL1)

    # Updating weights of the input layer
    retWin = windowCreator(i + 1, j + 1, 3, 3, dif)
    for row in range(len(wInpL1)):
        for col in range(len(wInpL1[0])):
            delta = dL1[row] * retWin[col] * l_rate
            wInpL1[row][col] = (wInpL1[row][col] + delta * 0.7)

    # Updating the weights at L2
    for row in range(len(wL1ToL2)):
        for col in range(len(wL1ToL2[0])):
            delta = dL2[row] * oInpL1[i][j][col] * l_rate
            wL1ToL2[row][col] = (wL1ToL2[row][col] + delta * 0.7)

    # Updating weight of last layer
    for k in range(len(wL2toOut)):
        for h in range(len(wL2toOut[0])):
            delta = dOut * oL1toL2[i][j][h] * l_rate
            wL2toOut[k][h] = (wL2toOut[k][h] + 0.7 * delta)
    weightVector = (wInpL1, wL1ToL2, wL2toOut)
    # print(weightVector)
    return weightVector

def BackPropagation(dif, lbl, r ,noOfEpochs, weightVector, l_rate):
    print("Epoch 1 Processing")
    (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(dif, r, weightVector)
    for i in range(r[0]):
        for j in range(r[1]):
            weightVector = BackPropagationSinglePoint(i, j, dif, lbl, weightVector, oInpL1, oL1toL2, oL2toOut, l_rate)
    print(weightVector)
    for epoch in range(1,noOfEpochs):
        print("Epoch ",epoch+1," Processing",sep='')
        (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(dif, r, weightVector)
        for i in range(r[0]):
            for j in range(r[1]):
                weightVector = BackPropagationSinglePoint(i, j, dif, lbl, weightVector, oInpL1, oL1toL2, oL2toOut, l_rate)
        #print(weightVector)
    (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(dif, r, weightVector)

    return (weightVector ,oL2toOut)
def threshold(o,th):
    out = []
    for i in range(len(o)):
        t =[]
        for j in range(len(o[0])):
            if o[i][j] > th:
                t.append(1)
            else:
                t.append(0)
        out.append(t)
    return out
def errorCalc(output, lbl, r):
    cnt = 0
    total = r[0]*r[1]
    for i in range(r[0]):
        for j in range(r[1]):
            if lbl[i][j] != output[i][j]:
                cnt = cnt + 1
    return (cnt/total)*100

def BackPropagationClassifier(dif, r, weightVector):
    (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(dif, r, weightVector)
    return (oL2toOut, r)
