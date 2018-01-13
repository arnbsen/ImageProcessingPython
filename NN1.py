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
    img1 = misc.imread(im1).astype(float)
    img2 = misc.imread(im2).astype(float)
    lbl = misc.imread(lb).astype(float).tolist()
    dif = abs(img1 - img2) / 255.0
    r = dif.shape
    dif = np.pad(dif, mode='reflect', pad_width=2)
    dif = dif.tolist()
    return dif, lbl, r


def assignIntWeights():
    wInpL1 = assignRandomWeight(5, 1)
    wL1ToL2 = assignRandomWeight(10, 5)
    wL2toL3 = assignRandomWeight(10, 10)
    wL3toL4 = assignRandomWeight(5, 10)
    wL4toOut = assignRandomWeight(1, 5)
    return wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut



def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            window.append(arr[i + h - 1][j + w - 1])
    return window

def BackPropagationOutput(dif, r, wv):
    (wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut) = wv
    oInpL1 = []
    # Computing Output of the neuron directly connected to Inputs
    for i in range(1, r[0] + 1):
        oTemp = []
        for j in range(1, r[1] + 1):
            temp = []
            for k in range(len(wInpL1)):
                temp = temp + [neuronSigmoid(dif[i][j], wInpL1[k])]
        oInpL1.append(oTemp)
    # Computing Layer 1 to Layer 2
    oL1toL2 = []
    for i in range(len(oInpL1)):
        oTemp = []
        for j in range(len(oInpL1[0])):
            temp =[]
            for k in range(len(wL1ToL2)):
                temp = temp + [neuronSigmoid(oInpL1[i][j], wL1ToL2[k])]
            oTemp.append(temp)
        oL1toL2.append(oTemp)
    # Computing Layer 2 to Layer 3
    oL2toL3 = []
    for i in range(len(oL1toL2)):
        oTemp = []
        for j in range(len(oL1toL2[0])):
            temp = []
            for k in range(len(wL2toL3)):
                temp = temp + [neuronSigmoid(oL1toL2[i][j], wL2toL3[k])]
            oTemp.append(temp)
        oL2toL3.append(oTemp)
    # Computing Layer 3 to Layer 4
    oL3toL4 = []
    for i in range(len(oL2toL3)):
        oTemp = []
        for j in range(len(oL2toL3[0])):
            temp = []
            for k in range(len(wInpL1)):
                temp = temp + [neuronSigmoid(oL2toL3[i][j], wL3toL4[k])]
            oTemp.append(temp)
        oL3toL4.append(oTemp)

    # Computing Layer 4 to Layer Output
    oL4toOut = []
    for i in range(len(oL3toL4)):
        oTemp = []
        for j in range(len(oL3toL4[0])):
            oTemp.append(neuron(oL1toL2[i][j], wL4toOut[0], 1))
        oL4toOut.append(oTemp)
    weightVector = (wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut)
    return weightVector, oInpL1, oL1toL2, oL4toOut, r




def BackPropagationSinglePoint(i, j, img1, img2, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toOut, l_rate, alpha):
    (wInpL1, wL1ToL2, wL2toOut) = wv
    t1 = (lbl[i][j] - oL2toOut[i][j][0]) * ReLuDer(oL2toOut[i][j][0])
    t2 = (lblInv[i][j] - oL2toOut[i][j][1]) * ReLuDer(oL2toOut[i][j][1])
    dOut = [t1, t2]
    dL2 = []
    for k in range(len(wL2toOut[0])):
        s = 0
        for h in range(len(wL2toOut)):
            s = s + wL2toOut[h][k] * dOut[h]
        dL2.append(s * ReLuDer(oL1toL2[i][j][k]))
    # print(dL2)
    # d for the first layer
    dL1 = []
    for k in range(len(wL1ToL2[0])):
        s = 0
        for h in range(len(wL1ToL2)):
            s = s + wL1ToL2[h][k] * dL2[h]
        d = s * SigmoidDer(oInpL1[i][j][k])
        dL1.append(d)
    # print(dL1)
    # Updating weights of the input layer
    retWin = [img1[i][j], img2[i][j]]
    for row in range(len(wInpL1)):
        for col in range(len(wInpL1[0])):
            delta = dL1[row] * retWin[col] * l_rate
            wInpL1[row][col] = (wInpL1[row][col] + delta * alpha)
    # Updating the weights at L2
    for row in range(len(wL1ToL2)):
        for col in range(len(wL1ToL2[0])):
            delta = dL2[row] * oInpL1[i][j][col] * l_rate
            wL1ToL2[row][col] = (wL1ToL2[row][col] + delta * alpha)

    # Updating weight of last layer
    for k in range(len(wL2toOut)):
        for h in range(len(wL2toOut[0])):
            delta = dOut[k] * oL1toL2[i][j][h] * l_rate
            wL2toOut[k][h] = (wL2toOut[k][h] + alpha * delta)
    wv = (wInpL1, wL1ToL2, wL2toOut)
    # print(weightVector)
    return wv

