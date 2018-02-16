import statistics
import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
import math
from supportLibrary import normalisation, writeImage2, errorCalc
import random


# Creating a simple neuron based on Threshold
def neuron(x, w, th):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    # print(total)


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
    lbl = np.pad(misc.imread(lb).astype(float), pad_width=(1, 1), mode='reflect').tolist()
    r = img1.shape
    print(r)
    dif = abs(img1 - img2)
    return np.pad(dif, pad_width=(1, 1), mode='reflect'), lbl, r


def twoDimClusterImageRead(im1, lb):
    img1 = misc.imread(im1).astype(float)
    lbl = misc.imread(lb).astype(float).tolist()
    r = img1.shape
    img1 = np.pad(img1, pad_width=(1, 1), mode='reflect').tolist()
    return img1, lbl, r


def assignIntWeights():
    wInpL1 = assignRandomWeight(6, 9)
    wL1ToL2 = assignRandomWeight(5, 6)
    wL2toL3 = assignRandomWeight(4, 5)
    wL3toL4 = assignRandomWeight(3, 4)
    wL4toOut = assignRandomWeight(2, 3)
    return wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut


def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            try:
                window.append(arr[i + k - 1][j + l - 1])
            except IndexError:
                print("i = ", i, " j = ", j)
    return window


def BackPropagationOutput(dif, r, wv):
    (wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut) = wv
    oInpL1 = []
    # Computing Output of the neuron directly connected to Inputs
    for i in range(len(dif)):
        oTemp = []
        for j in range(len(dif[0])):
            temp = []
            for k in range(len(wInpL1)):
                temp = temp + [neuronSigmoid([dif[i][j]], wInpL1[k])]
            oTemp.append(temp)
        oInpL1.append(oTemp)
    # Computing Layer 1 to Layer 2
    oL1toL2 = []
    for i in range(len(oInpL1)):
        oTemp = []
        for j in range(len(oInpL1[0])):
            temp = []
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
            for k in range(len(wL3toL4)):
                temp = temp + [neuronSigmoid(oL2toL3[i][j], wL3toL4[k])]
            oTemp.append(temp)
        oL3toL4.append(oTemp)

    # Computing Layer 4 to Layer Output
    oL4toOut = []
    for i in range(len(oL3toL4)):
        oTemp = []
        for j in range(len(oL3toL4[0])):
            oTemp.append([neuronSigmoid(oL3toL4[i][j], wL4toOut[0]), neuronSigmoid(oL3toL4[i][j], wL4toOut[1])])
        oL4toOut.append(oTemp)
    weightVector = (wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut)
    return weightVector, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, r


def BackPropagationOutSinglePoint(i, j, img1, r, wv):
    (wInpL1, wL1ToL2, wL2toL3, wL3toL4, wL4toOut) = wv
    oInpL1 = []
    for k in range(len(wInpL1)):
        oInpL1 = oInpL1 + [neuronSigmoid(windowCreator(i, j, 3, 3, img1), wInpL1[k])]
    oL1toL2 = []
    temp = []
    for k in range(len(wL1ToL2)):
        oL1toL2 = oL1toL2 + [neuronSigmoid(oInpL1, wL1ToL2[k])]
    oL2toL3 = []
    for k in range(len(wL2toL3)):
        oL2toL3 = oL2toL3 + [neuronSigmoid(oL1toL2, wL2toL3[k])]
    oL3toL4 = []
    for k in range(len(wL3toL4)):
        oL3toL4 = oL3toL4 + [neuronSigmoid(oL2toL3, wL3toL4[k])]
    oL4toOut = [neuronSigmoid(oL3toL4, wL4toOut[0]), neuronSigmoid(oL3toL4, wL4toOut[1])]
    return wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, r


def BackPropagationSinglePoint(i, j, img1, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, l_rate, alpha):
    (wInpL1, wL1toL2, wL2toL3, wL3toL4, wL4toOut) = wv
    # print(i,j,sep=" ")
    t1 = (lbl[i][j] - oL4toOut[i][j][0]) * SigmoidDer(oL4toOut[i][j][0])
    t2 = (lblInv[i][j] - oL4toOut[i][j][1]) * SigmoidDer(oL4toOut[i][j][1])
    dOut = [t1, t2]
    # print(dOut)
    # d for the fourth layer
    dL4 = []
    for k in range(len(wL4toOut[0])):
        s = 0
        for h in range(len(wL4toOut)):
            s = s + wL4toOut[h][k] * dOut[h]
        dL4.append(s * SigmoidDer(oL3toL4[i][j][k]))
    # print(dL4)
    # d for the third layer
    dL3 = []
    for k in range(len(wL3toL4[0])):
        s = 0
        for h in range(len(wL3toL4)):
            s = s + wL3toL4[h][k] * dL4[h]
        d = s * SigmoidDer(oL2toL3[i][j][k])
        dL3.append(d)
    # print(dL3)
    # d for second layer
    dL2 = []
    for k in range(len(wL2toL3[0])):
        s = 0
        for h in range(len(wL2toL3)):
            s = s + wL2toL3[h][k] * dL3[h]
        d = s * SigmoidDer(oL1toL2[i][j][k])
        dL2.append(d)
    # print(dL2)
    # d for first layer
    dL1 = []
    for k in range(len(wL1toL2[0])):
        s = 0
        for h in range(len(wL1toL2)):
            s = s + wL1toL2[h][k] * dL2[h]
        d = s * SigmoidDer(oInpL1[i][j][k])
        dL1.append(d)
    # print(dL1)
    retWin = windowCreator(i, j, 3, 3, img1)
    # Updating weights
    for row in range(len(wInpL1)):
        for col in range(len(wInpL1[0])):
            delta = dL1[row] * retWin[col] * l_rate
            wInpL1[row][col] = (wInpL1[row][col] + delta * alpha)

    for row in range(len(wL1toL2)):
        for col in range(len(wL1toL2[0])):
            delta = dL2[row] * oInpL1[i][j][col] * l_rate
            wL1toL2[row][col] = (wL1toL2[row][col] + delta * alpha)

    for row in range(len(wL2toL3)):
        for col in range(len(wL2toL3[0])):
            delta = dL3[row] * oL1toL2[i][j][col] * l_rate
            wL2toL3[row][col] = (wL2toL3[row][col] + delta * alpha)

    for row in range(len(wL3toL4)):
        for col in range(len(wL3toL4[0])):
            delta = dL4[row] * oL2toL3[i][j][col] * l_rate
            wL3toL4[row][col] = (wL3toL4[row][col] + delta * alpha)

    for k in range(len(wL4toOut)):
        for h in range(len(wL4toOut[0])):
            delta = dOut[k] * oL3toL4[i][j][h] * l_rate
            wL4toOut[k][h] = (wL4toOut[k][h] + alpha * delta)
    wv = (wInpL1, wL1toL2, wL2toL3, wL3toL4, wL4toOut)
    # print(weightVector)
    return wv


def BackPropagation(img1, lbl, lblInv, r, noOfEpochs, wv, l_rate, alpha):
    print("Epoch 1 Processing")
    (wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, r) = BackPropagationOutput(img1, r, wv)
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            (wv, oInpL1[i][j], oL1toL2[i][j], oL2toL3[i][j], oL3toL4[i][j], oL4toOut[i][j],
             r) = BackPropagationOutSinglePoint(i, j, img1, r, wv)
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            wv = BackPropagationSinglePoint(i, j, img1, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut,
                                            l_rate, alpha)
    # print(wv)
    for epoch in range(1, noOfEpochs):
        print("Epoch ", epoch + 1, " Processing", sep='')
        for i in range(1, r[0] + 1):
            for j in range(1, r[1] + 1):
                (wv, oInpL1[i][j], oL1toL2[i][j], oL2toL3[i][j], oL3toL4[i][j], oL4toOut[i][j],
                 r) = BackPropagationOutSinglePoint(i, j, img1, r, wv)
                wv = BackPropagationSinglePoint(i, j, img1, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toL3, oL3toL4,
                                                oL4toOut, l_rate, alpha)
        # print(oL4toOut)
    # (wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, r) = BackPropagationOutput(img1, r, wv)
    return wv, oL4toOut

def BackPropogationClassifier(img1, r, wv):
    (wv, oInpL1, oL1toL2, oL2toL3, oL3toL4, oL4toOut, r) = BackPropagationOutput(img1, r, wv)
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            (wv, oInpL1[i][j], oL1toL2[i][j], oL2toL3[i][j], oL3toL4[i][j], oL4toOut[i][j],
             r) = BackPropagationOutSinglePoint(i, j, img1, r, wv)
    o = writeImage2(oL4toOut, r)
    return o
