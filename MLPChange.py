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
    img1 = img1.tolist()
    img2 = img2.tolist()
    return (img1, img2, lbl, r)


def assignIntWeights():
    wInpL1 = assignRandomWeight(5, 2)
    wL1ToL2 = assignRandomWeight(3, 5)
    wL2toOut = assignRandomWeight(2, 3)
    return (wInpL1, wL1ToL2, wL2toOut)


def BackPropagationOutput(img1, img2, r, wv):
    (wInpL1, wL1ToL2, wL2toOut) = wv
    # Calculating First Layer to second layer
    oInpL1 = []
    for i in range(0, r[0]):
        oTemp = []
        for j in range(0, r[1]):
            temp = [neuronSigmoid([img1[i][j], img2[i][j]], wInpL1[0])] + [
                neuronSigmoid([img1[i][j], img2[i][j]], wInpL1[1])] + [
                       neuronSigmoid([img1[i][j], img2[i][j]], wInpL1[2])] + [
                       neuronSigmoid([img1[i][j], img2[i][j]], wInpL1[3])] + [
                       neuronSigmoid([img1[i][j], img2[i][j]], wInpL1[4])]
            oTemp.append(temp)
        oInpL1.append(oTemp)
        # Computing Layer 1 to Layer 2
    oL1toL2 = []
    for i in range(len(oInpL1)):
        oTemp = []
        for j in range(len(oInpL1[0])):
            temp = [neuronSigmoid(oInpL1[i][j], wL1ToL2[0])] + [neuronSigmoid(oInpL1[i][j], wL1ToL2[1])] + [
                neuronSigmoid(oInpL1[i][j], wL1ToL2[1])]
            oTemp.append(temp)
        oL1toL2.append(oTemp)
    # Computing Layer 2 to Output
    oL2toOut = []
    for i in range(len(oL1toL2)):
        oTemp = []
        for j in range(len(oL1toL2[0])):
            oTemp.append([neuronReLu(oL1toL2[i][j], wL2toOut[0])] + [neuronReLu(oL1toL2[i][j], wL2toOut[1])])
        oL2toOut.append(oTemp)
    weightVector = (wInpL1, wL1ToL2, wL2toOut)
    return (weightVector, oInpL1, oL1toL2, oL2toOut, r)

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
def BackPropagation(img1, img2, lbl, lblInv, r, noOfEpochs, wv, l_rate, alpha):
    print("Epoch 1 Processing")
    (wv, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(img1, img2, r, wv)
    for i in range(r[0]):
        for j in range(r[1]):
            wv = BackPropagationSinglePoint(i, j, img1, img2, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toOut, l_rate, alpha)
    print(wv)
    for epoch in range(1, noOfEpochs):
        print("Epoch ", epoch + 1, " Processing", sep='')
        (wv, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(img1, img2, r, wv)
        for i in range(r[0]):
            for j in range(r[1]):
                wv = BackPropagationSinglePoint(i, j, img1, img2, lbl, lblInv, wv, oInpL1, oL1toL2, oL2toOut, l_rate, alpha)
        # print(weightVector)
    (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(img1, img2, r, wv)
    return (weightVector, oL2toOut)

def Classifier(img1, img2, wv, r):
    (weightVector, oInpL1, oL1toL2, oL2toOut, r) = BackPropagationOutput(img1, img2, r, wv)
    out = []
    for i in range(r[0]):
        outT = []
        for j in range(r[1]):
            if oL2toOut[i][j][0] > oL2toOut[i][j][1]:
                outT.append(0)
            else:
                outT.append(1)
        out.append(outT)
    misc.imsave('temp.png', out)
    return out

