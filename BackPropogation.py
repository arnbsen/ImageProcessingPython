#BackPropogation using Gradient Descent
#Starting with 3 hidden layers of size (in) -> 5 -> 3 -> 1 -> (out)
#This library is for for change detection only


import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
import math
from PIL import Image
import os
import random
#Creating a simple neuron based on Threshold
def neuron(x,w,th):
    total = 0
    for i in range(len(x)):
        total = total + x[i]*w[i]
    if total > th:
        return 1
    else:
        return 0
#Creating a neuron based on the sigmoid function
def neuronSigmoid(x,w):
    total = 0
    for i in range(len(x)):
        total = total + x[i] * w[i]
    return 1/ (1 + math.exp(-total))
def convertImageToBinary(img1):
    binImage = []
    for i in range(0,len(img1)):
        tmp = []
        for j in range(0,len(img1[0])):
            if img1[i][j] == 0:
                tmp.append(0)
            else:
                tmp.append(1)
        binImage.append(tmp)
    return binImage

def windowCreator(i,j,h,w,arr):
    window = []
    for k in range(h):
        for l in range(w):
            window.append(arr[i + h - 1][j + w - 1])
    return window

def assignRandomWeight(l,b):
    wv = []
    for i in range(l):
        wvi = []
        for j in range(b):
            wvi.append(random.uniform(0, 1))
        wv.append(wvi)
    return wv
def twoDimImread(im1,im2,lb):
    img1 = misc.imread(im1).astype(int)
    img2 = misc.imread(im2).astype(int)
    lbl = misc.imread(lb).astype(int).tolist()
    dif = abs(img1 - img2)
    r = dif.shape
    dif = np.pad(dif, mode = 'reflect', pad_width = 2)
    dif = dif.tolist()
    return (dif,lbl,r)

def assignIntWeights():
    wInpL1 = assignRandomWeight(3, 9)
    wL1ToL2 = assignRandomWeight(2, 3)
    wL2toOut = assignRandomWeight(1, 2)
    return (wInpL1,wL1ToL2,wL2toOut)

def BackPropogationOutput(dif,r,weightVector, th1, th2):
    (wInpL1, wL1ToL2, wL2toOut) = weightVector
    oInpL1 = []
    #Computing Output of the neuron directly connected to Inputs
    for i in range(1,r[0]+1):
        oTemp = []
        for j in range(1,r[1]+1):
            retWin = windowCreator(i,j,3,3,dif)
            temp = [neuron(retWin, wInpL1[0], th1)] + [neuron(retWin, wInpL1[1], th1)] + [neuron(retWin, wInpL1[2], th1)]
            oTemp.append(temp)
        oInpL1.append(oTemp)
    #Computing Layer 1 to Layer 2
    oL1toL2 = []
    for i in range(len(oInpL1)):
        oTemp = []
        for j in range(len(oInpL1[0])):
            temp = [neuronSigmoid(oInpL1[i][j],wL1ToL2[0])] + [neuronSigmoid(oInpL1[i][j],wL1ToL2[1])]
            oTemp.append(temp)
        oL1toL2.append(oTemp)
    #Computimg Layer 2 to Output
    oL2toOut = []
    for i in range(len(oL1toL2)):
        oTemp = []
        for j in range(len(oL1toL2[0])):
            oTemp.append(neuronSigmoid(oL1toL2[i][j], wL2toOut[0]))
        oL2toOut.append(oTemp)
    weightVector = (wInpL1, wL1ToL2, wL2toOut)
    return (weightVector,oInpL1,oL1toL2,oL2toOut,r)

def BackPropogationWeightCal(dif,lbl,weightVector,oInpL1,oL1toL2,oL2toOut,r):
    (wInpL1, wL1ToL2, wL2toOut) = weightVector
    #Computing error term for the each value
    dOut = []
    for i in range(len(lbl)):
        dTemp = []
        for j in range(len(lbl[0])):
            d = oL2toOut[i][j]*(1 - oL2toOut[i][j])*(lbl[i][j] - oL2toOut[i][j])
            dTemp.append(d)
        dOut.append(dTemp)
    #BackPropogating the Output Layer to Update the weights
    for i in range(len(dOut)):
        for j in range(len(dOut[0])):
            for k in range(len(wL2toOut)):
                delta = dOut[i][j]*oL2toOut[i][j][k]
                wL2toOut[k] = wL2toOut[k] + delta
    #Computing the error at L2
    for i in range(len(oL1toL2)):
        for j in range(len(oL1toL2[0])):













