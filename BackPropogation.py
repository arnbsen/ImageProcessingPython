#BackPropogation using Gradient Descent
#Starting with 3 hidden layers of size (in) -> 5 -> 3 -> 1 -> (out)
#This library is for for change detection only


import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
from PIL import Image
import os
import random
#Creating a simple neuron
def neuron(x,w,th):
    total = 0
    for i in range(len(x)):
        total = total + x[i]*w[i]
    if total > th:
        return 1
    else:
        return 0

def convertImageToBinary(im):
    sz = misc.imread(im).shape
    img1 = misc.imread(im).astype(int).tolist()
    binImage = []
    for i in range(0,sz[[0]]):
        tmp = []
        for j in range(0,sz[1]):
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



def BackPropogationRound1(dif,lbl,r):
    wInpL1 = assignRandomWeight(3, 3)
    wL1ToL2 = assignRandomWeight(3, 2)
    wL2toOut = assignRandomWeight(2, 1)
    dInpL1 = []
    #Computing Output of the neuron directly connected to Inputs
    for i in range(1,r[0]+1):
        dTemp = []
        for j in range(1,r[1]+1):
            retWin = windowCreator(i,j,3,3,dif)
            temp = [neuron(retWin[0:3],wInpL1[0],0.5)] + [neuron(retWin[3:6],wInpL1[1],0.5)] + [neuron(retWin[6:],wInpL1[2],0.5)]
            dTemp.append(temp)
        dInpL1.append(dTemp)
    return dInpL1








