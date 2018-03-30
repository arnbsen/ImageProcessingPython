from scipy.spatial import distance as ed
from scipy import misc
import numpy as np
import sys
import math
import random

def determineBMU(SOMmap, inputVector):
    minDist = sys.maxsize
    winCood = (0,0)
    for i in range(len(SOMmap)):
        for j in range(len(SOMmap[0])):
            dist = ed.euclidean(SOMmap, inputVector)
            if dist < minDist:
                winCood = (i,j)
                minDist = dist

    return winCood


def calculateNeighbourCoordinates(winCood, radius):
    topRight = (winCood[0] + round(math.cos(3*math.pi/4)*radius), winCood[1] + round(math.cos(3*math.pi/4)*radius))
    bottomLeft = (winCood[0] + round(math.cos(-1*math.pi/4)*radius), winCood[1] + round(math.cos(-1*math.pi/4)*radius))
    neighbours = []
    for i in range(topRight[1],bottomLeft[1]+1):
        for j in range(topRight[0],bottomLeft[0]+1):
            if i>=0 and j>=0:
                neighbours = neighbours + [[i, j]]

    return neighbours


def radiusOfNeighbourhood(initRadius, mapRadius, itr):
    l = mapRadius/itr
    return initRadius * math.exp(-1*itr/l)


def learningRate(initRate, mapRadius, itr):
    l = mapRadius / itr
    return initRate * math.exp(-1 * itr / l)

def initWeightVector(windowSize):
    r = []
    for i in range(windowSize):
        r = r + [round(random.uniform(0,1)*255)]
    return r


def initSOM(h, w, windowSize):
    SOMmap = []
    for i in range(h):
        temp =[]
        for j in range(w):
            temp = temp + initWeightVector(windowSize)
        SOMmap.append(temp)
    return SOMmap

def prepareSOMdata(im1, im2):
    img1 = np.pad(misc.imread(im1), pad_width = (1, 1), mode = 'reflect').astype(float).tolist()
    img2 = np.pad(misc.imread(im2), pad_width = (1, 1), mode = 'reflect').astype(float).tolist()
    dif = abs(img1 - img2)
    data = []
    for i in range(1, len(data)-1):
        for j in range(1, len(data[0]) - 1):
            temp = dif[i-1][j-1:j+2] + dif[i][j-1:j+2] + dif[i+1][j-1:j+2]
            data = data + [temp]

    return data

def SOMinit():

