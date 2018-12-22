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
            dist = ed.euclidean(SOMmap[i][j], inputVector)
            if dist < minDist:
                winCood = (i,j)
                minDist = dist

    return winCood


def calculateNeighbourCoordinates(winCood, radius, h, w):
    topRight = (winCood[0] + round(math.cos(3*math.pi/4)*radius), winCood[1] + round(math.cos(3*math.pi/4)*radius))
    bottomLeft = (winCood[0] + round(math.cos(-1*math.pi/4)*radius), winCood[1] + round(math.cos(-1*math.pi/4)*radius))
    neighbours = []
    for i in range(topRight[1],bottomLeft[1]+1):
        for j in range(topRight[0],bottomLeft[0]+1):
            if i >= 0 and j >= 0 and i < h and j < w:
                neighbours = neighbours + [[i, j]]

    return neighbours


def exponentialDecay(initRadius, tau1, itr):
    return initRadius * math.exp(-1*itr/ tau1)


def learningRate(initRate, tau2, itr):
    return initRate * math.exp(-1 * itr / tau2)

def initWeightVector(windowSize):
    r = []
    for i in range(windowSize):
        r = r + [random.uniform(0, 1)]
    return r


def initSOM(h, w, windowSize):
    SOMmap = []
    for i in range(h):
        temp =[]
        for j in range(w):
            temp = temp + [initWeightVector(windowSize)]
        SOMmap.append(temp)
    return np.array(SOMmap)

def prepareSOMdata(im1, im2):
    img1 = np.pad(misc.imread(im1), pad_width = (1, 1), mode = 'reflect').astype(float) / 255.0
    img2 = np.pad(misc.imread(im2), pad_width = (1, 1), mode = 'reflect').astype(float) / 255.0
    dif = abs(img1 - img2).tolist()
    data = []
    for i in range(1, len(dif)-1):
        for j in range(1, len(dif[0]) - 1):
            temp = dif[i-1][j-1:j+2] + dif[i][j-1:j+2] + dif[i+1][j-1:j+2]
            data = data + [temp]

    return np.array(data)

def SOMTraining(im1, im2, h=100, w=100, windowSize=9, initRaduis=75, initLR=0.1, no_of_itr=10000):
    data = prepareSOMdata(im1, im2)
    SOMmap = initSOM(h, w, windowSize)

    for itr in range(1, no_of_itr+1):

        learn_rate = initLR
        current_radius = initRaduis
        rand_index = int(random.uniform(0,len(data)-1))
        BMU = determineBMU(SOMmap, data[rand_index])
        BMU_arr = [[BMU[0], BMU[1]]]
        print("Iteration  number: ", itr, " Raduis: ", current_radius, sep="", end="\n")
        for i in BMU_arr:
            #print(i)
            diff = data[rand_index] - SOMmap[i[0]][i[1]]
            #distsq = (BMU[0] - i[0])**2 + (BMU[1] - i[1])**2
            #theta = math.exp(-1*distsq/(2*current_radius))
            SOMmap[i[0]][i[1]] = SOMmap[i[0]][i[1]] + learn_rate*diff
            #print(SOMmap[i[0]][i[1]])

    return SOMmap


