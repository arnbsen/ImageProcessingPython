from scipy.spatial import distance as ed
import numpy as np
import sys
import math

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
