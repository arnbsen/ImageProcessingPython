#BackPropogation using Gradient Descent
#Starting with 3 hidden layers of size (in) -> 5 -> 3 -> 1 -> (out)
from change_knn import *
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
            window.append(arr[i + h - 2][j + w - 2])
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
    img1 = misc.imread(im1).astype(int).tolist()
    img2 = misc.imread(im2).astype(int).tolist()
    lbl = misc.imread(lb).astype(int).tolist()
