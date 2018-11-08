from skimage import io
from math import sqrt, atan, cos, sin
from supportLibrary import globalMean, globalStdDev
from statistics import mean, stdev, variance
import numpy as np


def readImage(impath):
    I = io.imread(impath)
    return I

def normalization(arr, m0, var0):
    tarr = arr.ravel().tolist()
    m = mean(tarr)
    var = variance(tarr)
    g = []
    for i in range(len(arr)):
        temp = []
        for j in range(len(arr[0])):
            if arr[i][j] > m:
                temp = temp + [m0 + sqrt((var0 * (arr[i][j] - m)**2) / var)]
            else:
                temp = temp - [m0 + sqrt((var0 * (arr[i][j] - m) ** 2) / var)]
        g = g + [temp]
    return np.array(g)

def orientation(g):
    opre = np.pad(g, (9, 9), mode='reflect').tolist()
    gx = []
    gy = []
    # Using Sobel operator to measure gradient
    for i in range(1, len(opre)-1):
        tx = []
        ty = []
        for j in range(1, len(opre[0])-1):
            tcx = opre[i-1][j-1] + 2 * opre[i-1][j] + opre[i-1][j+1] - opre[i+1][j-1] - 2 * opre[i+1][j] - opre[i+1][j+1]
            tcy = opre[i-1][j-1] + 2 * opre[i][j-1] + opre[i+1][j-1] - opre[i-1][j+1] - 2 * opre[i][j+1] - opre[i+1][j+1]
            tx = tx + [tcx]
            ty = ty + [tcy]
        gx = gx + [tx]
        gy = gy + [ty]
    theta = []
    for i in range(7, len(gx)-7):
        tthetha = []
        for j in range(8, len(gx[0])-7):
            tcvx = 0
            tcvy = 0
            for k in range(i-7, i+8):
                for l in range(j-7, j+8):
                    tcvx = tcvx + 2 * gx[k][l] * gy[k][l]
                    tcvy = tcvy + gx[k][l]**2 - gy[k][l]**2
            tthetha = tthetha + [0.5 * atan(tcvy/tcvx)]
        theta = theta + [tthetha]
    phix = []
    phiy = []
    for i in range(len(theta)):
        tpx = []
        tpy = []
        for j in range(len(theta[0])):
            tpx = tpx + [cos(2 * theta[i][j])]
            tpy = tpy + [sin(2 * theta[i][j])]
        phix = phix + [tpx]
        phiy = phiy + [tpy]
    phidashx = []
    phidashy = []
    guass_mask = np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4],[1, 4, 7, 4, 1]])
    phix = np.pad(np.array(phix), pad_width=(2 ,2), mode='reflect')
    phiy= np.pad(np.array(phiy), pad_width=(2, 2), mode='reflect')
    for i in range(2, len(phix)-2):
        tdx = []
        tdy = []
        for j in range(2, len(phiy)-2):
            tx = sum(sum(phix[i-2:i+3, j-2:j+3] * guass_mask)) / 273
            ty = sum(sum(phiy[i-2:i+3, j-2:j+3] * guass_mask)) / 273
            tdx = tdx + [tx]
            tdy = tdy + [ty]
        phidashx = phidashx + [tdx]
        phidashy = phidashy + [tdy]
    O = []
    for i in range(len(phidashx)):
        to = []
        for j in range(len(phidashx[0])):
            to = to + [0.5 * atan(phidashy[i][j]/phidashx[i][j])]
        O = O + [to]
    return O
