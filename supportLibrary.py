# Some of the basic functions required for prepossessing for Image Processing
import numpy as np
import statistics
from scipy import misc


def meanFilter(data, padWidth):
    data = np.pad(data, pad_width=(padWidth, padWidth), mode='reflect').tolist()
    output = []
    for i in range(padWidth, len(data) - padWidth):
        temp = []
        for j in range(padWidth, len(data[0]) - padWidth):
            val = sum(data[i - padWidth + 2:i + padWidth - 2][j - padWidth + 2:j + padWidth - 2]) / padWidth * padWidth
            temp = temp + [val]
        output = output + [temp]
    return output


def stdDevFilter(data, windowWidth):
    r = (len(data), len(data[0]))
    data = np.pad(data, pad_width=(windowWidth - 2, windowWidth - 2), mode='reflect').tolist()
    output = []
    for i in range(windowWidth - 2, r[0] + windowWidth - 2):
        temp = []
        for j in range(windowWidth - 2, r[1] + windowWidth - 2):
            tarr = data[i - windowWidth + 2][j - windowWidth + 2:j + windowWidth - 1] + data[i][
                                                                                        j - windowWidth + 2:j + windowWidth - 1] + \
                   data[i + windowWidth - 2][j - windowWidth + 2:j + windowWidth - 1]
            temp = temp + [statistics.stdev(tarr)]
        output = output + [temp]
    return output


def addlpc(data1, data2, mean1, mean2):
    r = len(data1)
    sum1 = 0
    for i in range(r):
        sum1 = sum1 + (data1[i] - mean1) * (data2[i] - mean2)
    return sum1


def pearsonCorrelationFilter(data1, data2, r, windowWidth):
    data1 = np.pad(data1, pad_width=(windowWidth - 2, windowWidth - 2), mode='reflect').astype(float).tolist()
    data2 = np.pad(data2, pad_width=(windowWidth - 2, windowWidth - 2), mode='reflect').astype(float).tolist()
    output = []
    for i in range(windowWidth - 2, r[0] + windowWidth - 2):
        temp = []
        for j in range(windowWidth - 2, r[1] + windowWidth - 2):
            tarr1 = data1[i - windowWidth + 2][j - windowWidth + 2:j + windowWidth - 1] + data1[i][
                                                                                          j - windowWidth + 2:j + windowWidth - 1] + \
                    data1[i + windowWidth - 2][j - windowWidth + 2:j + windowWidth - 1]
            tarr2 = data2[i - windowWidth + 2][j - windowWidth + 2:j + windowWidth - 1] + data2[i][
                                                                                          j - windowWidth + 2:j + windowWidth - 1] + \
                    data2[i + windowWidth - 2][j - windowWidth + 2:j + windowWidth - 1]
            mean1 = statistics.mean(tarr1)
            mean2 = statistics.mean(tarr2)
            upp = addlpc(tarr1, tarr2, mean1, mean2)
            # print(statistics.stdev(tarr1), statistics.stdev(tarr1), sep='  ')
            temp = temp + [upp / statistics.stdev(tarr1) * statistics.stdev(tarr1)]

        output = output + [temp]
    return output


""""
Sample code to check 
img1 = abs(misc.imread('cell1.png') - misc.imread('cell2.png')).tolist()
img2 = misc.imread('cell2.png').tolist()
r = (len(img1), len(img1[0]))
test = stdDevFilter(img1, 3)
out = []
for i in range(len(test)):
    t = []
    for j in range(len(test[0])):
        if test[i][j] < 20:
            t = t + [0]
        else:
            t = t + [255]
    out = out + [t]
misc.imsave('disaster.png', out)

"""