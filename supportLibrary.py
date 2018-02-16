# Some of the basic functions required for prepossessing for Image Processing
import numpy as np
import statistics

from scipy import misc


def meanFilter(data, windowWidth):
    r = (len(data), len(data[0]))
    data = np.pad(data, pad_width=(windowWidth - 2, windowWidth - 2), mode='reflect').tolist()
    output = []
    for i in range(windowWidth - 2, r[0] + windowWidth - 2):
        temp = []
        for j in range(windowWidth - 2, r[1] + windowWidth - 2):
            tarr = data[i - windowWidth + 2][j - windowWidth + 2:j + windowWidth - 1] + data[i][
                                                                                        j - windowWidth + 2:j + windowWidth - 1] + \
                   data[i + windowWidth - 2][j - windowWidth + 2:j + windowWidth - 1]
            temp = temp + [statistics.mean(tarr)]
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

def normalisation(data):
    min1 = 999999
    max1 = -999999
    for i in range(len(data)):
        min1 = min(min1, min(data[i]))
        max1 = max(max1, max(data[i]))
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = round((data[i][j] - min1) / (max1 - min1), 2)
    return data

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


def windowCreator(i, j, h, w, arr):
    window = []
    for k in range(h):
        for l in range(w):
            try:
                window.append(arr[i + k - 1][j + l - 1])
            except IndexError:
                print("i = ", i, " j = ", j)
    return window


def prepareData(im1, im2, lb):
    # dif = abs(misc.imread(im1).astype(float)/255.0 - misc.imread(im2).astype(float)/255.0)
    img1 = (misc.imread(im1).astype(float)).tolist()
    img2 = (misc.imread(im2).astype(float)).tolist()
    # dif = np.pad(dif, pad_width=10, mode='reflect').tolist()
    lbl = misc.imread(lb).tolist()
    binImage = np.array(lbl).astype(int).ravel().tolist()
    # binImageInv = np.array(binImageInv).astype(float).ravel().tolist()
    data = []
    labels = []
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            data = data + [[abs(img1[i][j] - img2[i][j])]]
            labels = labels + [[binImage[i + j] // 255]]
    r = (len(img1), len(img1[0]))
    data = np.array(data).tolist()
    labels = np.array(labels).tolist()
    return data, labels, r


def prepareClusterData(im1, lb):
    print("Preparing Data. Please Wait...")
    dif = np.pad(misc.imread(im1).astype(float) / 255.0, mode='reflect', pad_width=(2, 2)).tolist()
    print(len(dif), len(dif[0]))
    binImage = (misc.imread(lb).astype(float) / 255.0).ravel()
    binImageInv = abs(np.full(binImage.shape, 1.0) - binImage).tolist()
    data = []
    labels = []
    for i in range(2, len(dif) - 2):
        for j in range(2, len(dif[0]) - 2):
            data = data + [windowCreator(i, j, 3, 3, dif)]
            labels = labels + [[binImage[i + j], binImageInv[i + j]]]
    r = (len(dif), len(dif[0]))
    return data, labels, r


def prepareDataAlternateNorm(im1, im2, lb):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).ravel().tolist()
    min1 = min(img1)
    max1 = max(img1)
    img2 = misc.imread(im2).ravel().tolist()
    min2 = min(img2)
    max2 = max(img2)
    binImage = abs(misc.imread(lb).astype(float) // 255).astype(int).ravel()
    binImageInv = abs(np.full(binImage.shape, 1.0) - binImage).tolist()
    img1 = np.array(img1)
    img2 = np.array(img2)
    binImage = binImage.tolist()
    # Normalisation of data
    img1 = ((img1 - np.full(img1.shape, min1)) / (max1 - min1))
    img2 = ((img2 - np.full(img2.shape, min2)) / (max2 - min2))
    img1 = img1.tolist()
    img2 = img2.tolist()
    data = []
    labels = []
    for i in range(len(img1)):
        data = data + [[abs(img2[i] - img1[i])]]
        labels = labels + [[binImage[i]]]
    data = data
    labels =labels
    return data, labels, r


def prepareDataInter(im1, im2, lb):
    img1 = misc.imread(im1).astype(np.float32)
    r = img1.shape
    img2 = misc.imread(im2).astype(np.float32)
    img1 = np.pad(img1, pad_width=(1, 1), mode='reflect').tolist()
    img2 = np.pad(img2, pad_width=(1, 1), mode='reflect').tolist()
    binImage = misc.imread(lb).astype(np.int64) // 255
    binImageInv = abs(np.full(binImage.shape, 1) - binImage).tolist()
    binImage = binImage.tolist()
    data = []
    labels = []
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            data = data + [[img1[i - 1][j - 1], img2[i - 1][j - 1], img1[i - 1][j], img2[i - 1][j], img1[i - 1][j + 1],
                     img2[i - 1][j + 1], img1[i][j - 1], img2[i][j - 1], img1[i][j], img2[i][j],
                     img1[i][j], img2[i][j + 1], img1[i + 1][j - 1], img2[i + 1][j - 1], img1[i + 1][j], img2[i + 1][j],
                     img1[i + 1][j + 1], img2[i + 1][j + 1]]]

    for i in range(r[0]):
        for j in range(r[1]):
            labels = labels + [[binImage[i][j], abs(1 - binImage[i][j])]]
    return np.array(data, dtype=np.float32).tolist(), np.array(labels, dtype=np.float32).tolist(), r
def writeImage(model, data, r):
    output = model.predict(data)
    output = output.tolist()
    temp = []
    for i in range(1, r[0]+1):
        itemp = []
        for j in range(1, r[1]+1):
            if output[i+ j][0] < output[i+j][1]:
                itemp = itemp + [0]
            else:
                itemp = itemp + [255]
        temp = temp + [itemp]

    temp = np.array(temp).astype(np.uint8)
    misc.imsave('disaster.png', temp)

def writeImage2(output, r):
    temp = []
    for i in range(1, r[0]+1):
        itemp = []
        for j in range(1, r[1]+1):
            if output[i][j][0] < output[i][j][1]:
                itemp = itemp + [0]
            else:
                itemp = itemp + [255]
        temp = temp + [itemp]

    temp = np.array(temp).astype(np.uint8)
    misc.imsave('disaster.png', temp)
    return (temp // 255).astype(int).tolist()

def prepareDataStdDev(im1, im2, lb):
    dif = normalisation(abs(misc.imread(im1).astype(float) - misc.imread(im2).astype(float)).tolist())
    lbl = (misc.imread(lb).astype(int) // 255).ravel().reshape(-1, 1).tolist()
    difstd = np.pad(normalisation(stdDevFilter(dif, 3)), pad_width=(1, 1), mode='reflect')
    dif = np.pad(dif, pad_width=(1, 1), mode='reflect').tolist()
    t = []
    for i in range(1, len(dif)-1):
        for j in range(1, len(dif[0])-1):
            t = t + [windowCreator(i, j, 3, 3, difstd)] #+ windowCreator(i, j, 3, 3, difstd)]
    r = (len(dif)-2, len(dif[0])-2)
    return t, lbl, r

def normalisation2(im1, im2, lb):
    it1 = misc.imread(im1).ravel().tolist()
    it2 = misc.imread(im2).ravel().tolist()
    var1 = statistics.variance(it1)
    var2 = statistics.variance(it2)
    mean1 = statistics.mean(it1)
    mean2 = statistics.mean(it2)
    dif = abs(misc.imread(im1) - (var1/var2)*(misc.imread(im2) - mean2) + mean1)
    lbl = (misc.imread(lb).astype(int) // 255).ravel().reshape(-1, 1).tolist()

def meanDiff(im1, im2, lbl):
    img1 = misc.imread(im1).astype(float)
    img2 = misc.imread(im2).astype(float)
    img1 = meanFilter(img1, 3)
    img2 = meanFilter(img2, 3)


def prepareDataClusStd(clImg, lb):
    img1 = misc.imread(clImg).astype(float)
    r = img1.shape
    img1 = np.pad(img1, pad_width=(1, 1), mode='reflect').tolist()
    binImage = misc.imread(lb).astype(np.int64) // 255
    data = []
    labels = []
    for i in range(1, r[0] + 1):
        for j in range(1, r[1] + 1):
            data = data + [[img1[i - 1][j - 1], img1[i - 1][j],  img1[i - 1][j + 1], img1[i][j - 1], img1[i][j], img1[i][j],  img1[i + 1][j - 1], img1[i + 1][j], img1[i + 1][j + 1]]]

    for i in range(r[0]):
        for j in range(r[1]):
            labels = labels + [[binImage[i][j], abs(1 - binImage[i][j])]]

    return data, labels, r


def errorCalc(output, lbl, r):
    cnt = 0
    total = r[0] * r[1]
    output = np.pad(output, pad_width=(1, 1), mode='reflect').tolist()
    for i in range(1, r[0]):
        for j in range(1, r[1]):
            if lbl[i][j] != output[i][j]:
                cnt = cnt + 1
    return (cnt / total) * 100


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