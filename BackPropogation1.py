im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/0001.png'
in2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/'
l = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/groundtruth/gt'
from BackPropogation import *

print("Training with two examples")
im2 = in2 + '0882.png'
lb = l + '000882.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)
(wv, o1) = BackPropagation(dif,lbl,r,100,assignIntWeights(),0.00002)
im2 = in2 + '1598.png'
lb = l + '001598.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)
(wv, o2) = BackPropagation(dif,lbl,r,100,wv,0.00002)
im2 = in2 + '1307.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o3,r)= BackPropagationClassifier(dif, r, wv)
im2 = in2 + '1517.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o4,r)= BackPropagationClassifier(dif, r, wv)
im2 = in2 + '1547.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o5,r)= BackPropagationClassifier(dif, r, wv)
im2 = in2 + '1648.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o6,r)= BackPropagationClassifier(dif, r, wv)
im2 = in2 + '1598.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o7,r)= BackPropagationClassifier(dif, r, wv)
im2 = in2 + '1700.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
(o8,r)= BackPropagationClassifier(dif, r, wv)