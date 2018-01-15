im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/in000001.png'
in2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/in00'
l = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/groundtruth/gt'
from NN1 import *

print("Training with two examples")
im2 = in2 + '0882.png'
lb = l + '000882.png'
(img1, img2, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
misc.imsave('inv.png', lblInv)
(wv, o) = BackPropagation(img1, img2, lbl, lblInv, r, 50, assignIntWeights(), 0.001, 0.001)

