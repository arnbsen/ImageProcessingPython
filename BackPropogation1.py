im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/in000001.png'
im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/input/in000882.png'
lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/DB/groundtruth/gt000882.png'
from BackPropogation import *
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)