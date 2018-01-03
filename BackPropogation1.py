im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/IMAGE_DB/1cell.png'
im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/IMAGE_DB/2cell.png'
lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/IMAGE_DB/ref1.png'
from BackPropogation import *
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)