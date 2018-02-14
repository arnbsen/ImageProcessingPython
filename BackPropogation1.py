im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell1small.png'
in2 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell2small.png'
l = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/gtsmall.png'
from NN1 import *

print("Training......")
im2 = in2 # + '0882.png'
lb = l # + '000882.png'
(img1, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
misc.imsave('inv.png', lblInv)
(wv, o) = BackPropagation(normalisation(img1.tolist()), lbl, lblInv, r, noOfEpochs=3000, wv= assignIntWeights(), l_rate= 0.6, alpha=1)
o1 = writeImage2(o,r)
