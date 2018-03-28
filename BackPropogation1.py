im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/c1.png'
in2 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/c2.png'
l = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/gt1.png'
from NN1 import *
from change_knn import *
from supportLibrary import globalNormalisation
print("Training......")
im2 = in2 # + '0882.png'
lb = l # + '000882.png'
(img1, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
misc.imsave('inv.png', lblInv)
wv1 = assignIntWeights()
(wv1, o) = BackPropagation(globalNormalisation(img1.tolist(), 161, 0), lbl, lblInv, r, noOfEpochs=5000, wv= wv1, l_rate=0.06, alpha=1)
o1 = writeImage2(o,r)
print(errorCalc(output=o1, lbl=lbl, r=r))


