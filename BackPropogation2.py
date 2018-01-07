from BackPropogation import *
im1 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\input\\in000001.png'
in2 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\input\\in'
l = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\groundtruth\\gt'
print("Training with two example")
im2 = in2 + '000882.png'
lb = l + '000882.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)
(wv, o) = BackPropagation(dif,lbl,r,50,assignIntWeights(),0.0002)
im2 = in2 + '001598.png'
lb = l + '001598.png'
(dif, lbl, r) = twoDimImread(im1, im2, lb)
lbl = convertImageToBinary(lbl)
(wv, o) = BackPropagation(dif,lbl,r,50,wv,0.0002)