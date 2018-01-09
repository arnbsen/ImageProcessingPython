from MLPChange import *
im1 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\input\\in000001.png'
in2 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\input\\in'
l = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\DB\\groundtruth\\gt'
print("Training with two example")
im2 = in2 + '000882.png'
lb = l + '000882.png'
(img1, img2, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
(wv, o) = BackPropagation(img1, img2, lbl, lblInv, r, 50, assignIntWeights(), 0.01, 0.01)


