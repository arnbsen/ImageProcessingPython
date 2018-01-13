from NN1 import *
im1 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\img1.png'
im2 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\img2.png'
lb = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\lbl.png'
print("Training")

(img1, img2, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
(wv, o) = BackPropagation(img1, img2, lbl, lblInv,  r, 3, assignIntWeights(), 0.001, 0.9)


