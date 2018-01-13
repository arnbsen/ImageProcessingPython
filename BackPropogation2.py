from MLPChange import *
im1 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\1cell.png'
im2 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\2cell.png'
lb = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\ref1.png'
print("Training with two example")

(img1, img2, lbl, r) = twoDimImread(im1, im2, lb)
(lbl, lblInv) = convertImageToBinary(lbl)
(wv, o) = BackPropagation(img1, img2, lbl, lblInv, r, 50, assignIntWeights(), 0.01, 0.01)


