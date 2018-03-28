from NN1 import *
from change_knn import *
sel = np.random.permutation(960)
sel = sel[:int(0.15*960)]
wv1=assignIntWeights()
for j in range(5):
    print("Master epoch processing.....: ", str(j))
    for i in sel:
        print("Training index: ", str(i))
        im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/SWINS/CELL1/cell1_' + str(i) + '.png'
        im2 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/SWINS/CELL2/cell2_' + str(i) + '.png'
        lb = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/SWINS/GT/gt_' + str(i) + '.png'
        (img1, lbl, r) = twoDimImread(im1, im2, lb)
        (lbl, lblInv) = convertImageToBinary(lbl)
        (wv1, o) = BackPropagation(normalisation(img1.tolist()), lbl, lblInv, r, noOfEpochs=20000, wv=wv1, l_rate=0.6,
                               alpha=1)

print(wv1)