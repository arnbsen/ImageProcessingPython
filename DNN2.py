# Contributed by Kasturi Saha
import numpy as np
from cv2 import imread, imwrite, imshow
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy import misc

im1 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\cell1.png'
im2 = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\cell2.png'
lb = 'C:\\Users\\arnbs\\PycharmProjects\\ImageProcessingPython\\IMAGE_DB\\changeSD.png'
I = imread(im1,0).astype(int) #read image1
(r,c)=I.shape
J = imread(im2,0).astype(int)
D = np.round(abs(I-J).reshape(-1,1)/255.0, 2)
G = imread(lb,0).astype(int).ravel().tolist()
(lbl, lblnv) = convertImageToBinary(G)
model = MLPClassifier(alpha = 1e-5, batch_size = 10, hidden_layer_sizes = (100, 5, 2), verbose = True, solver='sgd')
model.fit(D,G)
l = len(G)
Dr = np.array([model.predict([D[x]]) for x in range(l)]).reshape(r,c).astype(np.uint8)
print(Dr)
