#Supports only .png formats
#The file contains methods that link the sklearn-knn(Training and Classification) and scipy(Image Read and Write) using numpy (Array)

import numpy as np
import pickle
from sklearn import neighbors
from scipy import misc
from PIL import Image
import os
def read_image_gray(im1,im2,lb):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).astype(int).ravel()
    img2 = misc.imread(im2).astype(int).ravel()
    dif = np.array(abs(img1 - img2)).reshape(-1,1)
    lbl = misc.imread(lb).ravel()
    return (dif,lbl,r)
def knn_trainer(dif,lbl,n):
    trainer = neighbors.KNeighborsClassifier(n_neighbors = n)
    trainer.fit(dif,lbl)
    return trainer
def model_save(filename,model):
    pickle.dump(model, open(filename, 'wb'))
def model_load(filename):
    model = pickle.load(open(filename, 'rb'))
    return model
def classifier_gray(im1,im2,model):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).astype(int).ravel()
    img2 = misc.imread(im2).astype(int).ravel()
    dif = np.array(abs(img1 - img2)).tolist()
    out = []
    for i in range(len(dif)):
        temp = model.predict(dif[i])
        temp = temp.tolist()
        #print(temp[0])
        out.append(temp[0])
    return (out,r)
def write_file(filename,out,r):
    c = r[1]
    r = r[0]
    final = []
    k = 0
    for i in range(r):
        col = []
        for j in range(c):
            col.append(out[k])
            k = k + 1
        final.append(col)
    final = np.array(final)
    misc.imsave(filename,final)

def read_image_col(im1,im2,lb):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).astype(int)
    img2 = misc.imread(im2).astype(int)
    dif = np.array(abs(img1 - img2)).tolist()
    lbl = misc.imread(lb).tolist()
    lbtemp = []
    dtemp = []
    k = 0
    for i in range(r[0]):
        for j in range(r[1]):
            dtemp.insert(k,dif[i][j])
            lbtemp.insert(k,lbl[i][j])
            k = k + 1
    dif = np.array(dtemp)
    lbl = np.array(lbtemp)
    return (dif,lbl,r)
def classifier_col(im1,im2,model):
    r = misc.imread(im1).shape
    img1 = misc.imread(im1).astype(int)
    img2 = misc.imread(im2).astype(int)
    dif = np.array(abs(img1 - img2)).tolist()
    lbtemp = []
    dtemp = []
    k = 0
    for i in range(r[0]):
        for j in range(r[1]):
            dtemp.insert(k, dif[i][j])
            k = k + 1
    dif = np.array(dtemp)
    out = model.predict(dif)
    return (out,r)
def read_image(im1,im2,lb):
     r = misc.imread(im1).shape
     if r == 3:
         (dif, lbl, r) = read_image_gray(im1, im2, lb)
     else:
         (dif, lbl, r) = read_image_col(im1, im2, lb)
     return (dif, lbl, r)

def classifier(im1,im2,model):
    r = misc.imread(im1).shape
    if r == 3:
        (out, r) = classifier_gray(im1, im2, model)
    else:
        (out, r) = classifier_col(im1, im2, model)
    return  (out, r)
def disp(out,r):
    write_file('temp1.png', out, r)
    Image.open('temp1.png').show()
    os.remove('temp1.png')
