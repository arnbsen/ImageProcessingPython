from change_knn import *
imext = '.jpg'
gtext = '.pgm'
model = model_load('knnColImgK7.knndata')
model1 = model_load('knnColImgK20.knndata')
for i in range(1000,2051,100):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in00' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/groundtruth/gt00' + str(i) + gtext
    (dif, lbl, r) = read_image(im1, im2, lb)
    a = model.fit(dif, lbl)
    a = model1.fit(dif, lbl)
    print(i)
model_save('knnColImgK7.knndata',model)
model_save('knnColImgK20.knndata',model1)
