from change_knn import *
model = model_load("kNNOfficek7.knndata")
imext = '.jpg'
gtext = '.pgm'
for i in range(1,10):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in00000' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/output/cl00000' + str(i) + gtext
    (out,r) = classifier(im1,im2,model)
    write_file(lb,out,r)
    print("Classifying Instance",i, sep=" ")
for i in range(10,100):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in0000' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/output/cl0000' + str(i) + gtext
    (out,r) = classifier(im1,im2,model)
    write_file(lb,out,r)
    print("Classifying Instance",i, sep=" ")
for i in range(100,1000):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/output/cl000' + str(i) + gtext
    (out,r) = classifier(im1,im2,model)
    write_file(lb,out,r)
    print("Classifying Instance",i, sep=" ")
for i in range(1002,2051):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in00' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/output/cl00' + str(i) + gtext
    (out,r) = classifier(im1,im2,model)
    write_file(lb,out,r)
    print("Classifying Instance",i, sep=" ")