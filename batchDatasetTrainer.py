from change_knn import  *
#This file is designed for the batch training of Data.
#The data split is as follows 85% Test Set 15% For training Data
globalDif = np.array(int)
globalLab = np.array(int)
imext = '.jpg'
gtext = '.pgm'
im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001' + imext
lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/groundtruth/gt000001' +  gtext
(dif, lbl, r) = read_image(im1, im2, lb)
np.concatenate((globalDif, dif))
np.concatenate((globalDif, lbl))
#Initailing Model
#Creating a model of Neighbours = 7
model = knn_trainer(dif, lbl, 7)
for i in range(12,100,6):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in0000' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/groundtruth/gt0000' + str(i) + gtext
    np.concatenate((globalDif, dif))
    np.concatenate((globalDif, lbl))
    print("Training Instance",i, sep=" ")
for i in range(102,1000,6):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/groundtruth/gt000' + str(i) + gtext
    np.concatenate((globalDif, dif))
    np.concatenate((globalDif, lbl))
    print("Training Instance",i, sep=" ")
for i in range(1002,2051,6):
    im1 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in000001.jpg'
    im2 = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/input/in00' + str(i) + imext
    lb = '/Users/arnabsen/Documents/IMAGEPROCESSING_PYTHON/office/groundtruth/gt00' + str(i) + gtext
    np.concatenate((globalDif, dif))
    np.concatenate((globalDif, lbl))
    print("Training Instance",i, sep=" ")
model_save("kNNOfficek7.knndata",model)