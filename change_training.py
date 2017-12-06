from change_knn import *
im1 = input("Enter the file name of original image:  ")
im2 = input("Enter the filename of changed image:  ")
lb = input("Enter the filename of ground truth image:  ")
(dif,lbl,r) = read_image(im1,im2,lb)
model = model_load("med_change.knndata")
a = model.fit(dif,lbl)
model_save("med_change.knndata",model)
print("Model sucessfully trained")
