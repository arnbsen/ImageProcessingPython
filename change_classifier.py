from change_knn import *
model = model_load("med_change.knndata")
im1 = input("Enter the filename of Original Image:   ")
im2 = input("Enter the filename of Changed Image:    ")
(out,r) = classifier(im1,im2,model)
filename = input("Enter the filename to save (including format (.png preferably)")
write_file(filename,out,r)
print("File successfully written")