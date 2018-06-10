import tflearn
from supportLibrary import prepareData3

def netinit():
    print("Initializing the net........")
    net = tflearn.input_data(shape=[None, 9])
    net = tflearn.fully_connected(net, 36, activation='sigmoid')
    net = tflearn.fully_connected(net, 18, activation='sigmoid')
    net = tflearn.fully_connected(net, 9, activation='sigmoid')
    net = tflearn.fully_connected(net, 5, activation='sigmoid')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', loss='binary_crossentropy', learning_rate=0.006)
    model = tflearn.DNN(net)
    return model

def trainData(model, data, labels):
    model.fit(data, labels, show_metric=True, n_epoch=1000)
    return model




im1 = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/cell2.png'
l = '/Users/arnabsen/PycharmProjects/ImageProcessingPython/IMAGE_DB/changeSD.png'
(a, l) = prepareData3(im1, l, 3, 3)

model = netinit()
model = trainData(model, a, l)
print(model)