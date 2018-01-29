import tflearn
data = [[0, 0], [0, 1], [1, 0],[1, 1]]
labels = [[0], [1], [1], [0]]
net = tflearn.input_data(shape=[None, 2])
net = tflearn.fully_connected(net, 2, activation='tanh')
net = tflearn.fully_connected(net, 1, activation='tanh')
net = tflearn.regression(net, optimizer='sgd', learning_rate=5, loss='binary_crossentropy')
model = tflearn.DNN(net)
model.fit(data, labels, show_metric=True, n_epoch=5000)