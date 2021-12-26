import random
from data import load_synth
from MLP_from_scratch import MLP

#load the data
(x_train, y_train), (x_val, y_val), num_cls = load_synth()

lr = 0.001
n_epochs = 20
model = MLP(2, 3, 2)

#random intialization of weights
w1 = [[random.gauss(0, 1) for i in range(3)] for i in range(2)]
w2 = [[random.gauss(0, 1) for i in range(2)] for i in range(3)]

#initialize the rest with zeros
b1 = [0., 0., 0.]
b2 = [0., 0.]
d_y = [0.,0.]
d_w2 = [[0.0, 0.0],[0.0, 0.0],[0.0, 0.0]]
d_h = [0.0, 0.0, 0.0]
d_b1 = [0.0, 0.0, 0.0]
d_w1 = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
d_k = [0.0, 0.0, 0.0]
d_b1 = [0., 0., 0.]
d_b2 = [0., 0.]


#one hot encode the labels
labels = []
[labels.append([0,1]) if i == 0 else labels.append([1,0]) for i in y_train]

model.train(x_train, labels, w2, b2, w1, b1, d_w2, d_b2, d_w1, d_b1, lr, n_epochs)