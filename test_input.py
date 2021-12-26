from MLP_from_scratch import MLP

#some test inputs for simple MLP
inputs = [1, -1]
labels = [1, 0]

#and some (non-randomly) initialized weights + bias to check whether the gradients are correct
w1 = [[1., 1., 1.], [-1., -1., -1.]]
w2 = [[1., 1.], [-1., -1.], [-1., -1.]]
b1 = [0., 0., 0.]
b2 = [0., 0.]

model = MLP(2, 3, 2)

#a single forward and backward pass to check gradients
h, yh = model.forward(inputs, labels, w1, b1, w2, b2)
d_w1, d_w2, d_b1, d_b2 = model.backward(inputs, labels, yh, w1, w2, h)

print("Derivative wrt W, b: ", d_w1, d_b1)
print("Derivative wrt V, c: ", d_w2, d_b2)