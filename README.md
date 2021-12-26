## Simple MLP from scratch

A python implementation of a simple MLP from scratch without any libraries (only math module).

The MLP:
- input layer with 2 nodes
- 1 hidden linear layer with 3 nodes (linear layer followed by sigmoid activation) + 1 bias node
- output layer with 2 nodes  (with softmax activation) + 1 bias node

test_inputs.py consist of a test input to check the output of the network and performs a single forward and backward pass.

train.py trains the neural network on synthetic training data.