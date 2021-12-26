import math
from matplotlib import pyplot as plt

class MLP():
    
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

    def sigmoid(self, x):
        return 1/ (1+math.e ** -x)

    def softmax(self, yh):
        s = 0
        x = []
        for i in yh:
            x.append(math.e**i)
            s += math.e**i

        result = []
        for i in x:
            result.append(i/s)
        return result

    def d_softmax(self, yh, y):
        dy = [0.]*2
        for i in range(len(yh)):
            dy[i] = yh[i] - y[i]
        return dy

    def calculate_loss(self, yh, y):
        loss = 0
        for i, pred in enumerate(yh):
            loss += (y[i] * (-math.log(pred)))
        return loss

    def forward(self, x, y, w1, b1, w2, b2):

        #outputs of the linear layer
        k = [0.]*self.n_hidden
        #outputs of the hidden layer
        h = [0.]*self.n_hidden
        #outputs after activation
        yh = [0.]*self.n_outputs

        #input layer * w1 + bias
        for j in range(self.n_hidden):
            for i in range(self.n_inputs):
                k[j] += w1[i][j] * x[i]
            k[j] += b1[j]

        #activation function
        for i in range(self.n_hidden):
            h[i] = self.sigmoid(k[i])

        #w2 * output layer + bias
        for j in range(self.n_outputs):
            for i in range(self.n_hidden):
                yh[j] += w2[i][j] * h[i]
            yh[j] += b2[j]

        #softmax
        yh = self.softmax(yh)

        return h, yh

    def backward(self, x, y, yh, w1, w2, h):
        d_w2 = [[0 for i in range(self.n_outputs)] for i in range(self.n_hidden)]
        d_h = [0.]*self.n_hidden
        d_k = [0.]*self.n_hidden
        d_b1 = [0.]*self.n_hidden
        d_w1 = [[0 for i in range(self.n_hidden)] for i in range(self.n_inputs)]
        d_y = [0.]*self.n_outputs
        
        #softmax derivative
        d_y = self.d_softmax(yh, y)

        #derivative second weights, hidden, bias 2nd layer
        for j in range(self.n_outputs):
            for i in range(self.n_hidden): 
                d_w2[i][j] = d_y[j] * h[i]
                d_h[i] += d_y[j] * w2[i][j]  
            d_b2 = d_y

        #derivative input layer
        for i in range(self.n_hidden):
            d_k[i] = d_h[i] * h[i] * (1-h[i])

        #derivative first weights, bias 1st layer
        for j in range(self.n_hidden):
            for i in range(self.n_inputs):
                d_w1[i][j] = d_k[j] * x[i]

            d_b1[j] = d_k[j]

        return d_w1, d_w2, d_b1, d_b2

    def update(self, d_w1, d_w2, d_b1, d_b2, w1, w2, b1, b2, lr):
        #update 2nd layer weights      
        for i in range(self.n_outputs):
            for j in range(self.n_hidden):
                w2[j][i] = w2[j][i] - (lr * d_w2[j][i])

            b2[i] = b2[i] - (lr * d_b2[i])

        #update 1st layer weights
        for i in range(self.n_hidden):
            for j in range(self.n_inputs):
                w1[j][i] = w1[j][i] - (lr * d_w1[j][i])

            b1[i] = b1[i] - (lr * d_b1[i])

        return w1, w2, b1, b2

    def train(self, x, y, w2, b2, w1, b1, d_w2, d_b2, d_w1, d_b1, lr, n_epochs):
        epoch_loss = []

        for j in range(n_epochs):

            for i in range(len(x)):
                losses = 0
                x_train = x[i]
                y_train = y[i]
                h, yh = self.forward(x_train, y_train, w1, b1, w2, b2)
                loss = self.calculate_loss(yh, y_train)
                losses += loss
                d_w1, d_w2, d_b1, d_b2 = self.backward(x_train, y_train, yh, w1, w2, h)
                w1, w2, b1, b2 = self.update(d_w1, d_w2, d_b1, d_b2, w1, w2, b1, b2, lr)
                
            #add avg loss per epoch
            epoch_loss.append(losses/len(x))
            print("Epoch ", j, "Train loss: ", loss)

        plt.plot(epoch_loss, label = "Training")
        plt.title("Loss")
        plt.ylabel("Loss")
        plt.xlabel("Number of epochs")