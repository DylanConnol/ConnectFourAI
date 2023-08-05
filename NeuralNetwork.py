import numpy as np


class NeuralNetwork:
    def __init__(self, neurons_in_layer: tuple):
        self.neurons_in_layer = neurons_in_layer
        self.num_of_layers = len(neurons_in_layer)
        self.weights = [2 * (np.random.random((neurons_in_layer[i + 1], neurons_in_layer[i])) - 0.5) for i in
                        range(0, self.num_of_layers - 1)]
        self.biases = [2 * (np.random.random((neurons_in_layer[i], 1)) - 0.5) for i in range(1, len(neurons_in_layer))]

    def Relu(self, Z):
        return np.maximum(0, Z)

    def Derivative_Relu(self, Z):
        return Z > 0

    def sigmoid(self, Z):
        # return np.maximum(Z, 0)
        # Probably this:
        return 1.0 / (1.0 + np.exp(-Z))

    def Derivative_sigmoid(self, Z):
        # return Z > 0
        return self.sigmoid(Z) * (1 - self.sigmoid(Z))

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps)

    def forwardprop(self, inp, last_activation=False):
        self.inp = inp
        a = []
        z = []
        # [(32, 54), (64, 32), (128, 64), (12, 128)]
        for i in range(self.num_of_layers - 1):
            current_z = self.weights[i].dot(a[-1]) + self.biases[i] if i != 0 else self.weights[i].dot(inp) + \
                                                                                   self.biases[i]
            if i != self.num_of_layers - 2 or last_activation:
                current_a = self.sigmoid(current_z)
            else:
                current_a = current_z
            # current_a = self.sigmoid(current_z)
            z.append(current_z)
            a.append(current_a)
        self.z = z
        self.a = a
        return a[-1]

    def backprop(self, target, learning_rate, last_activation=False):
        a = self.a
        z = self.z
        deltaL = []
        dWL = [0] * (self.num_of_layers - 1)
        dBL = [0] * (self.num_of_layers - 1)

        for i in range(self.num_of_layers - 2, -1, -1):
            current_delta = self.Derivative_sigmoid(z[i]) * (
                self.weights[i + 1].T.dot(deltaL[-1])) if i != self.num_of_layers - 2 \
                else ((a[-1] - target) * (self.Derivative_sigmoid(z[-1]) if last_activation else 1))
            djdw = current_delta.dot(a[i - 1].T) if i != 0 else (current_delta).dot(self.inp.T)
            djdb = current_delta
            dWL[i] = (djdw)
            dBL[i] = (djdb)
            deltaL.append(current_delta)
        loss = np.sum(np.square(z[-1] - target))
        # self.weights = [self.weights[i] - learning_rate * (dWL[i]) for i in range(len(self.weights))]
        # self.biases = [self.biases[i] - learning_rate * (dBL[i]) for i in range(len(self.biases))]

        return dWL, dBL, loss

    def gradient_descent(self, minibatch: list[tuple], lr):
        d = len(minibatch)
        overallloss = 0
        for x, y in minibatch:
            self.forwardprop(x)
            a, b, loss = self.backprop(y, lr)
            overallloss += (loss / d)
            try:
                z = [m + n for m, n in zip(a, z)]
                o = [m + n for m, n in zip(b, o)]
            except:
                z = a.copy()
                o = b.copy()
        self.weights = [self.weights[i] - lr / d * (z[i]) for i in range(len(self.weights))]
        self.biases = [self.biases[i] - lr / d * (o[i]) for i in range(len(self.biases))]
        return overallloss
