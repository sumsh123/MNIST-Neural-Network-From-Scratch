import numpy as np


class NeuralNetwork:


    def __init__(self):

        self.layers = []


    def add(self, layer):

        self.layers.append(layer)



    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output



    def backward(self, gradient):

        for layer in reversed(self.layers):

            gradient = layer.backward(
                gradient
            )



    def predict(self, X):

        output = self.forward(X)

        return np.argmax(
            output,
            axis=1
        )