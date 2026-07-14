import numpy as np


class Dense:


    def __init__(self, inputs, neurons):

        self.weights = np.random.randn(
            inputs,
            neurons
        ) * np.sqrt(2 / inputs)


        self.biases = np.zeros(
            (1, neurons)
        )


    def forward(self, inputs):

        self.inputs = inputs

        self.output = (
            np.dot(inputs,self.weights)
            +
            self.biases
        )

        return self.output



    def backward(self, gradient):

        self.dweights = np.dot(
            self.inputs.T,
            gradient
        )


        self.dbiases = np.sum(
            gradient,
            axis=0,
            keepdims=True
        )


        self.dinputs = np.dot(
            gradient,
            self.weights.T
        )


        return self.dinputs