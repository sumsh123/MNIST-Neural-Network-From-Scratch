import numpy as np


class ReLU:

    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)


    def backward(self, gradient):

        self.dinputs = gradient.copy()

        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs



class Softmax:

    def forward(self, inputs):

        self.inputs = inputs

        exp_values = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )

        probabilities = exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )

        self.output = probabilities

        return probabilities



    def backward(self, gradient):

        self.dinputs = np.empty_like(
            gradient
        )


        for index, single_output in enumerate(self.output):

            single_output = single_output.reshape(-1,1)


            jacobian_matrix = np.diagflat(
                single_output
            ) - np.dot(
                single_output,
                single_output.T
            )


            self.dinputs[index] = np.dot(
                jacobian_matrix,
                gradient[index]
            )


        return self.dinputs