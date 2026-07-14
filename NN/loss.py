import numpy as np


class CrossEntropy:


    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        y_pred_clipped = np.clip(
            y_pred,
            1e-7,
            1-1e-7
        )


        if len(y_true.shape) == 1:

            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        else:

            correct_confidences = np.sum(
                y_pred_clipped*y_true,
                axis=1
            )


        negative_log = -np.log(
            correct_confidences
        )


        return np.mean(
            negative_log
        )



    def backward(self, y_pred, y_true):

        samples = len(y_pred)

        labels = len(y_pred[0])


        if len(y_true.shape)==1:

            y_true = np.eye(labels)[y_true]


        self.dinputs = (
            -y_true / y_pred
        ) / samples


        return self.dinputs