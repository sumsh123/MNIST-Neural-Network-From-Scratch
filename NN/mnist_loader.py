from sklearn.datasets import fetch_openml
import numpy as np


def load_data():

    print("Loading MNIST dataset...")


    mnist = fetch_openml(
        "mnist_784",
        version=1
    )


    X = mnist.data.to_numpy(
        dtype=np.float32
    )


    y = mnist.target.to_numpy(
        dtype=np.int64
    )


    X /= 255.0



    X_train = X[:60000]
    X_test = X[60000:]


    y_train = y[:60000]
    y_test = y[60000:]


    return (
        X_train,
        y_train,
        X_test,
        y_test
    )