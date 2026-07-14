import numpy as np

from layers import Dense
from activations import ReLU, Softmax
from loss import CrossEntropy
from network import NeuralNetwork
from model import save_model
from mnist_loader import load_data



def train():


    X_train,y_train,X_test,y_test = load_data()


    model = NeuralNetwork()


    model.add(Dense(784,128))
    model.add(ReLU())

    model.add(Dense(128,64))
    model.add(ReLU())

    model.add(Dense(64,10))
    model.add(Softmax())


    loss_function = CrossEntropy()


    epochs = 20

    batch_size = 64

    learning_rate = 0.01



    for epoch in range(epochs):


        permutation = np.random.permutation(
            len(X_train)
        )


        X_train = X_train[permutation]

        y_train = y_train[permutation]



        for i in range(
            0,
            len(X_train),
            batch_size
        ):


            X_batch = X_train[i:i+batch_size]

            noise = np.random.normal(
                0,
                0.02,
                X_batch.shape
            )

            X_batch = X_batch + noise

            X_batch = np.clip(
                X_batch,
                0,
                1
            )

            # random pixel shifting

            for index in range(len(X_batch)):

                image = X_batch[index].reshape(28,28)

                shift_x = np.random.randint(-2,3)
                shift_y = np.random.randint(-2,3)


                shifted = np.roll(
                    image,
                    (shift_y,shift_x),
                    axis=(0,1)
                )


                X_batch[index] = shifted.reshape(784)

            y_batch = y_train[i:i+batch_size]



            predictions = model.forward(
                X_batch
            )


            loss = loss_function.forward(
                predictions,
                y_batch
            )


            gradient = loss_function.backward(
                predictions,
                y_batch
            )


            model.backward(
                gradient
            )


            for layer in model.layers:


                if hasattr(layer,"weights"):


                    layer.weights -= (
                        learning_rate *
                        layer.dweights
                    )


                    layer.biases -= (
                        learning_rate *
                        layer.dbiases
                    )



        test_predictions = model.predict(
            X_test
        )


        accuracy = np.mean(
            test_predictions == y_test
        )


        print(
            f"Epoch {epoch+1}/{epochs}",
            "Accuracy:",
            accuracy
        )



    save_model(model)

    print("Model saved!")



    return model