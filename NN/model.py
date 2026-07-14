import pickle


def save_model(model):

    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)



def load_model():

    with open("model.pkl", "rb") as file:
        return pickle.load(file)