from sklearn.neural_network import MLPRegressor
import pickle

def train_model(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=500)
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
