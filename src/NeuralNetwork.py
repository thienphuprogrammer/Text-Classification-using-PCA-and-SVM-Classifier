import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.001, no_of_iterations=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.weights = []
        self.bias = []
        self.activations = []
        self.loss = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = [np.random.randn(n_features, self.layers[0])]
        self.bias = [np.zeros((1, self.layers[0]))]
        for i in range(1, len(self.layers)):
            self.weights.append(np.random.randn(self.layers[i - 1], self.layers[i]))
            self.bias.append(np.zeros((1, self.layers[i])))
        self.weights.append(np.random.randn(self.layers[-1], 1))
        self.bias.append(np.zeros((1, 1)))

        for _ in range(self.no_of_iterations):
            # forward pass
            activations = []
            input = X
            for i in range(len(self.layers)):
                input = np.dot(input, self.weights[i]) + self.bias[i]
                activations.append(input)
            self.activations = activations
            # calculate the loss
            loss = self.mean_squared_error(y, input)
            self.loss.append(loss)
            # backward pass
            error = y - input
            for i in range(len(self.layers), 0, -1):
                self.weights[i] += self.learning_rate * np.dot(self.activations[i - 1].T, error)
                self.bias[i] += self.learning_rate * np.sum(error, axis=0, keepdims=True)
                error = np.dot(error, self.weights[i].T)
        return self.loss

    def predict(self, X):
        for i in range(len(self.layers)):
            X = np.dot(X, self.weights[i]) + self.bias[i]
        return X

    @staticmethod
    def mean_squared_error(y, input):
        return np.mean((y - input) ** 2)
