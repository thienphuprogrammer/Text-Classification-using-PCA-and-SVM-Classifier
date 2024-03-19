import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, no_of_iterations=1000, lambda_param=0.01):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_param = lambda_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.no_of_iterations):
            for idx, x in enumerate(X):
                linear_output = np.dot(x, self.weights) + self.bias
                # update the weights
                if y[idx] * linear_output >= 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)

    @classmethod
    def intercept(cls, support_vectors, dual_coef, kernel_matrix, nu):
        intercept = 0
        for i in range(len(support_vectors)):
            prediction = 0
            for j in range(len(support_vectors)):
                prediction += dual_coef[j] * kernel_matrix[j, i]
            intercept += support_vectors[i] - prediction
        return intercept / nu