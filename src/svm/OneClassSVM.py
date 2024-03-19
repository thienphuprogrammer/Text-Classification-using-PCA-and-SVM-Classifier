import numpy as np
from src.svm.SVM import SVM


class OneClassSVM:
    def __init__(self, kernel, gamma, nu):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.support_vectors = None
        self.dual_coef = None
        self.intercept = None

    def fit(self, X):
        n_samples, n_features = X.shape
        # calculate the kernel matrix
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j], self.gamma)
        # calculate the dual coefficients
        dual_coef = SVM.fit(X, kernel_matrix, self.nu)
        # calculate the support vectors
        support_vectors = X[dual_coef > 0]
        # calculate the intercept
        intercept = SVM.intercept(support_vectors, dual_coef, kernel_matrix, self.nu)
        self.support_vectors = support_vectors
        self.dual_coef = dual_coef
        self.intercept = intercept

    def predict(self, X):
        n_samples, n_features = X.shape
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            prediction = 0
            for j in range(len(self.support_vectors)):
                prediction += self.dual_coef[j] * self.kernel(self.support_vectors[j], X[i], self.gamma)
            prediction -= self.intercept
            y_pred[i] = np.sign(prediction)
        return y_pred
