from typing import Dict, Tuple
import numpy as np


# class that encapsulates the PCA model
class PCA(object):
    # initializer function
    def __init__(self, desired_principal_components: int = 2) -> None:
        self.desired_principal_components = desired_principal_components
        self.eigen_values = np.array([])
        self.eigen_vectors = np.array([])
        self.extracted_eigenvalues = np.array([])
        self.column_index = np.array([])
        self.feature_mean = np.array([])

    def __del__(self):
        del self.desired_principal_components
        del self.eigen_values
        del self.eigen_vectors
        del self.extracted_eigenvalues
        del self.column_index
        del self.feature_mean

    # function to transform the data
    # public function to train the model
    def fit(self, X_train: np.array) -> None:
        # since I'm assuming the input matrix X_train has shape (samples,features), compute the transpose
        X = np.transpose(X_train)
        # remove the mean from each feature
        X -= np.mean(X, axis=1).reshape(-1, 1)
        # compute the covariance matrix of X
        C = np.matmul(X, np.transpose(X))
        # find the eigenvalues & eigenvectors of the covariance
        self.eigen_values, self.eigen_vectors = np.linalg.eig(C)
        # sort the negated eigenvalues (to get sort in descending order)
        self.column_index = np.argsort(-self.eigen_values)

    def transform(self, X) -> np.array:
        # check that the model has been trained?
        if self.eigen_values.size and self.eigen_vectors.size and self.column_index.size:
            # since I'm assuming the input matrix X_train has shape (samples,features), compute the transpose
            X = np.transpose(X)
            # project data onto the PCs
            X_new = np.matmul(np.transpose(self.eigen_vectors), X)
            # transform back via tranpose + sort by variance explained
            X_new = np.transpose(X_new)
            X_new = X_new[:, self.column_index]
            # if n_components was specified, return only this number of features back
            if self.desired_principal_components:
                X_new = X_new[:, :self.desired_principal_components]
            # return
            return X_new
        else:
            print('Empty eigenvectors and eigenvalues, did you forget to train the model?')
            return np.array([])

    # public function to return % explained variance per PC
    def explained_variance_ratio(self) -> np.array:
        # check that the model has been trained?
        if self.eigen_values.size and self.column_index.size:
            # compute the sorted % explained variances
            perc = self.eigen_values[self.column_index]
            perc = perc / np.sum(perc)
            # if n_components was specified, return only this number of features back
            if self.desired_principal_components != 0:
                perc = perc[:self.desired_principal_components]
            # return
            return perc
        else:
            print('Empty eigenvalues, did you forget to train the model?')
            return np.array([])

    # public function to return the eigenvalues & eigenvectors
    def return_eigen_vectors_values(self) -> Tuple[np.array, np.array]:
        # check that the model has been trained?
        if self.eigen_values.size and self.eigen_vectors.size and self.column_index.size:
            # sort the eigenvalues and eigenvectors
            e_val = self.eigen_values[self.column_index]
            e_vec = self.eigen_vectors[:, self.column_index]
            # if n_components was specified, return only this number of features back
            if self.desired_principal_components != 0:
                e_val = e_val[:self.desired_principal_components]
                e_vec = e_vec[:, :self.desired_principal_components]
            # return
            return e_vec, e_val
        else:
            print('Empty eigenvalues & eigenvectors, did you forget to train the model?')
            return np.array([]), np.array([])


