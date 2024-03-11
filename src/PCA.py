import numpy as np


class PCA:
    def __init__(self, desired_principal_components):
        self.desired_principal_components = desired_principal_components
        self.extracted_eigenvalues = None
        self.feature_mean  = None

    def fit(self, feature_table):
        # mean center the feature table
        self.feature_mean = np.mean(feature_table, axis=0)
        feature_table -= self.feature_mean - self.feature_mean
        # calculate the covariance matrix
        # row = 1 sample, column = 1 feature
        covariance_matrix = np.cov(feature_table.T)  # transpose the feature table
        # calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        # sort the eigenvalues and eigenvectors
        eigenvectors = eigenvectors.T
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[indices]
        # extract the desired principal components
        self.extracted_eigenvalues = eigenvalues[:self.desired_principal_components]
        eigenvectors = eigenvectors[:self.desired_principal_components]
        # transform the feature table
        feature_table = np.dot(feature_table, eigenvectors.T)
        return feature_table

    def transform(self, feature_table):
        # mean center the feature table
        feature_table = feature_table - self.feature_mean
        # transform the feature table
        feature_table = np.dot(feature_table, self.extracted_eigenvalues.T)
        return feature_table
