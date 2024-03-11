from tensorflow.python.tpu import datasets


def load_data():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y
# Compare this snippet from src/utils.py:
# import numpy as np