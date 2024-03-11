from tensorflow.python.tpu import datasets


class load_data:
    def __init__(self):
        pass

    def load_data(self):
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        return X, y