"""
    This is the main file for the project. It will be used to run the project.
    The project is a Text Classification using PCA and SVM Classifier.

    The project is divided into 3 files:
    1. main.py
    2. NeuralNetwork.py
    3. PCA.py
    4. SVM.py
    5. data.py
    6. utils.py
    7. test.py
    8. requirements.txt
    9. README.md
    10. LICENSE
    11. .gitignore
    12. .travis.yml

    The main file will be used to run the project. It will import the necessary classes from the other files and use them to
    run the project.

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.PCA import PCA
from src.SVM import SVM_classifier
from src.NeuralNetwork import NeuralNetwork
from src.data import load_data

# Load the data
X, y = load_data()
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply PCA
pca = PCA()
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# SVM Classifier
svm = SVM_classifier()
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
print(f"SVM Accuracy: {np.mean(predictions == y_test)}")

# Neural Network
nn = NeuralNetwork(layers=[10, 5, 1], learning_rate=0.01, no_of_iterations=1000)
nn.fit(X_train, y_train)
predictions = nn.predict(X_test)
predictions = np.where(predictions > 0.5, 1, -1)
print(f"NN Accuracy: {np.mean(predictions == y_test)}")
# Plot the loss
plt.plot(nn.loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()