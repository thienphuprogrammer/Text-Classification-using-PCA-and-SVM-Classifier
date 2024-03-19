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
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# Preprocess the text data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# Tokenize the text and remove stopwords
from nltk.tokenize import word_tokenize

# Load the 20newsgroups dataset
newsgroups_data = fetch_20newsgroups(subset='all')

corpus = newsgroups_data.data
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

tokenized_corpus = [word_tokenize(review) for review in corpus]
filtered_corpus = [[word for word in review if word not in stop_words] for review in tokenized_corpus]

# Convert the text into a document-term matrix with term frequency
vectorizer = TfidfVectorizer(use_idf=True)
X = vectorizer.fit_transform(filtered_corpus)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)

# Split the data into training and test sets
X_train, X_test = train_test_split(X_pca, test_size=0.2, random_state=42)

# Train the OCSVM classifier using the negative class data
y_train = [0] * len(X_train)
ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma=0.01)
ocsvm.fit(X_train, y_train)

# Test the OCSVM classifier using the positive class data
y_test = [1] * len(X_test)
y_pred = ocsvm.predict(X_test)

# Print the classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
