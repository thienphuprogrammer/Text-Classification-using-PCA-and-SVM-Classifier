from sklearn.datasets import fetch_20newsgroups


def load_data_train():
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train


def load_data_test():
    newsgroups_test = fetch_20newsgroups(subset='test')
    return newsgroups_test

# Path: src/dataloader/__init__.py


