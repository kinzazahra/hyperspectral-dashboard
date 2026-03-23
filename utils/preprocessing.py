import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def flatten_data(image, gt):
    X = image.reshape(-1, image.shape[2])
    y = gt.reshape(-1)

    mask = y > 0
    return X[mask], y[mask]

def split_data(X, y, test_size=0.3):
    return train_test_split(X, y, test_size=test_size, random_state=42)

def apply_pca(X_train, X_test, n_components=20):
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test, pca