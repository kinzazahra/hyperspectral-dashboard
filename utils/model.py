from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def get_model(model_type="knn", k=3):
    if model_type == "knn":
        return KNeighborsClassifier(n_neighbors=k)
    elif model_type == "naive_bayes":
        return GaussianNB()

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model