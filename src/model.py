from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def train_model(X, y):
    model = LogisticRegression(max_iter=2000, class_weight='balanced')
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    preds = model.predict(X)
    return f1_score(y, preds)
