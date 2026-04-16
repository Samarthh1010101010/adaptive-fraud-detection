import numpy as np

def simulate_drift(X, y, batch_id):
    X_new = X.copy()
    y_new = y.copy()

    if batch_id <= 2:
        return X_new, y_new

    if batch_id in [3, 4]:
        X_new[:, 4] *= 0.8
        X_new[:, 11] += 0.8

    if batch_id == 5:
        X_new[:, 4] *= 0.5
        X_new[:, 11] += 2.0
        X_new[:, 7] *= -0.6

    if batch_id >= 6:
        X_new[:, 4] *= 0.5
        X_new[:, 11] += 2.0
        X_new[:, 7] *= -0.6

    return X_new, y_new
