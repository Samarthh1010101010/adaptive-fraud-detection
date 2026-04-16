from collections import deque
import numpy as np

class SlidingWindowAdapter:
    def __init__(self, window_size=4):
        self.window = deque(maxlen=window_size)

    def update(self, X, y):
        self.window.append((X, y))

    def get_training_data(self):
        X_all = np.vstack([x for x, _ in self.window])
        y_all = np.hstack([y for _, y in self.window])
        return X_all, y_all
