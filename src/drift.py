import numpy as np

def calculate_psi(expected, actual, bins=10):
    def scale(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

    expected = scale(expected)
    actual = scale(actual)

    breakpoints = np.linspace(0, 1, bins + 1)

    psi = 0
    for i in range(bins):
        e = ((expected >= breakpoints[i]) & (expected < breakpoints[i+1])).sum()
        a = ((actual >= breakpoints[i]) & (actual < breakpoints[i+1])).sum()

        e = e / len(expected) + 1e-6
        a = a / len(actual) + 1e-6

        psi += (a - e) * np.log(a / e)

    return psi


def detect_drift(X_ref, X_new, threshold=0.2):
    drifted = []
    for i in range(X_ref.shape[1]):
        psi = calculate_psi(X_ref[:, i], X_new[:, i])
        if psi > threshold:
            drifted.append((i, psi))
    return drifted


def compute_psi_matrix(X_ref, batch_list):
    psi_matrix = []

    for X_new in batch_list:
        row = []
        for i in range(X_ref.shape[1]):
            row.append(calculate_psi(X_ref[:, i], X_new[:, i]))
        psi_matrix.append(row)

    return np.array(psi_matrix)
