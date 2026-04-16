from src.data_loader import load_data
from src.model import train_model, evaluate_model
from src.simulate_drift import simulate_drift
from src.drift import detect_drift, compute_psi_matrix
from src.adapt import SlidingWindowAdapter
from src.evaluate import plot_performance
from src.visualize_drift import plot_psi_heatmap

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = load_data("data/creditcard.csv")

model = train_model(X_train, y_train)

adapter = SlidingWindowAdapter(window_size=4)

scores = []
drift_points = []
batch_data_list = []

for batch in range(1, 9):
    print(f"\n--- Batch {batch} ---")

    X_batch, y_batch = simulate_drift(X_test, y_test, batch)

    # Split batch (IMPORTANT)
    X_b_train, X_b_eval, y_b_train, y_b_eval = train_test_split(
        X_batch, y_batch, test_size=0.3, random_state=42
    )
drift(X_train, X_b_train)

    if drifted:
        print(f"Drift detected at batch {batch}: {drifted}")
        drift_points.append(batch)

    # Evaluate BEFORE adaptation
    score = evaluate_model(model, X_b_eval, y_b_eval)
    scores.append(score)

    # Store for PSI heatmap
    batch_data_list.append(X_b_train)

    batch_data_list.append(X_b_train)

    adapter.update(X_b_train, y_b_train)
h sliding window...")
        model = train_model(X_adapt, y_adapt)

# Plots
plot_performance(scores, drift_points)

psi_matrix = compute_psi_matrix(X_train, batch_data_list)
plot_psi_heatmap(psi_matrix)
