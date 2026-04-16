import matplotlib.pyplot as plt
import os

def plot_performance(scores, drift_points):
    os.makedirs("results", exist_ok=True)

    batches = list(range(1, len(scores) + 1))

    plt.figure()

    plt.plot(batches, scores, marker='o')
    plt.title("Model Performance Over Time with Drift")
    plt.xlabel("Batch")
    plt.ylabel("F1 Score")

    for dp in drift_points:
        plt.axvline(x=dp, linestyle='--')
        plt.text(dp, min(scores), f"Drift @ {dp}", rotation=90)

    plt.tight_layout()
    plt.savefig("results/performance.png", dpi=150)
    print("✓ Performance plot saved to results/performance.png")
    plt.close()
