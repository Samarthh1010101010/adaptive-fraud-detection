import matplotlib.pyplot as plt
import numpy as np
import os

def plot_psi_heatmap(psi_matrix):
    os.makedirs("results", exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    
    im = ax.imshow(psi_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, ax=ax, label="PSI")
    
    ax.set_title("PSI Drift Heatmap (Features × Batches)")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Batch")

    ax.set_yticks(range(psi_matrix.shape[0]))
    ax.set_yticklabels([f"Batch {i+1}" for i in range(psi_matrix.shape[0])])

    # Highlight drift cells
    threshold = 0.2
    for i in range(psi_matrix.shape[0]):
        for j in range(psi_matrix.shape[1]):
            if psi_matrix[i, j] > threshold:
                ax.add_patch(plt.Rectangle(
                    (j-0.5, i-0.5), 1, 1,
                    fill=False, edgecolor='red', linewidth=1.5
                ))

    plt.tight_layout()
    plt.savefig("results/psi_heatmap.png", dpi=150)
    print("✓ PSI heatmap saved to results/psi_heatmap.png")
    plt.close()
