import matplotlib.pyplot as plt

def plot_contributions(contributions, top_n=5):
    # Sort by absolute contribution
    sorted_items = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    colors = ["red" if v > 0 else "green" for v in values]

    plt.figure(figsize=(8, 5))
    plt.barh(features, values, color=colors)
    plt.xlabel("Contribution to Prediction")
    plt.title("Top Feature Contributions")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
