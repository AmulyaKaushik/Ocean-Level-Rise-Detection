# dashboard_xgb.py
import pandas as pd
import matplotlib.pyplot as plt

def generate_dashboard():
    # Load accuracy results
    acc_df = pd.read_csv("model_accuracy_xgb.csv")

    # Cities and their forecast plots
    cities = [
        ("San_Francisco_CA", "plots/San_Francisco_CA_forecast_xgb.png"),
        ("New_Orleans_LA", "plots/New_Orleans_LA_forecast_xgb.png")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for i, (city, img_path) in enumerate(cities):
        # Show plots as images
        img = plt.imread(img_path)
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(city.replace("_", " "), fontsize=14, fontweight="bold")

    plt.suptitle("Sea Level Forecasting (XGBoost) – 30d Evaluation + 7d Forecast",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    # Save combined plots
    plt.savefig("plots/combined_forecasts.png", dpi=300)
    plt.close()
    print("Saved combined forecast plots to plots/combined_forecasts.png")

    # Accuracy table visualization
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis("off")
    table = ax.table(cellText=acc_df.values,
                     colLabels=acc_df.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.title("Model Accuracy (Evaluation: Last 30 Days)", fontsize=12, fontweight="bold")
    plt.savefig("plots/accuracy_table.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved accuracy table to plots/accuracy_table.png")

 
if __name__ == "__main__":
    generate_dashboard()
