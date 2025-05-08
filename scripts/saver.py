# scripts/saver.py

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Base paths
ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT_DIR / "outputs" / "plots"
REPORTS_DIR = ROOT_DIR / "outputs" / "reports"
DATA_DIR = ROOT_DIR / "data"

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(fig: plt.Figure, name: str) -> None:
    """
    Save a matplotlib figure to outputs/plots/.

    Args:
        fig (plt.Figure): The figure to save.
        name (str): Filename without extension.
    """
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Plot saved to: {path}")

def save_dataset(df: pd.DataFrame, filename: str, folder: str = "reports") -> None:
    """
    Save a DataFrame as a CSV file to outputs/reports/ by default,
    or to data/ if folder='data' is specified.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Filename, must include .csv.
        folder (str): 'reports' (default) or 'data'.
    """
    if folder == "data":
        path = DATA_DIR / filename
    else:
        path = REPORTS_DIR / filename

    df.to_csv(path, index=False)
    print(f"Dataset saved to: {path}")
