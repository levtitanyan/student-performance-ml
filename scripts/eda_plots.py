# scripts/eda.py

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="white")
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import math

def plot_missing_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Create and return a clean heatmap of missing values.

    Args:
        df (pd.DataFrame): The dataset to visualize.

    Returns:
        matplotlib.figure.Figure: The figure object of the plot.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        df.isna(),
        ax=ax,
        cbar=False,
        cmap="Blues",
        linewidths=0
    )

    ax.set_title("Missing Value Heatmap", fontsize=14, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels("")


    fig.tight_layout()
    return fig



def plot_numeric_boxplots(df: pd.DataFrame, num_cols: list, n_cols: int = 4) -> plt.Figure:
    """
    Plot boxplots for numerical columns in a grid layout.

    Args:
        df (pd.DataFrame): DataFrame containing numeric data.
        num_cols (list): List of numeric column names.
        n_cols (int): Number of columns in the plot grid.

    Returns:
        matplotlib.figure.Figure: The complete figure.
    """
    fig, axes = plt.subplots(
        nrows=math.ceil(len(num_cols) / n_cols),
        ncols=n_cols,
        figsize=(4.5 * n_cols, 3.5 * math.ceil(len(num_cols) / n_cols))
    )
    fig.suptitle("Boxplots of Numerical Features", fontsize=23, weight="bold")

    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(col, weight="bold", size=15)
        axes[i].set_xlabel("")  

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
    return fig




def plot_numeric_distributions(df: pd.DataFrame, num_cols: list, n_cols: int = 4) -> plt.Figure:
    """
    Plot histograms for numerical columns in a grid layout without x-axis labels or titles.

    Args:
        df (pd.DataFrame): DataFrame containing numeric data.
        num_cols (list): List of numeric column names.
        n_cols (int): Number of columns in the plot grid.

    Returns:
        matplotlib.figure.Figure: The complete figure.
    """
    import math
    fig, axes = plt.subplots(
        nrows=math.ceil(len(num_cols) / n_cols),
        ncols=n_cols,
        figsize=(4.5 * n_cols, 3.5 * math.ceil(len(num_cols) / n_cols))
    )
    fig.suptitle("Distributions of Numerical Features", fontsize=23, weight="bold")

    axes = axes.flatten()
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(col, weight="bold", size=15)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")        
        axes[i].set_xticklabels([])

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
    return fig



def plot_correlation_matrix(df: pd.DataFrame, num_cols: list, alpha: float = 0.01) -> plt.Figure:
    """
    Plot a correlation heatmap with all values visible, but annotate only statistically significant correlations.

    Args:
        df (pd.DataFrame): The dataset.
        num_cols (list): List of numerical columns.
        alpha (float): Significance threshold for correlation (default 0.01).

    Returns:
        matplotlib.figure.Figure: The correlation heatmap figure.
    """
    corr_matrix = df[num_cols].corr()
    p_matrix = pd.DataFrame(np.ones_like(corr_matrix), columns=num_cols, index=num_cols)

    # Compute p-values
    # Compute p-values
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            if i != j:
                x = df[num_cols[i]]
                y = df[num_cols[j]]
                valid = x.notna() & y.notna()
                if valid.sum() > 1:
                    _, p_val = pearsonr(x[valid], y[valid])
                    p_matrix.iloc[i, j] = p_val


    # Create annotation matrix: show only significant correlations
    annot_matrix = corr_matrix.round(2).astype(str)
    annot_matrix[p_matrix > alpha] = ""

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=annot_matrix,
        cmap="coolwarm",
        fmt="",
        linewidths=0.5,
        square=True,
        ax=ax
    )
    ax.set_title("Correlation Matrix (Significant Values Annotated)", fontsize=14, weight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    return fig


def plot_countplots(df: pd.DataFrame, cat_cols: list, n_cols: int = 5) -> plt.Figure:
    """
    Plot countplots for categorical features with consistent color and clean axes.

    Args:
        df (pd.DataFrame): DataFrame with categorical data.
        cat_cols (list): List of categorical column names.
        n_cols (int): Number of columns per row in the grid.

    Returns:
        matplotlib.figure.Figure: The figure containing all subplots.
    """
    import math
    fig, axes = plt.subplots(
        nrows=math.ceil(len(cat_cols) / n_cols),
        ncols=n_cols,
        figsize=(4.5 * n_cols, 3.5 * math.ceil(len(cat_cols) / n_cols)),
        constrained_layout=True
    )
    fig.suptitle("Countplots of Categorical Features", fontsize=25, weight="bold")

    axes = axes.flatten()
    for i, col in enumerate(cat_cols):
        sns.countplot(data=df, x=col, ax=axes[i], color="steelblue", order=df[col].value_counts().index)
        axes[i].set_title(col, fontsize=16, weight="bold")
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].set_xlabel("")  # Remove x-axis label/title

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    return fig

