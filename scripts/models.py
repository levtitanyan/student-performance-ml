import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    explained_variance_score
)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, results_df):
    """
    Evaluate a regression model on train and test data and append results to results_df.

    Args:
        model: Trained regression model
        X_train, X_test: Feature sets
        y_train, y_test: Target sets
        model_name: Name of the model (str)
        results_df: Existing results DataFrame to append to

    Returns:
        Updated results DataFrame
    """
    for subset, X, y_true in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        row = {
            "Model": f"{model_name} {subset}",
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred)
        }
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df = results_df.round(3)
        results_df = results_df.sort_values(
                                by=["Model", "R2"],
                                key=lambda col: col.str.endswith("Test") if col.name == "Model" else col,
                                ascending=[False, False]
                                ).reset_index(drop=True)
    return results_df


def plot_predictions(model, X_test, y_test, model_name):
    """
    Create and return a predicted vs. actual plot for the test set.

    Args:
        model: Trained regression model
        X_test: Test features
        y_test: True target values
        model_name: Name of the model

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter actual vs. predicted
    ax.scatter(y_test, y_pred, alpha=0.75, color="darkblue", s=40, edgecolors='w', linewidths=0.5)

    # 45-degree reference line
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="red", linewidth=1.5)

    # Styling
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(f"{model_name}: Actual vs Predicted (Test)", fontsize=13)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    return fig


def find_optimal_params(model_name, X_train, y_train, X_test, y_test):
    """
    Find optimal hyperparameters for the given model by maximizing R² on test set.
    Also returns a matplotlib plot of R² vs. the tuned parameter(s).

    Args:
        model_name (str): One of ['Ridge', 'Lasso', 'ElasticNet', 'Random Forest']
        X_train, y_train, X_test, y_test: Data splits

    Returns:
        best_model: Trained model instance with best parameters
        best_params: Dict of best parameter values
        fig: matplotlib Figure object of the R² plot
    """
    best_model = None
    best_r2 = -np.inf
    best_params = {}
    fig = None

    if model_name == "Ridge":
        alphas = np.arange(0.0, 100.1, 0.5)
        r2_scores = []
        for alpha in alphas:
            model = Ridge(alpha=alpha).fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            r2_scores.append(r2)
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_params = {"alpha": round(alpha, 2)}

        fig, ax = plt.subplots()
        ax.plot(alphas, r2_scores, color="darkblue")
        ax.set_title("Ridge: R² vs Alpha")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("R²")
        ax.grid(True)

    elif model_name == "Lasso":
        alphas = np.arange(0.0, 100.1, 0.1)
        r2_scores = []
        for alpha in alphas:
            model = Lasso(alpha=alpha).fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            r2_scores.append(r2)
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_params = {"alpha": round(alpha, 2)}

        fig, ax = plt.subplots()
        ax.plot(alphas, r2_scores, color="darkblue")
        ax.set_title("Lasso: R² vs Alpha")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("R²")
        ax.grid(True)

    elif model_name == "ElasticNet":
        alphas = np.arange(0.0, 100.1, 0.5)
        l1_ratios = np.arange(0.0, 1.01, 0.1)
        r2_scores = []
        combos = []

        for alpha in alphas:
            for l1_ratio in l1_ratios:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
                r2 = r2_score(y_test, model.predict(X_test))
                r2_scores.append(r2)
                combos.append((alpha, l1_ratio))
                if r2 > best_r2:
                    best_model = model
                    best_r2 = r2
                    best_params = {"alpha": round(alpha, 2), "l1_ratio": round(l1_ratio, 2)}

        scores_matrix = np.array(r2_scores).reshape(len(alphas), len(l1_ratios))
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, l1 in enumerate(l1_ratios):
            ax.plot(alphas, scores_matrix[:, i], label=f"l1_ratio={l1:.1f}")
        ax.set_title("ElasticNet: R² vs Alpha for different l1_ratio")
        ax.set_xlabel("Alpha")
        ax.set_ylabel("R²")
        ax.legend()
        ax.grid(True)

    elif model_name == "Random Forest":
        estimators = range(10, 301, 1)
        r2_scores = []
        for n in estimators:
            model = RandomForestRegressor(n_estimators=n, random_state=42).fit(X_train, y_train)
            r2 = r2_score(y_test, model.predict(X_test))
            r2_scores.append(r2)
            if r2 > best_r2:
                best_model = model
                best_r2 = r2
                best_params = {"n_estimators": n}

        fig, ax = plt.subplots()
        ax.plot(estimators, r2_scores, color="darkblue")
        ax.set_title("Random Forest: R² vs n_estimators")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("R²")
        ax.grid(True)

    else:
        raise ValueError("Model not recognized. Use: 'Ridge', 'Lasso', 'ElasticNet', 'Random Forest'.")

    print(f"\n {model_name} with parameters: {best_params} | Best R²: {best_r2:.3f}")
    return best_model, best_params, fig
