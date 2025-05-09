# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Evaluation metrics
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error, median_absolute_error,
    mean_absolute_percentage_error, explained_variance_score,
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix)


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


def find_optimal_classification_params(model_name, X_train, y_train, X_test, y_test):
    """
    Find optimal hyperparameters for a classification model by maximizing F1 score on test set.

    Args:
        model_name (str): One of ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'Gradient Boosting', 'Naive Bayes']
        X_train, y_train, X_test, y_test: Classification train/test splits

    Returns:
        best_model: trained classifier
        best_params: dict of best parameters
        fig: matplotlib figure of F1 vs. hyperparameter(s)
    """

    best_model = None
    best_f1 = -np.inf
    best_params = {}
    fig = None

    if model_name == "Logistic Regression":
        Cs = np.arange(0.01, 5.1, 0.1)
        f1_scores = []
        for c in Cs:
            model = LogisticRegression(C=c, max_iter=1000).fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            f1_scores.append(f1)
            if f1 > best_f1:
                best_model = model
                best_f1 = f1
                best_params = {"C": round(c, 2)}

        fig, ax = plt.subplots()
        ax.plot(Cs, f1_scores, color="darkblue")
        ax.set_title("Logistic Regression: F1 vs C")
        ax.set_xlabel("C")
        ax.set_ylabel("F1 Score")
        ax.grid(True)

    elif model_name == "KNN":
        ks = range(1, 31)
        f1_scores = []
        for k in ks:
            model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            f1_scores.append(f1)
            if f1 > best_f1:
                best_model = model
                best_f1 = f1
                best_params = {"n_neighbors": k}

        fig, ax = plt.subplots()
        ax.plot(ks, f1_scores, color="darkblue")
        ax.set_title("KNN: F1 vs n_neighbors")
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel("F1 Score")
        ax.grid(True)

    elif model_name == "Decision Tree":
        depths = range(1, 21)
        f1_scores = []
        for d in depths:
            model = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            f1_scores.append(f1)
            if f1 > best_f1:
                best_model = model
                best_f1 = f1
                best_params = {"max_depth": d}

        fig, ax = plt.subplots()
        ax.plot(depths, f1_scores, color="darkblue")
        ax.set_title("Decision Tree: F1 vs max_depth")
        ax.set_xlabel("max_depth")
        ax.set_ylabel("F1 Score")
        ax.grid(True)

    elif model_name == "Random Forest":
        estimators = range(10, 201, 10)
        f1_scores = []
        for n in estimators:
            model = RandomForestClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            f1_scores.append(f1)
            if f1 > best_f1:
                best_model = model
                best_f1 = f1
                best_params = {"n_estimators": n}

        fig, ax = plt.subplots()
        ax.plot(estimators, f1_scores, color="darkblue")
        ax.set_title("Random Forest: F1 vs n_estimators")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("F1 Score")
        ax.grid(True)

    elif model_name == "Gradient Boosting":
        estimators = range(10, 201, 10)
        f1_scores = []
        for n in estimators:
            model = GradientBoostingClassifier(n_estimators=n, random_state=42).fit(X_train, y_train)
            f1 = f1_score(y_test, model.predict(X_test))
            f1_scores.append(f1)
            if f1 > best_f1:
                best_model = model
                best_f1 = f1
                best_params = {"n_estimators": n}

        fig, ax = plt.subplots()
        ax.plot(estimators, f1_scores, color="darkblue")
        ax.set_title("Gradient Boosting: F1 vs n_estimators")
        ax.set_xlabel("n_estimators")
        ax.set_ylabel("F1 Score")
        ax.grid(True)

    elif model_name == "Naive Bayes":
        model = GaussianNB().fit(X_train, y_train)
        f1 = f1_score(y_test, model.predict(X_test))
        best_model = model
        best_f1 = f1
        best_params = {}
        fig = None
        print("Naive Bayes has no tunable parameters.")

    else:
        raise ValueError("Model not recognized.")

    print(f"\n {model_name} Best Params: {best_params} | Best F1: {best_f1:.3f}")
    return best_model, best_params, fig


def evaluate_classifier(model, X_train, X_test, y_train, y_test, model_name, results_df):

    """
    Train and evaluate a classification model on both train and test sets.

    Args:
        model: Instantiated classifier
        X_train, X_test: Feature sets
        y_train, y_test: Label sets
        model_name (str): Model name for logging
        results_df: DataFrame to append results to

    Returns:
        model: fitted model
        results_df: updated results table
    """
    model.fit(X_train, y_train)

    for subset, X, y_true in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

        row = {
            "Model": f"{model_name} {subset}",
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        }
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
        results_df = results_df.round(3)
        results_df = results_df.sort_values(
                                by=["Model", "F1"],
                                key=lambda col: col.str.endswith("Test") if col.name == "Model" else col,
                                ascending=[False, False]
                                ).reset_index(drop=True)

    return results_df.round(3)


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Create and return a confusion matrix plot for a trained classification model.

    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
        model_name (str): Model name to display on the plot

    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name}: Confusion Matrix", fontsize=12, fontweight="bold")
    fig.tight_layout()

    return fig
