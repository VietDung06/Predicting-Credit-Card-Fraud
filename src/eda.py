from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import OUTPUT_DIR, TARGET_COLUMN, TRAIN_FILE


def save_class_balance(df: pd.DataFrame) -> None:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    ax = counts.plot(kind="bar", color=["#4C78A8", "#E45756"])
    ax.set_title("Phân bố nhãn giao dịch")
    ax.set_xlabel("is_fraud")
    ax.set_ylabel("Số lượng")
    ax.set_xticklabels(["Không gian lận", "Gian lận"], rotation=0)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_balance.png", dpi=150)
    plt.close(fig)


def save_distribution(df: pd.DataFrame) -> None:
    candidates = [
        column
        for column in [
            "amount_usd",
            "velocity_score",
            "merchant_risk_score",
            "distance_from_home_km",
        ]
        if column in df.columns
    ]
    if not candidates:
        return
    sample = df[candidates + [TARGET_COLUMN]].sample(
        min(len(df), 50_000),
        random_state=42,
    )
    fig, axes = plt.subplots(
        len(candidates),
        1,
        figsize=(9, 3.2 * len(candidates)),
        constrained_layout=True,
    )
    if len(candidates) == 1:
        axes = [axes]
    for ax, column in zip(axes, candidates):
        sns.kdeplot(
            data=sample,
            x=column,
            hue=TARGET_COLUMN,
            common_norm=False,
            fill=True,
            ax=ax,
        )
        ax.set_title(f"Phân phối {column} theo nhãn")
    fig.savefig(OUTPUT_DIR / "important_distributions.png", dpi=150)
    plt.close(fig)


def save_correlation(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number").drop(
        columns=["id"],
        errors="ignore",
    )
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=False,
        ax=ax,
    )
    ax.set_title("Ma trận tương quan")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)


def save_feature_importance() -> None:
    model_file = __import__("src.config", fromlist=["MODEL_FILE"]).MODEL_FILE
    if not model_file.exists():
        return
    import joblib

    artifact = joblib.load(model_file)
    importances = []
    for pipeline in artifact["models"]:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        names = preprocessor.get_feature_names_out()
        importances.append(
            pd.Series(classifier.feature_importances_, index=names)
        )
    average = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False).head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    average.sort_values().plot(kind="barh", ax=ax, color="#72B7B2")
    ax.set_title("Feature importance trung bình (XGBoost)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(TRAIN_FILE))
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    save_class_balance(df)
    save_distribution(df)
    save_correlation(df)
    save_feature_importance()
    print(f"EDA plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
