"""
STEP 3 - Test the model on an area it has never seen (Montauban).

Usage:
    python step3_evaluate.py            # evaluates outputs/model_rf.joblib
    python step3_evaluate.py --model hgb

Creates in outputs/:
    metrics_<model>.txt         accuracy, F1 per class
    confusion_<model>.png       which classes get mixed up
    map_<model>.png             WorldCover vs our prediction, side by side
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)

from common import CLASS_COLORS, CLASSES, DATA, NO_CLASS, OUT


def predict_map(model, X, valid, chunk=200_000):
    """Predict every valid pixel, in chunks to keep memory low."""
    pred = np.full(valid.shape, NO_CLASS, dtype="uint8")
    rows, cols = np.nonzero(valid)
    for i in range(0, len(rows), chunk):
        r, c = rows[i:i + chunk], cols[i:i + chunk]
        pred[r, c] = model.predict(X[r, c])
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    parser.add_argument("--area", default="montauban")
    args = parser.parse_args()

    model = joblib.load(OUT / f"model_{args.model}.joblib")
    d = np.load(DATA / f"{args.area}.npz")
    X, labels, valid = d["X"], d["labels"], d["valid"]

    print(f"Predicting {valid.sum():,} pixels over {args.area}...")
    pred = predict_map(model, X, valid)

    scored = valid & (labels != NO_CLASS)
    y_true, y_pred = labels[scored], pred[scored]
    present = sorted(set(y_true) | set(y_pred))
    names = [CLASSES[c] for c in present]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, labels=present, target_names=names, digits=3)
    text = (f"Model: {args.model}  |  test area: {args.area} ({d['date']})\n"
            f"Overall accuracy: {acc:.3f}\nMacro F1: {f1:.3f}\n\n{report}")
    print("\n" + text)
    (OUT / f"metrics_{args.model}.txt").write_text(text, encoding="utf-8")

    # Confusion matrix (rows = truth, normalised so each row sums to 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=present, display_labels=names,
                                            normalize="true", values_format=".2f",
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {args.area} ({args.model.upper()})")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_{args.model}.png", dpi=150)
    plt.close()

    # Side-by-side maps
    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, arr, title in [(axes[0], labels, "ESA WorldCover 2021 (reference)"),
                           (axes[1], pred, f"Our prediction ({args.model.upper()})")]:
        ax.imshow(np.ma.masked_equal(arr, NO_CLASS), cmap=cmap, vmin=0,
                  vmax=len(CLASSES) - 1, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLORS]
    fig.legend(handles, CLASSES, loc="lower center", ncol=len(CLASSES), frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    plt.savefig(OUT / f"map_{args.model}.png", dpi=150)
    plt.close()
    print(f"Saved metrics, confusion matrix and map in {OUT}/")


if __name__ == "__main__":
    main()
