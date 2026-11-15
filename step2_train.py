"""
STEP 2 - Train a land-cover classifier on the Toulouse area.

Usage:
    python step2_train.py            # Random Forest (default)
    python step2_train.py --model hgb  # Histogram Gradient Boosting, to compare

Creates outputs/model_<name>.joblib and outputs/feature_importance.png
"""
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from common import CLASSES, DATA, FEATURES, NO_CLASS, OUT

PER_CLASS = 4000   # pixels sampled per class (balanced training set)
SEED = 42


def sample_pixels(X, labels, valid, per_class, rng):
    """Pick up to per_class random labelled pixels for each class."""
    Xs, ys = [], []
    usable = valid & (labels != NO_CLASS)
    for cls in range(len(CLASSES)):
        rows, cols = np.nonzero(usable & (labels == cls))
        if len(rows) == 0:
            print(f"  warning: no pixels for class {CLASSES[cls]}")
            continue
        pick = rng.choice(len(rows), size=min(per_class, len(rows)), replace=False)
        Xs.append(X[rows[pick], cols[pick]])
        ys.append(np.full(len(pick), cls))
        print(f"  {CLASSES[cls]:<12} {len(rows):>8} pixels available, {len(pick)} sampled")
    return np.concatenate(Xs), np.concatenate(ys)


def make_model(name):
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=SEED)
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  n_jobs=-1, random_state=SEED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "hgb"], default="rf")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = np.load(DATA / "toulouse.npz")
    print("Sampling training pixels from Toulouse:")
    X, y = sample_pixels(d["X"], d["labels"], d["valid"], PER_CLASS, rng)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = make_model(args.model)
    print(f"\nTraining {type(model).__name__} on {len(X_tr)} pixels...")
    model.fit(X_tr, y_tr)

    present = sorted(set(y_val))
    report = classification_report(y_val, model.predict(X_val), labels=present,
                                   target_names=[CLASSES[c] for c in present], digits=3)
    print("\nValidation inside Toulouse (same area, different pixels):\n" + report)

    joblib.dump(model, OUT / f"model_{args.model}.joblib")

    # Which features matter most?
    imp = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=SEED, n_jobs=-1)
    order = np.argsort(imp.importances_mean)
    plt.figure(figsize=(7, 5))
    plt.barh(np.array(FEATURES)[order], imp.importances_mean[order], color="#2f5496")
    plt.xlabel("Drop in accuracy when the feature is shuffled")
    plt.title(f"Feature importance ({args.model.upper()})")
    plt.tight_layout()
    plt.savefig(OUT / f"feature_importance_{args.model}.png", dpi=150)
    plt.close()
    print(f"Saved outputs/model_{args.model}.joblib and outputs/feature_importance_{args.model}.png")


if __name__ == "__main__":
    main()
