from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    auc,
    average_precision_score,
    brier_score_loss,
    calibration_curve,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def eval_metrics(
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    Y_test: np.ndarray,
    case_name: str,
    output_dir: Path,
    save_name: str,
) -> list:
    """
    Compute and persist a standard suite of binary classification metrics.
    """
    precision_curve, recall_curve, _ = precision_recall_curve(Y_test, y_proba)

    acc    = accuracy_score(Y_test, y_pred)
    prec   = precision_score(Y_test, y_pred, zero_division=0)
    rec    = recall_score(Y_test, y_pred, zero_division=0)
    fone   = f1_score(Y_test, y_pred, zero_division=0)
    auc_sc = roc_auc_score(Y_test, y_proba)
    avprec = average_precision_score(Y_test, y_proba)
    brier  = brier_score_loss(Y_test, y_proba)
    prauc  = auc(recall_curve, precision_curve)

    print(f"[{case_name}] acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  "
          f"f1={fone:.4f}  AUC={auc_sc:.4f}  AP={avprec:.4f}  "
          f"Brier={brier:.4f}  PR-AUC={prauc:.4f}")

    cm = confusion_matrix(Y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # --- Persist calibration and curve plots ---
    cal_dir = output_dir / "model_results" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    stem = cal_dir / f"output_{save_name}{case_name}"

    RocCurveDisplay.from_predictions(Y_test, y_proba)
    plt.savefig(f"{stem}_roc_curve.png")
    plt.clf()
    plt.close()

    PrecisionRecallDisplay.from_predictions(Y_test, y_proba)
    plt.savefig(f"{stem}_precrec_curve.png")
    plt.clf()
    plt.close()

    prob_true, prob_pred = calibration_curve(Y_test, y_pred, n_bins=10)
    plt.plot(prob_pred, prob_true, marker="o", linewidth=1, label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
    plt.title("Probability Calibration Curve")
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Probability")
    plt.legend(loc="best")
    plt.savefig(f"{stem}_calibration.png")
    plt.clf()
    plt.close()

    pd.DataFrame(prob_pred).to_csv(f"{stem}_calibration_prob_pred.csv", index=False)
    pd.DataFrame(prob_true).to_csv(f"{stem}_calibration_prob_true.csv", index=False)

    return [acc, prec, rec, fone, auc_sc, avprec, brier, prauc, tn, fp, fn, tp]


def threshold_func(
    Y_test: np.ndarray,
    y_proba: np.ndarray,
    output_dir: Path,
    save_name: str,
) -> tuple[np.ndarray, float, float]:
    """
    Select the classification threshold that maximises F1 on the PR curve.
    """
    precision, recall, thresholds = precision_recall_curve(Y_test, y_proba)
    precision = precision[:-1]
    recall    = recall[:-1]

    fscore = np.nan_to_num(2 * precision * recall / (precision + recall + 1e-8))
    ix = int(np.argmax(fscore))
    best_threshold = float(thresholds[ix])
    best_f1        = float(fscore[ix])
    y_pred_thres   = (y_proba >= best_threshold).astype(int)

    cal_dir = output_dir / "model_results" / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall curve")
    plt.scatter(recall[ix], precision[ix], color="red",
                label=f"Best Threshold: {best_threshold:.2f}", marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve with Optimal Threshold")
    plt.legend()
    plt.savefig(cal_dir / f"output_{save_name}_threshold.png")
    plt.clf()
    plt.close()

    return y_pred_thres, best_threshold, best_f1
