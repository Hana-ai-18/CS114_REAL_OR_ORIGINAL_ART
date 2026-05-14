"""
03_train_models.py
==================
Train và so sánh nhiều ML model trên 3 pipeline features:
  Pipeline A: Fractal + Wavelet + GLCM
  Pipeline B: FFT + Noise + LBP
  Pipeline C: A + B + HOG + Color + Statistical

Models:
  1. Random Forest (RF)
  2. SVM với RBF kernel
  3. SVM Linear
  4. Gradient Boosting (XGBoost-style với sklearn)
  5. Extra Trees
  6. AdaBoost
  7. Logistic Regression
  8. KNN
  9. Naive Bayes (GaussianNB)

Kết quả được so sánh và lưu chi tiết.
"""

import pickle
import json
import warnings
import time
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report,
                              precision_score, recall_score, roc_curve,
                              precision_recall_curve)
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
import joblib


# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
FEATURES_DIR  = Path("processed/features")
MODELS_DIR    = Path("models")
PLOTS_DIR     = Path("plots")
RESULTS_FILE  = Path("processed/results.json")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Real Art", "AI Art"]

# ─────────────────────────────────────────
# ĐỊNH NGHĨA MODELS
# ─────────────────────────────────────────
def get_model_definitions():
    """
    Trả về dict {name: (estimator, needs_scaling, description)}
    needs_scaling: True → dùng StandardScaler trước khi fit
    """
    return {
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, max_depth=None,
                                   min_samples_leaf=2, max_features='sqrt',
                                   n_jobs=-1, random_state=42, class_weight='balanced'),
            False,
            "Ensemble cây quyết định, robust với features không cùng scale"
        ),
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, max_depth=None,
                                 min_samples_leaf=2, max_features='sqrt',
                                 n_jobs=-1, random_state=42, class_weight='balanced'),
            False,
            "Extremely Randomized Trees — nhanh hơn RF, thường tương đương"
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                       max_depth=4, subsample=0.8,
                                       min_samples_leaf=4, random_state=42),
            False,
            "Sequential boosting — mạnh nhưng chậm hơn RF"
        ),
        "AdaBoost": (
            AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                               random_state=42, algorithm='SAMME'),
            False,
            "Adaptive Boosting — baseline boosting method"
        ),
        "SVM RBF": (
            SVC(kernel='rbf', C=10, gamma='scale', probability=True,
                class_weight='balanced', random_state=42),
            True,
            "SVM với RBF kernel — rất mạnh sau khi scale features"
        ),
        "SVM Linear": (
            SVC(kernel='linear', C=1.0, probability=True,
                class_weight='balanced', random_state=42),
            True,
            "SVM tuyến tính — nhanh, hiệu quả với high-dim features"
        ),
        "Logistic Regression": (
            LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs',
                               class_weight='balanced', random_state=42),
            True,
            "Baseline probabilistic model — interpretable"
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=7, metric='euclidean', n_jobs=-1),
            True,
            "K-Nearest Neighbors — non-parametric, đơn giản"
        ),
        "Naive Bayes": (
            GaussianNB(),
            True,
            "Gaussian Naive Bayes — giả định independence"
        ),
    }


# ─────────────────────────────────────────
# LOAD & CHUẨN BỊ DATA
# ─────────────────────────────────────────
def load_pipeline_data(pipeline_name: str) -> dict:
    path = FEATURES_DIR / f"{pipeline_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_data(data: dict, apply_pca: bool = False, pca_var: float = 0.95):
    """
    Ghép train+val để train cuối cùng trên nhiều data hơn.
    Trả về X_train, y_train, X_test, y_test
    """
    X_train = np.vstack([data["train"]["X"], data["val"]["X"]])
    y_train = np.concatenate([data["train"]["y"], data["val"]["y"]])
    X_test  = data["test"]["X"]
    y_test  = data["test"]["y"]

    # Xử lý NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0, posinf=1e6, neginf=-1e6)
    X_test  = np.nan_to_num(X_test,  nan=0, posinf=1e6, neginf=-1e6)

    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────
def train_and_evaluate(model_name: str, estimator, needs_scaling: bool,
                        X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        pipeline_name: str) -> dict:
    """
    Train model, đánh giá trên test set.
    Trả về dict với đầy đủ metrics.
    """
    # Xây pipeline sklearn
    steps = []
    if needs_scaling:
        steps.append(("scaler", RobustScaler()))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)

    # Train
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predict
    t0 = time.time()
    y_pred   = pipe.predict(X_test)
    infer_ms = (time.time() - t0) * 1000 / len(X_test)

    # Probabilities (nếu có)
    try:
        y_prob = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm   = confusion_matrix(y_test, y_pred).tolist()

    # ROC curve
    roc_data = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    print(f"    {model_name:20s} | acc={acc:.4f} | f1={f1:.4f} | "
          f"auc={auc:.4f if auc else 'N/A':>6} | "
          f"train={train_time:.1f}s | infer={infer_ms:.2f}ms/img")

    return {
        "model_name"   : model_name,
        "pipeline"     : pipeline_name,
        "accuracy"     : acc,
        "precision"    : prec,
        "recall"       : rec,
        "f1"           : f1,
        "auc"          : auc,
        "confusion_matrix": cm,
        "train_time_s" : train_time,
        "infer_ms_img" : infer_ms,
        "roc_data"     : roc_data,
        "pipe"         : pipe,   # sklearn pipeline object
    }


# ─────────────────────────────────────────
# VISUALIZATION — COMPARISON PLOTS
# ─────────────────────────────────────────

PALETTE = {
    "Random Forest"      : "#00d4aa",
    "Extra Trees"        : "#00a8ff",
    "Gradient Boosting"  : "#ffd700",
    "AdaBoost"           : "#ff9944",
    "SVM RBF"            : "#ff6b6b",
    "SVM Linear"         : "#ff44aa",
    "Logistic Regression": "#aa88ff",
    "KNN"                : "#66ffcc",
    "Naive Bayes"        : "#aaaaaa",
}

PIPELINE_COLORS = {
    "pipeline_A": "#55aaff",
    "pipeline_B": "#ff6b6b",
    "pipeline_C": "#ffd700",
}


def plot_model_comparison(all_results: list, save_path: str = "plots/03_model_comparison.png"):
    """Mega-plot so sánh tất cả models × pipelines"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    pipelines = ["pipeline_A", "pipeline_B", "pipeline_C"]
    model_names = list(get_model_definitions().keys())
    metrics = ["accuracy", "f1", "auc", "train_time_s"]
    metric_labels = ["Accuracy", "F1-Score", "AUC-ROC", "Train Time (s)"]

    # Index results
    idx = {}
    for r in all_results:
        idx[(r["pipeline"], r["model_name"])] = r

    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#0a0a15')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    TC = '#e0e0e0'
    fig.text(0.5, 0.97, "📊 Model Comparison — All Pipelines × All Models",
             ha='center', fontsize=18, color='white', fontweight='bold', fontfamily='monospace')

    # ── Plot 1-3: Accuracy/F1/AUC grouped bar ────
    for metric_idx, (metric, mlabel) in enumerate(zip(["accuracy", "f1", "auc"], metric_labels[:3])):
        ax = fig.add_subplot(gs[0, metric_idx])
        ax.set_facecolor('#1a1a2e')

        x = np.arange(len(model_names))
        width = 0.25
        for i, pipe in enumerate(pipelines):
            vals = []
            for mn in model_names:
                r = idx.get((pipe, mn))
                v = r[metric] if r and r[metric] is not None else 0
                vals.append(v)
            offset = (i - 1) * width
            bars = ax.bar(x + offset, vals, width=width,
                          label=pipe, color=PIPELINE_COLORS[pipe],
                          alpha=0.85, edgecolor='#0a0a15', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([mn.replace(' ', '\n') for mn in model_names],
                           fontsize=7, color=TC)
        ax.set_ylabel(mlabel, color=TC, fontsize=10)
        ax.set_title(mlabel, color='white', fontsize=12, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.spines[:].set_color('#333355')
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color='#555', linestyle=':', linewidth=1)
        if metric_idx == 2:
            legend = ax.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=8)

    # ── Plot 4: Heatmap accuracy ─────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.set_facecolor('#1a1a2e')
    heatmap_data = np.zeros((len(pipelines), len(model_names)))
    for i, pipe in enumerate(pipelines):
        for j, mn in enumerate(model_names):
            r = idx.get((pipe, mn))
            heatmap_data[i, j] = r["accuracy"] if r else 0

    im = ax.imshow(heatmap_data, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels([mn.replace(' ', '\n') for mn in model_names], fontsize=6.5, color=TC, rotation=0)
    ax.set_yticks(range(len(pipelines)))
    ax.set_yticklabels(pipelines, color=TC, fontsize=8)
    ax.set_title("Accuracy Heatmap", color='white', fontsize=11, fontweight='bold')
    for i in range(len(pipelines)):
        for j in range(len(model_names)):
            ax.text(j, i, f"{heatmap_data[i,j]:.3f}", ha='center', va='center',
                    fontsize=7, color='black' if heatmap_data[i,j] > 0.7 else 'white', fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── Plot 5: Train time comparison ────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.set_facecolor('#1a1a2e')

    # Lấy best pipeline cho mỗi model (pipeline_C)
    times = []
    labels = []
    colors_bar = []
    for mn in model_names:
        r = idx.get(("pipeline_C", mn))
        if r:
            times.append(r["train_time_s"])
            labels.append(mn)
            colors_bar.append(PALETTE.get(mn, '#aaa'))

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, times, color=colors_bar, alpha=0.85, edgecolor='#0a0a15')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, color=TC)
    ax.set_xlabel("Train Time (seconds)", color='#888', fontsize=9)
    ax.set_title("Training Time (Pipeline C)", color='white', fontsize=11, fontweight='bold')
    ax.tick_params(colors='#888')
    ax.spines[:].set_color('#333355')
    for i, t in enumerate(times):
        ax.text(t + 0.1, i, f"{t:.1f}s", va='center', color=TC, fontsize=7)

    # ── Plot 6: Accuracy vs F1 scatter ───────────
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor('#1a1a2e')

    for pipe in pipelines:
        accs, f1s, names = [], [], []
        for mn in model_names:
            r = idx.get((pipe, mn))
            if r:
                accs.append(r["accuracy"])
                f1s.append(r["f1"])
                names.append(mn[:3])
        sc = ax.scatter(accs, f1s, label=pipe, s=120,
                        color=PIPELINE_COLORS[pipe], alpha=0.8, edgecolors='white', linewidth=0.5)
        for x_pt, y_pt, nm in zip(accs, f1s, names):
            ax.annotate(nm, (x_pt, y_pt), xytext=(3, 3), textcoords='offset points',
                        fontsize=6.5, color=TC)

    ax.set_xlabel("Accuracy", color='#888', fontsize=9)
    ax.set_ylabel("F1 Score", color='#888', fontsize=9)
    ax.set_title("Accuracy vs F1  (bubble = pipeline)", color='white', fontsize=11, fontweight='bold')
    ax.plot([0.5,1.0],[0.5,1.0], '--', color='#555', linewidth=1)
    ax.tick_params(colors='#888')
    ax.spines[:].set_color('#333355')
    ax.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=8)

    # ── Plot 7-9: ROC curves per pipeline ────────
    for pipe_idx, pipe in enumerate(pipelines):
        ax = fig.add_subplot(gs[2, pipe_idx])
        ax.set_facecolor('#1a1a2e')

        for mn in model_names:
            r = idx.get((pipe, mn))
            if r and r.get("roc_data"):
                fpr = r["roc_data"]["fpr"]
                tpr = r["roc_data"]["tpr"]
                auc = r["auc"]
                ax.plot(fpr, tpr, label=f"{mn[:15]} ({auc:.3f})",
                        color=PALETTE.get(mn, '#aaa'), linewidth=1.5, alpha=0.85)

        ax.plot([0,1],[0,1], '--', color='#555', linewidth=1)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("FPR", color='#888', fontsize=9)
        ax.set_ylabel("TPR", color='#888', fontsize=9)
        ax.set_title(f"ROC — {pipe}", color=PIPELINE_COLORS[pipe], fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.spines[:].set_color('#333355')
        ax.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=6.5, loc='lower right')
        ax.fill_between([0,1],[0,1], alpha=0.05, color='white')

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_confusion_matrices(all_results: list, pipeline: str = "pipeline_C",
                             save_path: str = "plots/03_confusion_matrices.png"):
    """Vẽ confusion matrix cho tất cả models trên 1 pipeline"""
    results_p = [r for r in all_results if r["pipeline"] == pipeline]
    n = len(results_p)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4.5))
    fig.patch.set_facecolor('#0a0a15')
    fig.suptitle(f"Confusion Matrices — {pipeline}",
                 color='white', fontsize=14, fontweight='bold', y=1.01)

    axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()

    for ax, r in zip(axes_flat, results_p):
        ax.set_facecolor('#1a1a2e')
        cm = np.array(r["confusion_matrix"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(CLASS_NAMES, color='#ccc', fontsize=9)
        ax.set_yticklabels(CLASS_NAMES, color='#ccc', fontsize=9, rotation=90, va='center')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha='center', va='center',
                        color='black' if cm_norm[i,j] > 0.5 else 'white',
                        fontsize=9, fontweight='bold')

        acc = r["accuracy"]
        f1  = r["f1"]
        ax.set_title(f"{r['model_name']}\nacc={acc:.3f} | f1={f1:.3f}",
                     color='white', fontsize=9, fontweight='bold')

    # Tắt axes trống
    for ax in axes_flat[len(results_p):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_best_model_detail(best_result: dict, X_train: np.ndarray, y_train: np.ndarray,
                            save_path: str = "plots/03_best_model_detail.png"):
    """Chi tiết model tốt nhất: feature importance, PR curve, metrics radar"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor('#0a0a15')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    TC = '#e0e0e0'

    model_name = best_result["model_name"]
    pipeline   = best_result["pipeline"]
    fig.text(0.5, 0.97,
             f"🏆 Best Model: {model_name}  ({pipeline})  — Detailed Analysis",
             ha='center', fontsize=14, color='#ffd700', fontweight='bold')

    # ── Metrics bar ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#1a1a2e')
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    vals = [best_result.get(m, 0) or 0 for m in metrics]
    colors = ['#00d4aa','#00aaff','#ff9944','#ffd700','#ff6b6b']
    bars = ax1.bar(metrics, vals, color=colors, edgecolor='#0a0a15', width=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha='center', va='bottom', color=TC, fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Performance Metrics", color=TC, fontsize=11, fontweight='bold')
    ax1.tick_params(colors='#888')
    ax1.spines[:].set_color('#333355')
    ax1.axhline(0.9, color='yellow', linestyle='--', alpha=0.4, linewidth=1)

    # ── ROC ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#1a1a2e')
    if best_result.get("roc_data"):
        fpr = best_result["roc_data"]["fpr"]
        tpr = best_result["roc_data"]["tpr"]
        ax2.plot(fpr, tpr, color='#00d4aa', linewidth=2.5,
                 label=f"AUC = {best_result['auc']:.4f}")
        ax2.fill_between(fpr, tpr, alpha=0.1, color='#00d4aa')
    ax2.plot([0,1],[0,1],'--', color='#555', linewidth=1.5)
    ax2.set_xlabel("False Positive Rate", color='#888')
    ax2.set_ylabel("True Positive Rate", color='#888')
    ax2.set_title("ROC Curve", color=TC, fontsize=11, fontweight='bold')
    ax2.tick_params(colors='#888')
    ax2.spines[:].set_color('#333355')
    ax2.legend(facecolor='#1a1a2e', labelcolor=TC, fontsize=10)

    # ── Radar chart ──────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2], polar=True)
    ax3.set_facecolor('#1a1a2e')
    categories = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    vals_radar  = [best_result.get(m, 0) or 0
                   for m in ["accuracy","precision","recall","f1","auc"]]
    N = len(categories)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    vals_radar += vals_radar[:1]
    ax3.fill(angles, vals_radar, alpha=0.25, color='#00d4aa')
    ax3.plot(angles, vals_radar, color='#00d4aa', linewidth=2)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories, color=TC, fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax3.set_yticklabels(['0.25','0.5','0.75','1.0'], color='#555', fontsize=7)
    ax3.grid(color='#333355')
    ax3.set_title("Radar — Performance", color=TC, fontsize=11, fontweight='bold', pad=15)
    ax3.set_facecolor('#1a1a2e')

    # ── Feature Importance (nếu RF/ET/GB) ───────
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor('#1a1a2e')
    pipe_obj = best_result["pipe"]
    model_obj = pipe_obj.named_steps.get("model")
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        top_k = min(25, len(importances))
        top_idx = np.argsort(importances)[-top_k:]
        top_imp = importances[top_idx]
        cmap_vals = plt.cm.plasma(top_imp / top_imp.max())
        ax4.barh(range(top_k), top_imp, color=cmap_vals)
        ax4.set_yticks(range(top_k))
        ax4.set_yticklabels([f"feat_{i}" for i in top_idx], fontsize=7.5, color=TC)
        ax4.set_xlabel("Importance", color='#888', fontsize=9)
        ax4.set_title(f"Top {top_k} Feature Importances", color=TC, fontsize=11, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, f"{model_name}\ndoes not expose\nfeature_importances_",
                 ha='center', va='center', color=TC, fontsize=12)
    ax4.tick_params(colors='#888')
    ax4.spines[:].set_color('#333355')

    # ── Summary text ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#1a1a2e')
    ax5.set_xticks([]); ax5.set_yticks([])
    ax5.spines[:].set_color('#333355')

    summary_lines = [
        f"🏆 BEST MODEL SUMMARY",
        f"",
        f"Model      : {model_name}",
        f"Pipeline   : {pipeline}",
        f"",
        f"Accuracy   : {best_result['accuracy']:.4f}",
        f"Precision  : {best_result['precision']:.4f}",
        f"Recall     : {best_result['recall']:.4f}",
        f"F1 Score   : {best_result['f1']:.4f}",
        f"AUC-ROC    : {best_result['auc']:.4f}" if best_result['auc'] else "AUC-ROC    : N/A",
        f"",
        f"Train time : {best_result['train_time_s']:.2f}s",
        f"Infer time : {best_result['infer_ms_img']:.3f}ms/img",
    ]
    for i, line in enumerate(summary_lines):
        color = '#ffd700' if i == 0 else '#00d4aa' if ':' in line else TC
        ax5.text(0.05, 0.95 - i * 0.065, line,
                 transform=ax5.transAxes, fontsize=9.5,
                 color=color, fontfamily='monospace', va='top')

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


def plot_pipeline_ranking(all_results: list, save_path: str = "plots/03_pipeline_ranking.png"):
    """So sánh trung bình các pipeline"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    pipelines = ["pipeline_A", "pipeline_B", "pipeline_C"]
    metrics = ["accuracy", "f1", "auc"]
    metric_labels = ["Accuracy", "F1-Score", "AUC-ROC"]

    pipeline_avg = {}
    for pipe in pipelines:
        rs = [r for r in all_results if r["pipeline"] == pipe]
        pipeline_avg[pipe] = {
            m: np.mean([r[m] for r in rs if r.get(m) is not None])
            for m in metrics
        }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a15')
    fig.suptitle("Pipeline Comparison — Average Performance Across All Models",
                 color='white', fontsize=13, fontweight='bold')

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        ax.set_facecolor('#1a1a2e')
        vals = [pipeline_avg[p][metric] for p in pipelines]
        bars = ax.bar(pipelines, vals,
                      color=[PIPELINE_COLORS[p] for p in pipelines],
                      edgecolor='#0a0a15', width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{val:.4f}", ha='center', va='bottom', color='white',
                    fontsize=11, fontweight='bold')
        ax.set_ylim(max(0, min(vals) - 0.05), 1.05)
        ax.set_ylabel(mlabel, color='#ccc')
        ax.set_title(mlabel, color='white', fontsize=11, fontweight='bold')
        ax.tick_params(colors='#888')
        ax.spines[:].set_color('#333355')
        ax.set_xticklabels(
            ["A: Fractal+\nWavelet+GLCM",
             "B: FFT+\nNoise+LBP",
             "C: Full\nEnsemble"],
            fontsize=8.5, color='#ccc'
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"[PLOT] Saved: {save_path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BƯỚC 3: TRAIN & SO SÁNH MODELS")
    print("=" * 60)

    model_defs = get_model_definitions()
    pipelines  = ["pipeline_A", "pipeline_B", "pipeline_C"]
    all_results = []
    best_data   = {}   # {pipeline: (X_train, y_train)}

    for pipeline_name in pipelines:
        print(f"\n{'═'*55}")
        print(f"  📐 Pipeline: {pipeline_name.upper()}")
        print(f"{'═'*55}")

        data = load_pipeline_data(pipeline_name)
        X_train, y_train, X_test, y_test = prepare_data(data)
        best_data[pipeline_name] = (X_train, y_train)

        print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
        print(f"  {'Model':20s} | {'Acc':>6} | {'F1':>6} | {'AUC':>6} | Train | Infer")
        print(f"  {'─'*65}")

        for model_name, (estimator, needs_scaling, desc) in model_defs.items():
            result = train_and_evaluate(
                model_name, estimator, needs_scaling,
                X_train, y_train, X_test, y_test, pipeline_name
            )
            all_results.append(result)

            # Lưu model
            model_path = MODELS_DIR / f"{pipeline_name}_{model_name.replace(' ', '_')}.pkl"
            joblib.dump(result["pipe"], model_path)

    # Tìm best model overall
    valid_results = [r for r in all_results if r["auc"] is not None]
    best_result   = max(valid_results, key=lambda r: r["auc"])
    print(f"\n{'='*60}")
    print(f"  🏆 BEST OVERALL: {best_result['model_name']} ({best_result['pipeline']})")
    print(f"     AUC={best_result['auc']:.4f} | Acc={best_result['accuracy']:.4f}")
    print(f"{'='*60}")

    # Lưu kết quả (bỏ pipeline object để serialize)
    serializable = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if k != "pipe"}
        serializable.append(rc)
    with open(RESULTS_FILE, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n[SAVE] Results: {RESULTS_FILE}")

    # Lưu best model riêng
    best_pipe_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_result["pipe"], best_pipe_path)
    best_meta = {
        "model_name" : best_result["model_name"],
        "pipeline"   : best_result["pipeline"],
        "accuracy"   : best_result["accuracy"],
        "f1"         : best_result["f1"],
        "auc"        : best_result["auc"],
    }
    with open(MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)
    print(f"[SAVE] Best model: {best_pipe_path}")

    # Plots
    print("\n[VIZ] Tạo biểu đồ so sánh...")
    plot_model_comparison(all_results)
    plot_confusion_matrices(all_results, pipeline="pipeline_C")
    plot_pipeline_ranking(all_results)

    X_train_best, y_train_best = best_data[best_result["pipeline"]]
    plot_best_model_detail(best_result, X_train_best, y_train_best)

    print("\n✅ Hoàn thành bước 3!")
    print("   → Chạy tiếp: python 04_web_app.py")


if __name__ == "__main__":
    main()