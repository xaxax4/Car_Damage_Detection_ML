import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

# ============================================================================
# CONFIG
# ============================================================================

LABELS_CSV_PATH = "dataset/labels.csv"
IMAGES_DIR = "dataset/images"
OUTPUT_DIR = "kmeans_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCATION_NAMES = [
    "Front_L", "Front_R", "Front_Middle",
    "Rear_Left", "Rear_Right", "Rear_Middle",
    "Left_Side", "Right_Side", "Roof"
]
NUM_PARTS = len(LOCATION_NAMES)

# ============================================================================
# HELPERS
# ============================================================================

def extract_id_number(image_id):
    return int(str(image_id).split("_")[0])


def load_image(path, bins):
    img = Image.open(path).convert("RGB")
    hist = img.histogram()  # 768 dims

    # compress histogram into bins
    hist = np.array(hist).reshape(3, 256)
    hist_binned = []
    bin_size = 256 // bins
    for c in range(3):
        for b in range(bins):
            start = b * bin_size
            end = (b + 1) * bin_size
            hist_binned.append(hist[c, start:end].sum())
    hist_binned = np.array(hist_binned, dtype=np.float32)

    hist_binned /= (hist_binned.sum() + 1e-8)
    return hist_binned


def load_all_features(df, bins):
    features = []
    valid_rows = []
    for idx, row in df.iterrows():
        image_id = row.iloc[0]
        path_jpg = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        path_jpeg = os.path.join(IMAGES_DIR, f"{image_id}.jpeg")

        if os.path.exists(path_jpg):
            path = path_jpg
        elif os.path.exists(path_jpeg):
            path = path_jpeg
        else:
            continue

        feat = load_image(path, bins)
        features.append(feat)
        valid_rows.append(idx)

    features = np.vstack(features)
    df_valid = df.iloc[valid_rows].reset_index(drop=True)

    return features, df_valid


def apply_pca(features, dim):
    if dim is None:
        return features

    max_dim = min(features.shape[0], features.shape[1])
    dim = min(dim, max_dim)

    if dim < 1:
        return features

    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(features)
    return reduced


# ============================================================================
# PER-PART CLUSTERING
# ============================================================================

def cluster_categories_per_part(features, true_cats_part, K_cat, seed, cat_threshold):
    km = KMeans(n_clusters=K_cat, random_state=seed, n_init='auto')
    cluster_ids = km.fit_predict(features)

    # Map each cluster to damaged/not using majority rule
    cluster_means = []
    for c in range(K_cat):
        mask = cluster_ids == c
        if mask.sum() == 0:
            cluster_means.append(0)
        else:
            cluster_means.append(true_cats_part[mask].mean())

    # cluster -> label
    cluster_to_label = np.array([1 if m >= cat_threshold else 0 for m in cluster_means])

    pred = cluster_to_label[cluster_ids]

    # Accuracy
    acc = (pred == true_cats_part).mean() * 100

    # ROC
    scores = np.array([cluster_means[c] for c in cluster_ids])
    fpr, tpr, _ = roc_curve(true_cats_part, scores)
    roc_auc = auc(fpr, tpr)

    # Calculate inertia (within-cluster sum of squares) as a proxy for "loss"
    inertia = km.inertia_

    return pred, acc, fpr, tpr, roc_auc, inertia


def cluster_severity_per_part(features, true_sev_part, K_sev, seed):
    km = KMeans(n_clusters=K_sev, random_state=seed, n_init='auto')
    cluster_ids = km.fit_predict(features)

    cluster_sev = []
    for c in range(K_sev):
        mask = cluster_ids == c
        if mask.sum() == 0:
            cluster_sev.append(0.0)
        else:
            cluster_sev.append(true_sev_part[mask].mean())

    pred = np.array([int(round(cluster_sev[c])) for c in cluster_ids])
    pred = np.clip(pred, 0, 5)

    # Only evaluate over damaged parts
    damaged_mask = true_sev_part > 0
    if damaged_mask.sum() == 0:
        return pred, np.nan, km.inertia_

    acc = (pred[damaged_mask] == true_sev_part[damaged_mask]).mean() * 100
    return pred, acc, km.inertia_


# ============================================================================
# MASTER PIPELINE
# ============================================================================

def run_kmeans_pipeline(bins, pca_dim, K_cat, K_sev, seed, df, run_dir, cat_threshold):
    os.makedirs(run_dir, exist_ok=True)

    # Load features
    features, df_valid = load_all_features(df, bins)

    # Apply PCA
    features_pca = apply_pca(features, pca_dim)

    cats = df_valid.iloc[:, 1::2].values  # shape (N, 9)
    sevs = df_valid.iloc[:, 2::2].values  # shape (N, 9)

    part_cat_acc = []
    part_cat_auc = []
    part_sev_acc = []
    part_cat_inertia = []
    part_sev_inertia = []

    fprs = []
    tprs = []

    for part in range(NUM_PARTS):
        true_c_part = cats[:, part].astype(int)
        true_s_part = sevs[:, part].astype(int)

        # Category clustering
        pred_cat, acc_cat, fpr, tpr, roc_auc, cat_inertia = cluster_categories_per_part(
            features_pca, true_c_part, K_cat, seed, cat_threshold
        )
        part_cat_acc.append(acc_cat)
        part_cat_auc.append(roc_auc)
        part_cat_inertia.append(cat_inertia)
        fprs.append(fpr)
        tprs.append(tpr)

        # Severity clustering
        pred_sev, acc_sev, sev_inertia = cluster_severity_per_part(
            features_pca, true_s_part, K_sev, seed
        )
        part_sev_acc.append(acc_sev)
        part_sev_inertia.append(sev_inertia)

    mean_cat_acc = np.nanmean(part_cat_acc)
    mean_cat_auc = np.nanmean(part_cat_auc)
    mean_sev_acc = np.nanmean(part_sev_acc)
    mean_cat_inertia = np.nanmean(part_cat_inertia)
    mean_sev_inertia = np.nanmean(part_sev_inertia)

    # Save summary
    summary = {
        "bins": bins,
        "pca_dim": pca_dim,
        "K_cat": K_cat,
        "K_sev": K_sev,
        "seed": seed,
        "cat_threshold": cat_threshold,
        "mean_cat_acc": float(mean_cat_acc),
        "mean_cat_auc": float(mean_cat_auc),
        "mean_sev_acc": float(mean_sev_acc),
        "mean_cat_inertia": float(mean_cat_inertia),
        "mean_sev_inertia": float(mean_sev_inertia),
        "per_part_cat_acc": part_cat_acc,
        "per_part_cat_auc": part_cat_auc,
        "per_part_sev_acc": part_sev_acc,
        "per_part_cat_inertia": part_cat_inertia,
        "per_part_sev_inertia": part_sev_inertia,
    }

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ========================================================================
    # PLOT 1: Category & Severity Accuracy Bar Chart
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.6

    accuracies = [mean_cat_acc, mean_sev_acc]
    labels = ['Category Accuracy', 'Severity Accuracy']
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(x, accuracies, width, color=colors)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Category & Severity Accuracy\n{os.path.basename(run_dir)}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()

    # ========================================================================
    # PLOT 2: Inertia (Loss) Bar Chart
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.6

    inertias = [mean_cat_inertia, mean_sev_inertia]
    labels = ['Category Loss', 'Severity Loss']
    colors = ['#1f77b4', '#ff7f0e']

    bars = ax.bar(x, inertias, width, color=colors)

    ax.set_ylabel('Inertia (K-Means Loss)', fontsize=12)
    ax.set_title(f'Category & Severity Inertia (Loss)\n{os.path.basename(run_dir)}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, inertias):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_comparison.png"), dpi=150)
    plt.close()

    # ========================================================================
    # PLOT 3: Per-Part Breakdown (Accuracy)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_PARTS)
    width = 0.35

    bars1 = ax.bar(x - width/2, part_cat_acc, width, label='Category Accuracy', color='#1f77b4')
    bars2 = ax.bar(x + width/2, part_sev_acc, width, label='Severity Accuracy', color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(f'Per-Part Accuracy Breakdown\n{os.path.basename(run_dir)}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATION_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "per_part_accuracy.png"), dpi=150)
    plt.close()

    # ========================================================================
    # PLOT 4: Per-Part Breakdown (Inertia/Loss)
    # ========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(NUM_PARTS)
    width = 0.35

    bars1 = ax.bar(x - width/2, part_cat_inertia, width, label='Category Loss', color='#1f77b4')
    bars2 = ax.bar(x + width/2, part_sev_inertia, width, label='Severity Loss', color='#ff7f0e')

    ax.set_ylabel('Inertia (K-Means Loss)', fontsize=12)
    ax.set_title(f'Per-Part Inertia (Loss) Breakdown\n{os.path.basename(run_dir)}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(LOCATION_NAMES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "per_part_loss.png"), dpi=150)
    plt.close()

    # Save individual ROC plots per part
    for part in range(NUM_PARTS):
        plt.figure()
        plt.plot(fprs[part], tprs[part], label=f"AUC={part_cat_auc[part]:.4f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC – {LOCATION_NAMES[part]}")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(run_dir, f"roc_part_{part}.png"))
        plt.close()

    return summary


# ============================================================================
# HYPERPARAMETER SWEEP
# ============================================================================

df = pd.read_csv(LABELS_CSV_PATH)

BINS_LIST = [16, 32, 64]
PCA_LIST = [None, 32, 64]
K_CAT_LIST = [2, 3]
K_SEV_LIST = [6, 7, 10]
SEEDS = [44444, 33]
CAT_THRESHOLD_LIST = [0.35, 0.42, 0.5]

global_summary = []

print("\nStarting K-Means sweep...\n")

for bins in BINS_LIST:
    for pca_dim in PCA_LIST:
        for K_cat in K_CAT_LIST:
            for K_sev in K_SEV_LIST:
                for cat_threshold in CAT_THRESHOLD_LIST:
                    for seed in SEEDS:

                        name = (
                            f"bins{bins}_"
                            f"pca{pca_dim}_"
                            f"kcat{K_cat}_"
                            f"ksev{K_sev}_"
                            f"ct{str(cat_threshold).replace('.', 'p')}_"
                            f"s{seed}"
                        )
                        name = name.replace("None", "none")

                        run_dir = os.path.join(OUTPUT_DIR, name)
                        print(f"Running {name}")

                        summary = run_kmeans_pipeline(
                            bins, pca_dim, K_cat, K_sev, seed, df, run_dir,
                            cat_threshold=cat_threshold
                        )

                        global_summary.append({
                            "run": name,
                            **summary
                        })


# ============================================================================
# GLOBAL SUMMARY: TOP PARAMETER COMBINATIONS
# ============================================================================

df_global = pd.DataFrame(global_summary)
global_path = os.path.join(OUTPUT_DIR, "global_summary.csv")
df_global.to_csv(global_path, index=False)

print(f"\nSaved global summary to {global_path}")

# Compute ranking metrics
df_global["mean_cat_auc"] = df_global["mean_cat_auc"].astype(float)
df_global["mean_cat_acc"] = df_global["mean_cat_acc"].astype(float)
df_global["mean_sev_acc"] = df_global["mean_sev_acc"].astype(float)

# ---- Top 4 by AUC ----
df_top_auc = df_global.sort_values("mean_cat_auc", ascending=False).head(4)
df_top_auc.to_csv(os.path.join(OUTPUT_DIR, "top4_by_mean_auc.csv"), index=False)

# ---- Top 4 by Category Accuracy ----
df_top_cat = df_global.sort_values("mean_cat_acc", ascending=False).head(4)
df_top_cat.to_csv(os.path.join(OUTPUT_DIR, "top4_by_mean_cat.csv"), index=False)

# ---- Top 4 by Severity Accuracy ----
df_top_sev = df_global.sort_values("mean_sev_acc", ascending=False).head(4)
df_top_sev.to_csv(os.path.join(OUTPUT_DIR, "top4_by_mean_sev.csv"), index=False)

# ---- Combined report ----
report_lines = []

def add_section(title, df_top, metric_name):
    report_lines.append(title)
    for _, row in df_top.iterrows():
        report_lines.append(
            f"- {row['run']}: "
            f"{metric_name}={row[metric_name]:.4f}, "
            f"mean_cat_acc={row['mean_cat_acc']:.2f}%, "
            f"mean_sev_acc={row['mean_sev_acc']:.2f}%"
        )
    report_lines.append("")

add_section("Best runs by mean Category AUC:", df_top_auc, "mean_cat_auc")
add_section("Best runs by mean Category Accuracy:", df_top_cat, "mean_cat_acc")
add_section("Best runs by mean Severity Accuracy:", df_top_sev, "mean_sev_acc")

report_text = "\n".join(report_lines)
report_path = os.path.join(OUTPUT_DIR, "combined_top4_report.txt")

with open(report_path, "w") as f:
    f.write(report_text)

print("\n========== K-MEANS TOP RESULTS ==========")
print(report_text)
print("Global report saved to:", report_path)
print("==========================================\n")
