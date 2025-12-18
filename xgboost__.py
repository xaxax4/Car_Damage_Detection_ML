import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage.color import rgb2gray
import xgboost as xgb
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA

# ============================================================
# CONFIG
# ============================================================

LABELS_CSV_PATH = "dataset/labels.csv"
IMAGES_DIR = "dataset/images"

OUTPUT_ROOT = "xgb_results"
BASELINE_DIR = os.path.join(OUTPUT_ROOT, "baseline")
SWEEP_DIR = os.path.join(OUTPUT_ROOT, "sweep")
GLOBAL_SUMMARY_DIR = os.path.join(OUTPUT_ROOT, "GLOBAL_SUMMARY")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(SWEEP_DIR, exist_ok=True)
os.makedirs(GLOBAL_SUMMARY_DIR, exist_ok=True)

# Car-level split
NUM_TEST_CAR_IDS = 10
SPLIT_SEED = 44444

# Baseline XGBoost hyperparameters
BASELINE_PCA_DIM = 32
BASELINE_N_ESTIMATORS = 200
BASELINE_MAX_DEPTH = 1
BASELINE_LEARNING_RATE = 0.1
BASELINE_SUBSAMPLE = 0.8
BASELINE_COLSAMPLE_BYTREE = 0.8
BASELINE_THRESHOLD = 0.5
BASELINE_XGB_SEED = 44444

# Location names
LOCATION_NAMES = [
    "Front_L", "Front_R", "Front_Middle",
    "Rear_Left", "Rear_Right", "Rear_Middle",
    "Left_Side", "Right_Side", "Roof"
]
NUM_PARTS = len(LOCATION_NAMES)


# ============================================================
# BASIC HELPERS
# ============================================================

def extract_id_number(image_id):
    return int(str(image_id).split("_")[0])


def load_image_hog(path, pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=9):
    img = Image.open(path).convert("RGB")
    img = img.resize((256, 256))
    img = np.array(img)
    gray = rgb2gray(img)

    hog_features = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True
    )

    return hog_features.astype(np.float32)


def load_features_for_df(df):
    features = []
    valid_idx = []

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

        feat = load_image_hog(path)
        features.append(feat)
        valid_idx.append(idx)

    X = np.vstack(features)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)
    return X, df_valid


def apply_pca_train_val(X_train, X_val, dim):
    if dim is None:
        return X_train, X_val

    max_dim = min(X_train.shape[0], X_train.shape[1])
    dim = min(dim, max_dim)
    if dim < 1:
        return X_train, X_val

    pca = PCA(n_components=dim)
    X_train_reduced = pca.fit_transform(X_train)
    X_val_reduced = pca.transform(X_val)
    return X_train_reduced, X_val_reduced


# ============================================================
# DATA SPLIT BY CAR ID
# ============================================================

df_full = pd.read_csv(LABELS_CSV_PATH)
df_full["id_number"] = df_full.iloc[:, 0].apply(extract_id_number)

unique_ids = sorted(df_full["id_number"].unique())


def has_damage_for_car(df, car_id):
    car_samples = df[df["id_number"] == car_id]
    damage_cols = car_samples.iloc[:, 1::2]
    return (damage_cols == 1).any().any()


damaged_ids = [cid for cid in unique_ids if has_damage_for_car(df_full, cid)]
undamaged_ids = [cid for cid in unique_ids if not has_damage_for_car(df_full, cid)]

np.random.seed(SPLIT_SEED)

num_damaged_test = max(1, NUM_TEST_CAR_IDS // 2)
num_undamaged_test = NUM_TEST_CAR_IDS - num_damaged_test

if len(damaged_ids) >= num_damaged_test:
    test_ids_damaged = np.random.choice(damaged_ids, size=num_damaged_test, replace=False)
else:
    test_ids_damaged = np.array(damaged_ids)

if len(undamaged_ids) >= num_undamaged_test:
    test_ids_undamaged = np.random.choice(undamaged_ids, size=num_undamaged_test, replace=False)
else:
    test_ids_undamaged = np.array(undamaged_ids)

test_ids = sorted(list(test_ids_damaged) + list(test_ids_undamaged))

train_df = df_full[~df_full["id_number"].isin(test_ids)].drop(columns=["id_number"]).reset_index(drop=True)
val_df = df_full[df_full["id_number"].isin(test_ids)].drop(columns=["id_number"]).reset_index(drop=True)

print("="*80)
print("XGBOOST - DATA SPLIT")
print("="*80)
print(f"Total unique car IDs: {len(unique_ids)}")
print(f"  - Damaged cars: {len(damaged_ids)}")
print(f"  - Undamaged cars: {len(undamaged_ids)}")
print(f"\nTrain cars: {len(unique_ids) - len(test_ids)}")
print(f"Val cars: {len(test_ids)}")
print(f"  - Damaged val cars: {len(test_ids_damaged)}")
print(f"  - Undamaged val cars: {len(test_ids_undamaged)}")
print(f"\nSelected val car IDs: {test_ids}")
print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
print("="*80)
print()


# ============================================================
# XGBOOST PIPELINE WITH FIXED SEVERITY TRAINING
# ============================================================

def run_xgb_pipeline(
        pca_dim,
        n_estimators,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        threshold,
        seed,
        train_df,
        val_df,
        run_dir=None,
        make_plots=False
):

    if run_dir is not None:
        os.makedirs(run_dir, exist_ok=True)

    # ---- 1. Features for train & val ----
    X_train, train_valid = load_features_for_df(train_df)
    X_val, val_valid = load_features_for_df(val_df)

    # Align labels
    y_train_cats = train_valid.iloc[:, 1::2].values.astype(int)
    y_train_sevs = train_valid.iloc[:, 2::2].values.astype(int)

    y_val_cats = val_valid.iloc[:, 1::2].values.astype(int)
    y_val_sevs = val_valid.iloc[:, 2::2].values.astype(int)

    print(f"[XGB] PCA={pca_dim}, n_estimators={n_estimators}, "
          f"max_depth={max_depth}, learning_rate={learning_rate}, "
          f"subsample={subsample}, colsample_bytree={colsample_bytree}, "
          f"threshold={threshold}, seed={seed}")
    print(f"  Train images: {X_train.shape[0]}, Val images: {X_val.shape[0]}")

    # ---- 2. PCA on train, apply to val ----
    X_train_pca, X_val_pca = apply_pca_train_val(X_train, X_val, pca_dim)
    print(f"  Feature dims after PCA: {X_train_pca.shape[1]}")

    # ---- 3. Train XGBoost for categories (per-part binary classifiers) ----
    print(f"\n  Training per-part category classifiers:")
    xgb_cats = []

    for p in range(NUM_PARTS):
        xgb_cat = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed,
            tree_method='hist',
            eval_metric='logloss'
        )
        xgb_cat.fit(X_train_pca, y_train_cats[:, p])
        xgb_cats.append(xgb_cat)
        print(f"    {LOCATION_NAMES[p]:<15} Trained")

    # ---- 4. Train SEPARATE XGBoost for severity of EACH part (only on damaged samples) ----
    print(f"\n  Training per-part severity classifiers:")
    xgb_sevs = []

    for p in range(NUM_PARTS):
        # Only train on samples where this part is damaged
        part_damaged_mask = y_train_cats[:, p] == 1
        n_damaged = part_damaged_mask.sum()

        if n_damaged < 2:  # Need at least 2 samples
            print(f"    {LOCATION_NAMES[p]:<15} Skipped (only {n_damaged} damaged sample{'s' if n_damaged != 1 else ''})")
            xgb_sevs.append(None)
            continue

        X_train_part = X_train_pca[part_damaged_mask]
        y_train_sev_part = y_train_sevs[part_damaged_mask, p]

        # Filter out severity=0 (undamaged) samples for severity training
        sev_mask = y_train_sev_part > 0
        if sev_mask.sum() < 2:
            print(f"    {LOCATION_NAMES[p]:<15} Skipped (only {sev_mask.sum()} sample(s) with severity > 0)")
            xgb_sevs.append(None)
            continue

        X_train_part_damaged = X_train_part[sev_mask]
        y_train_sev_part_damaged = y_train_sev_part[sev_mask]

        # Check if we have multiple severity levels
        unique_sevs = np.unique(y_train_sev_part_damaged)
        if len(unique_sevs) < 2:
            print(f"    {LOCATION_NAMES[p]:<15} Skipped (only severity level {unique_sevs[0]})")
            xgb_sevs.append(None)
            continue

        unique_sevs_sorted = np.sort(unique_sevs)
        sev_mapping = {old: new for new, old in enumerate(unique_sevs_sorted)}
        y_train_sev_remapped = np.array([sev_mapping[s] for s in y_train_sev_part_damaged])

        # Train classifier for this part
        xgb_sev_part = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=seed + p + 1,
            tree_method='hist',
            eval_metric='logloss'
        )
        xgb_sev_part.fit(X_train_part_damaged, y_train_sev_remapped)

        # Store both the model and the reverse mapping (for prediction)
        reverse_mapping = {new: old for old, new in sev_mapping.items()}
        xgb_sevs.append((xgb_sev_part, reverse_mapping))

        print(f"    {LOCATION_NAMES[p]:<15} Trained on {sev_mask.sum()} samples with severity > 0, levels: {sorted(unique_sevs)}")

    # ====================================================
    # EVALUATION - CATEGORIES
    # ====================================================

    y_pred_cats = np.zeros_like(y_val_cats)
    proba_cats = np.zeros((y_val_cats.shape[0], NUM_PARTS))

    for p in range(NUM_PARTS):
        proba_cats[:, p] = xgb_cats[p].predict_proba(X_val_pca)[:, 1]
        # Apply threshold
        y_pred_cats[:, p] = (proba_cats[:, p] >= threshold).astype(int)

    per_part_cat_acc = []
    per_part_cat_auc = []
    per_part_cat_tp = []
    per_part_cat_fp = []
    per_part_cat_tn = []
    per_part_cat_fn = []
    fprs = []
    tprs = []

    for p in range(NUM_PARTS):
        y_true = y_val_cats[:, p]
        y_pred = y_pred_cats[:, p]
        proba_pos = proba_cats[:, p]

        # Accuracy
        acc_p = (y_true == y_pred).mean() * 100

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # ROC
        if len(np.unique(y_true)) < 2:
            fpr_p, tpr_p, auc_p = [0], [0], np.nan
        else:
            fpr_p, tpr_p, _ = roc_curve(y_true, proba_pos)
            auc_p = auc(fpr_p, tpr_p)

        per_part_cat_acc.append(acc_p)
        per_part_cat_auc.append(auc_p)
        per_part_cat_tp.append(int(tp))
        per_part_cat_fp.append(int(fp))
        per_part_cat_tn.append(int(tn))
        per_part_cat_fn.append(int(fn))
        fprs.append(fpr_p)
        tprs.append(tpr_p)

    mean_cat_acc = float(np.mean(per_part_cat_acc))
    mean_cat_auc = float(np.nanmean(per_part_cat_auc))

    # Global ROC across all parts
    all_true = np.concatenate([y_val_cats[:, p] for p in range(NUM_PARTS)])
    all_scores = np.concatenate([proba_cats[:, p] for p in range(NUM_PARTS)])
    fpr_all, tpr_all, _ = roc_curve(all_true, all_scores)
    auc_all = auc(fpr_all, tpr_all)

    # ====================================================
    # EVALUATION - SEVERITY (using per-part classifiers)
    # ====================================================

    y_pred_sevs = np.zeros_like(y_val_sevs)

    # Predict severity for each part separately and remap back to original labels
    for p in range(NUM_PARTS):
        if xgb_sevs[p] is not None:
            model, reverse_mapping = xgb_sevs[p]
            y_pred_remapped = model.predict(X_val_pca)
            y_pred_sevs[:, p] = np.array([reverse_mapping[pred] for pred in y_pred_remapped])

    per_part_sev_acc = []
    per_part_sev_counts = []
    total_correct_damaged = 0
    total_damaged = 0

    for p in range(NUM_PARTS):
        y_true = y_val_sevs[:, p]
        y_pred = y_pred_sevs[:, p]
        # Only evaluate on actually damaged parts
        mask = y_true > 0

        if mask.sum() == 0:
            per_part_sev_acc.append(np.nan)
            per_part_sev_counts.append(0)
            continue

        correct = (y_true[mask] == y_pred[mask]).sum()
        total = mask.sum()
        acc_p = correct / total * 100

        total_correct_damaged += correct
        total_damaged += total

        per_part_sev_acc.append(acc_p)
        per_part_sev_counts.append(int(total))

    if total_damaged > 0:
        overall_sev_acc = total_correct_damaged / total_damaged * 100
    else:
        overall_sev_acc = np.nan

    mean_sev_acc = float(np.nanmean(per_part_sev_acc))

    # ====================================================
    # PRINT DETAILED RESULTS
    # ====================================================

    print(f"\n{'='*80}")
    print(f"CATEGORY PREDICTION PERFORMANCE")
    print(f"{'='*80}")
    print(f"Overall Category Accuracy: {mean_cat_acc:.2f}%")
    print(f"Mean Category AUC: {mean_cat_auc:.4f}")
    print(f"Global Category AUC: {auc_all:.4f}")
    print()

    for i, name in enumerate(LOCATION_NAMES):
        print(f"{name:<20} Acc: {per_part_cat_acc[i]:5.2f}% | AUC: {per_part_cat_auc[i]:.4f} | "
              f"TP:{per_part_cat_tp[i]:3d} FP:{per_part_cat_fp[i]:3d} "
              f"TN:{per_part_cat_tn[i]:3d} FN:{per_part_cat_fn[i]:3d}")

    print(f"\n{'-'*80}")
    print(f"SEVERITY PREDICTION PERFORMANCE (damaged parts only)")
    print(f"{'-'*80}")
    print(f"Overall Severity Accuracy: {overall_sev_acc:.2f}% ({total_correct_damaged}/{total_damaged})")
    print(f"Mean Severity Accuracy: {mean_sev_acc:.2f}%")
    print()

    for i, name in enumerate(LOCATION_NAMES):
        if np.isnan(per_part_sev_acc[i]):
            status = "N/A (no damaged parts)" if xgb_sevs[i] is not None else "N/A (no trained model)"
            print(f"{name:<20} Severity Acc: {status}")
        else:
            print(f"{name:<20} Severity Acc: {per_part_sev_acc[i]:5.2f}% ({per_part_sev_counts[i]} damaged)")

    # ====================================================
    # SAVE SUMMARY
    # ====================================================

    summary = {
        "pca_dim": pca_dim,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "threshold": threshold,
        "seed": seed,
        "mean_cat_acc": float(mean_cat_acc),
        "mean_cat_auc": float(mean_cat_auc),
        "mean_sev_acc": float(mean_sev_acc),
        "overall_sev_acc": float(overall_sev_acc),
        "auc_all": float(auc_all),
        "per_part_cat_acc": per_part_cat_acc,
        "per_part_cat_auc": per_part_cat_auc,
        "per_part_sev_acc": per_part_sev_acc,
        "per_part_sev_counts": per_part_sev_counts,
    }

    if run_dir is not None:
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Save predictions CSV
        predictions_df = pd.DataFrame({
            'Image_ID': val_valid.iloc[:, 0].values
        })

        for i, name in enumerate(LOCATION_NAMES):
            predictions_df[f'True_Cat_{name}'] = y_val_cats[:, i]
            predictions_df[f'Pred_Cat_{name}'] = y_pred_cats[:, i]
            predictions_df[f'Pred_Prob_{name}'] = proba_cats[:, i]
            predictions_df[f'True_Sev_{name}'] = y_val_sevs[:, i]
            predictions_df[f'Pred_Sev_{name}'] = y_pred_sevs[:, i]

        predictions_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

        if make_plots:
            # ========================================================
            # PLOT 1: Accuracy Comparison Bar Chart
            # ========================================================
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(2)
            width = 0.6

            accuracies = [mean_cat_acc, overall_sev_acc if not np.isnan(overall_sev_acc) else 0]
            labels = ['Category Accuracy', 'Severity Accuracy']
            colors = ['#1f77b4', '#ff7f0e']

            bars = ax.bar(x, accuracies, width, color=colors)

            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title(f'Category & Severity Accuracy\n{os.path.basename(run_dir)}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}%',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "accuracy_comparison.png"), dpi=150)
            plt.close()

            # ========================================================
            # PLOT 2: Global ROC Curve
            # ========================================================
            plt.figure(figsize=(8, 8))
            plt.plot(fpr_all, tpr_all, label=f"XGBoost Category (AUC = {auc_all:.4f})", linewidth=2)
            plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("XGBoost – Global Category ROC", fontsize=14)
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_global.png"), dpi=150)
            plt.close()

            # ========================================================
            # PLOT 3: Per-Part Accuracy Breakdown
            # ========================================================
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(NUM_PARTS)
            width = 0.35

            # Replace NaN with 0 for plotting
            plot_sev_acc = [0 if np.isnan(v) else v for v in per_part_sev_acc]

            bars1 = ax.bar(x - width/2, per_part_cat_acc, width, label='Category Accuracy', color='#1f77b4')
            bars2 = ax.bar(x + width/2, plot_sev_acc, width, label='Severity Accuracy', color='#ff7f0e')

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

            # ========================================================
            # PLOT 4: Per-Part AUC Breakdown
            # ========================================================
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(NUM_PARTS)

            bars = ax.bar(x, per_part_cat_auc, color='#2ca02c', alpha=0.8)

            ax.set_ylabel('AUC Score', fontsize=12)
            ax.set_title(f'Per-Part Category AUC\n{os.path.basename(run_dir)}', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(LOCATION_NAMES, rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random Classifier')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "per_part_auc.png"), dpi=150)
            plt.close()

            # ========================================================
            # PLOT 5: Per-Part ROC Curves (3x3 grid)
            # ========================================================
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()

            for p in range(NUM_PARTS):
                axes[p].plot(fprs[p], tprs[p], label=f"AUC={per_part_cat_auc[p]:.4f}", linewidth=2)
                axes[p].plot([0, 1], [0, 1], "k--", linewidth=1)
                axes[p].set_title(f"{LOCATION_NAMES[p]}", fontsize=11)
                axes[p].set_xlabel("FPR", fontsize=9)
                axes[p].set_ylabel("TPR", fontsize=9)
                axes[p].legend(fontsize=9)
                axes[p].grid(alpha=0.3)

            plt.suptitle("Per-Part ROC Curves", fontsize=16, y=0.995)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "roc_per_part_grid.png"), dpi=150)
            plt.close()

    return summary


# ============================================================
# BASELINE RUN
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST BASELINE RUN")
print("=" * 80)

baseline_summary = run_xgb_pipeline(
    pca_dim=BASELINE_PCA_DIM,
    n_estimators=BASELINE_N_ESTIMATORS,
    max_depth=BASELINE_MAX_DEPTH,
    learning_rate=BASELINE_LEARNING_RATE,
    subsample=BASELINE_SUBSAMPLE,
    colsample_bytree=BASELINE_COLSAMPLE_BYTREE,
    threshold=BASELINE_THRESHOLD,
    seed=BASELINE_XGB_SEED,
    train_df=train_df,
    val_df=val_df,
    run_dir=BASELINE_DIR,
    make_plots=True
)

print("\n✓ Baseline run complete. Results saved to:", BASELINE_DIR)


# ============================================================
# HYPERPARAMETER SWEEP
# ============================================================

print("\n" + "=" * 80)
print("XGBOOST HYPERPARAMETER SWEEP")
print("=" * 80)

PCA_LIST = [None, 32, 64]
N_ESTIMATORS_LIST = [100, 200]
MAX_DEPTH_LIST = [1, 3, 6]
LEARNING_RATE_LIST = [0.01, 0.1, 0.3]
SUBSAMPLE_LIST = [0.8]
COLSAMPLE_BYTREE_LIST = [0.8]
THRESHOLD_LIST = [0.35, 0.42, 0.5]
XGB_SEEDS = [44444]

sweep_records = []

total_runs = ( len(PCA_LIST) * len(N_ESTIMATORS_LIST) * len(MAX_DEPTH_LIST) * len(LEARNING_RATE_LIST) * len(SUBSAMPLE_LIST) * len(COLSAMPLE_BYTREE_LIST) * len(THRESHOLD_LIST) * len(XGB_SEEDS))

print(f"Total sweep runs: {total_runs}\n")

run_counter = 0

for pca_dim in PCA_LIST:
    for n_estimators in N_ESTIMATORS_LIST:
        for max_depth in MAX_DEPTH_LIST:
            for learning_rate in LEARNING_RATE_LIST:
                for subsample in SUBSAMPLE_LIST:
                    for colsample_bytree in COLSAMPLE_BYTREE_LIST:
                        for threshold in THRESHOLD_LIST:
                            for seed in XGB_SEEDS:
                                run_counter += 1

                                run_name = (
                                    f"pca{pca_dim}_"
                                    f"trees{n_estimators}_"
                                    f"depth{max_depth}_"
                                    f"lr{learning_rate}_"
                                    f"sub{subsample}_"
                                    f"col{colsample_bytree}_"
                                    f"thr{threshold}_"
                                    f"s{seed}"
                                )
                                run_name = run_name.replace("None", "none")

                                run_dir = os.path.join(SWEEP_DIR, run_name)
                                print(f"\n[{run_counter}/{total_runs}] Running: {run_name}")

                                summary = run_xgb_pipeline(
                                    pca_dim=pca_dim,
                                    n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    learning_rate=learning_rate,
                                    subsample=subsample,
                                    colsample_bytree=colsample_bytree,
                                    threshold=threshold,
                                    seed=seed,
                                    train_df=train_df,
                                    val_df=val_df,
                                    run_dir=run_dir,
                                    make_plots=True
                                )

                                sweep_records.append({
                                    "run_name": run_name,
                                    **{k: v for k, v in summary.items()
                                       if not isinstance(v, (list, np.ndarray))}
                                })

# ============================================================
# GLOBAL SUMMARY + TOP-4
# ============================================================

if sweep_records:
    df_runs = pd.DataFrame(sweep_records)
    summary_path = os.path.join(GLOBAL_SUMMARY_DIR, "all_runs_summary.csv")
    df_runs.to_csv(summary_path, index=False)

    print("\n" + "="*80)
    print(f"✓ Saved XGBoost sweep global summary to: {summary_path}")
    print("="*80)

    # Sort by different metrics
    top4_auc = df_runs.sort_values("mean_cat_auc", ascending=False).head(4)
    top4_cat = df_runs.sort_values("mean_cat_acc", ascending=False).head(4)
    top4_sev = df_runs.sort_values("mean_sev_acc", ascending=False).head(4)

    top4_auc.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_cat_auc.csv"), index=False)
    top4_cat.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_cat_acc.csv"), index=False)
    top4_sev.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_sev_acc.csv"), index=False)

    report_lines = []

    def add_section(title, df_top, metric_col):
        report_lines.append(title)
        for _, row in df_top.iterrows():
            report_lines.append(
                f"  • {row['run_name']}"
            )
            report_lines.append(
                f"    {metric_col}={row[metric_col]:.4f}, "
                f"cat_acc={row['mean_cat_acc']:.2f}%, "
                f"sev_acc={row['mean_sev_acc']:.2f}%, "
                f"global_auc={row['auc_all']:.4f}"
            )
        report_lines.append("")

    add_section("=== Best runs by mean Category AUC ===", top4_auc, "mean_cat_auc")
    add_section("=== Best runs by mean Category Accuracy ===", top4_cat, "mean_cat_acc")
    add_section("=== Best runs by mean Severity Accuracy ===", top4_sev, "mean_sev_acc")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(GLOBAL_SUMMARY_DIR, "combined_top4_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "="*80)
    print("✓ Top-4 summary written to:", report_path)
    print("="*80)
    print(report_text)
    print("="*80)
else:
    print("\n⚠ No sweep records generated (empty sweep).")