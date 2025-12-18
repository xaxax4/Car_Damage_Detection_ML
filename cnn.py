import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from torchvision.models import resnet18, ResNet18_Weights

# ============================================================
# CONFIG / HYPERPARAMETERS (BASELINE)
# ============================================================

# Data paths
LABELS_CSV_PATH = 'dataset/labels.csv'
IMAGES_DIR = 'dataset/images'
OUTPUT_DIR = "cnnresults"
BASELINE_DIR = os.path.join(OUTPUT_DIR, "baseline")
SWEEP_ROOT_DIR = os.path.join(OUTPUT_DIR, "sweep")
GLOBAL_SUMMARY_DIR = os.path.join(SWEEP_ROOT_DIR, "GLOBAL_SUMMARY")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BASELINE_DIR, exist_ok=True)
os.makedirs(SWEEP_ROOT_DIR, exist_ok=True)
os.makedirs(GLOBAL_SUMMARY_DIR, exist_ok=True)

# ---- BASELINE HYPERPARAMETERS (these are used for A+B+C+D run) ----
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.0003

CATEGORY_WEIGHT = 1.0
SEVERITY_WEIGHT = 3.0

CATEGORY_THRESHOLD = 0.45
SEVERITY_BOOST = 1.3

RANDOM_SEED = 44444
NUM_TEST_CAR_IDS = 10

# Baseline output paths
LOSS_PLOT_PATH_BASELINE = os.path.join(BASELINE_DIR, "hierarchical_loss_plot_all_models.png")
ROC_PLOT_PATH_BASELINE = os.path.join(BASELINE_DIR, "comparative_roc_curves.png")
ACC_BAR_PLOT_PATH_BASELINE = os.path.join(BASELINE_DIR, "comparative_accuracy_bar_chart.png")
PREDICTIONS_CSV_PREFIX_BASELINE = os.path.join(BASELINE_DIR, "hierarchical_predictions_")

# Location names
LOCATION_NAMES = [
    'Front_L', 'Front_R', 'Front_Middle', 'Rear_Left',
    'Rear_Right', 'Rear_Middle', 'Left_Side', 'Right_Side', 'Roof'
]

NUM_PARTS = len(LOCATION_NAMES)
NUM_SEVERITY_CLASSES = 6
NUM_ORDINAL_THRESHOLDS = 5

# Seed for baseline
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# DATASET
# ============================================================

class CarDamageDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row.iloc[0]

        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f"{image_id}.jpeg")

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        all_labels = row.iloc[1:].values.astype('float32')
        categories = all_labels[::2]
        severities = all_labels[1::2]

        return image, torch.tensor(categories), torch.tensor(severities), image_id

# ============================================================
# BACKBONE + MODEL VARIANTS (SMALL CNN)
# ============================================================

class ConvBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(32 * 128 * 128, 128)
        self.shared_relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.shared_fc(x)
        features = self.shared_relu(x)
        return features


class BaseHierarchicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvBackbone()
        self.category_head = nn.Linear(128, NUM_PARTS)

    def forward_backbone_and_category(self, x):
        features = self.backbone(x)
        category_logits = self.category_head(features)
        return features, category_logits

    def forward(self, x, apply_hierarchy=True, threshold=CATEGORY_THRESHOLD):
        raise NotImplementedError("Use concrete subclasses.")


class RegressionSeverityModel(BaseHierarchicalModel):
    def __init__(self):
        super().__init__()
        self.severity_head = nn.Linear(128, NUM_PARTS)

    def forward(self, x, apply_hierarchy=True, threshold=CATEGORY_THRESHOLD):
        features, category_logits = self.forward_backbone_and_category(x)
        severity_raw = self.severity_head(features)

        if not apply_hierarchy:
            return category_logits, severity_raw

        category_probs = torch.sigmoid(category_logits)
        is_damaged = (category_probs > threshold).float()
        severity_masked = severity_raw * category_probs
        severity_boosted = torch.where(
            is_damaged > 0,
            severity_masked * SEVERITY_BOOST,
            severity_masked
        )
        return category_logits, severity_boosted


class ClassificationSeverityModel(BaseHierarchicalModel):
    def __init__(self):
        super().__init__()
        self.severity_head = nn.Linear(128, NUM_PARTS * NUM_SEVERITY_CLASSES)

    def forward(self, x, apply_hierarchy=True, threshold=CATEGORY_THRESHOLD):
        features, category_logits = self.forward_backbone_and_category(x)
        logits = self.severity_head(features)
        severity_logits = logits.view(-1, NUM_PARTS, NUM_SEVERITY_CLASSES)

        if not apply_hierarchy:
            return category_logits, severity_logits

        severity_classes = torch.argmax(severity_logits, dim=-1).float()
        category_probs = torch.sigmoid(category_logits)
        is_damaged = (category_probs > threshold).float()
        severity_masked = severity_classes * category_probs
        severity_boosted = torch.where(
            is_damaged > 0,
            severity_masked * SEVERITY_BOOST,
            severity_masked
        )
        return category_logits, severity_boosted


class OrdinalSeverityModel(BaseHierarchicalModel):
    def __init__(self):
        super().__init__()
        self.severity_head = nn.Linear(128, NUM_PARTS * NUM_ORDINAL_THRESHOLDS)

    def forward(self, x, apply_hierarchy=True, threshold=CATEGORY_THRESHOLD):
        features, category_logits = self.forward_backbone_and_category(x)
        logits = self.severity_head(features)
        severity_logits = logits.view(-1, NUM_PARTS, NUM_ORDINAL_THRESHOLDS)

        if not apply_hierarchy:
            return category_logits, severity_logits

        probs = torch.sigmoid(severity_logits)
        passed = (probs > 0.5).float()
        severity_numeric = passed.sum(dim=-1)

        category_probs = torch.sigmoid(category_logits)
        is_damaged = (category_probs > threshold).float()
        severity_masked = severity_numeric * category_probs
        severity_boosted = torch.where(
            is_damaged > 0,
            severity_masked * SEVERITY_BOOST,
            severity_masked
        )
        return category_logits, severity_boosted

# ============================================================
# PRETRAINED RESNET18 BACKBONE + MODEL D (RESNET REGRESSION)
# ============================================================

class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Identity()
        self.output_dim = 512

    def forward(self, x):
        return self.model(x)


class ResNetHierarchicalRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNetBackbone()
        self.category_head = nn.Linear(self.backbone.output_dim, NUM_PARTS)
        self.severity_head = nn.Linear(self.backbone.output_dim, NUM_PARTS)

    def forward(self, x, apply_hierarchy=True, threshold=CATEGORY_THRESHOLD):
        feats = self.backbone(x)
        category_logits = self.category_head(feats)
        severity_raw = self.severity_head(feats)

        if not apply_hierarchy:
            return category_logits, severity_raw

        category_probs = torch.sigmoid(category_logits)
        is_damaged = (category_probs > threshold).float()
        severity_masked = severity_raw * category_probs
        severity_boosted = torch.where(
            is_damaged > 0,
            severity_masked * SEVERITY_BOOST,
            severity_masked
        )
        return category_logits, severity_boosted

# ============================================================
# LOSS FUNCTIONS
# ============================================================

class RegressionLoss(nn.Module):
    def __init__(self, category_weight=1.0, severity_weight=3.0):
        super().__init__()
        self.category_weight = category_weight
        self.severity_weight = severity_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, category_logits, severity_preds, category_targets, severity_targets):
        category_loss = self.bce(category_logits, category_targets)
        severity_targets = severity_targets.float()
        severity_mse = self.mse(severity_preds, severity_targets)
        weighted_severity = severity_mse * category_targets
        severity_loss = weighted_severity.mean()
        total_loss = self.category_weight * category_loss + self.severity_weight * severity_loss
        return total_loss, category_loss.item(), severity_loss.item()


class ClassificationLoss(nn.Module):
    def __init__(self, category_weight=1.0, severity_weight=3.0):
        super().__init__()
        self.category_weight = category_weight
        self.severity_weight = severity_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, category_logits, severity_logits, category_targets, severity_targets):
        category_loss = self.bce(category_logits, category_targets)

        severity_targets_long = severity_targets.long()
        logits_flat = severity_logits.view(-1, NUM_SEVERITY_CLASSES)
        targets_flat = severity_targets_long.view(-1)
        cat_flat = category_targets.view(-1)

        ce_loss = self.ce(logits_flat, targets_flat)
        weighted = ce_loss * cat_flat
        denom = torch.clamp(cat_flat.sum(), min=1.0)
        severity_loss = weighted.sum() / denom

        total_loss = self.category_weight * category_loss + self.severity_weight * severity_loss
        return total_loss, category_loss.item(), severity_loss.item()


class OrdinalLoss(nn.Module):
    def __init__(self, category_weight=1.0, severity_weight=3.0):
        super().__init__()
        self.category_weight = category_weight
        self.severity_weight = severity_weight
        self.bce = nn.BCEWithLoglogitsLoss if False else nn.BCEWithLogitsLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_ord = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, category_logits, severity_logits, category_targets, severity_targets):
        category_loss = self.bce(category_logits, category_targets)

        B, P, T = severity_logits.shape
        severity_targets_long = severity_targets.long()

        ordinal_targets = torch.zeros_like(severity_logits)
        for t in range(T):
            ordinal_targets[:, :, t] = (severity_targets_long > t).float()

        ord_loss = self.bce_ord(severity_logits, ordinal_targets)

        cat_expanded = category_targets.unsqueeze(-1).expand_as(ord_loss)
        weighted = ord_loss * cat_expanded
        denom = torch.clamp(cat_expanded.sum(), min=1.0)
        severity_loss = weighted.sum() / denom

        total_loss = self.category_weight * category_loss + self.severity_weight * severity_loss
        return total_loss, category_loss.item(), severity_loss.item()

# ============================================================
# DATA LOADING & SPLIT HELPERS
# ============================================================

df = pd.read_csv(LABELS_CSV_PATH)

def extract_id_number(image_id):
    return int(str(image_id).split('_')[0])

df['id_number'] = df.iloc[:, 0].apply(extract_id_number)

def has_damage_for_car(df_local, car_id):
    car_samples = df_local[df_local['id_number'] == car_id]
    damage_cols = car_samples.iloc[:, 1::2]
    return (damage_cols == 1).any().any()

def prepare_data(batch_size, random_seed, num_test_car_ids, verbose=True):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    unique_ids = sorted(df['id_number'].unique())
    if verbose:
        print(f"Total unique car IDs: {len(unique_ids)}")
        print(f"Car IDs range: {min(unique_ids)} to {max(unique_ids)}")
        print()

    damaged_ids = [cid for cid in unique_ids if has_damage_for_car(df, cid)]
    undamaged_ids = [cid for cid in unique_ids if not has_damage_for_car(df, cid)]

    if verbose:
        print(f"Cars with damage: {len(damaged_ids)}")
        print(f"Cars without damage: {len(undamaged_ids)}")
        print()

    num_damaged_test = max(1, num_test_car_ids // 2)
    num_undamaged_test = num_test_car_ids - num_damaged_test

    if len(damaged_ids) >= num_damaged_test:
        test_ids_damaged = np.random.choice(damaged_ids, size=num_damaged_test, replace=False)
    else:
        test_ids_damaged = np.array(damaged_ids)

    if len(undamaged_ids) >= num_undamaged_test:
        test_ids_undamaged = np.random.choice(undamaged_ids, size=num_undamaged_test, replace=False)
    else:
        test_ids_undamaged = np.array(undamaged_ids)

    test_ids = sorted(list(test_ids_damaged) + list(test_ids_undamaged))

    if verbose:
        print(f"Randomly selected test car IDs: {test_ids}")
        print(f"  - Damaged cars: {sorted(test_ids_damaged.tolist())}")
        print(f"  - Undamaged cars: {sorted(test_ids_undamaged.tolist())}")
        print()

    train_df = df[~df['id_number'].isin(test_ids)].drop(columns=['id_number']).reset_index(drop=True)
    val_df = df[df['id_number'].isin(test_ids)].drop(columns=['id_number']).reset_index(drop=True)

    if verbose:
        print(f"Training samples: {len(train_df)} (from {len(unique_ids) - len(test_ids)} cars)")
        print(f"Validation samples: {len(val_df)} (from {len(test_ids)} cars)")
        print()
        print("=" * 80)
        print("VALIDATION DATA DISTRIBUTION")
        print("=" * 80)
        val_categories = val_df.iloc[:, 1::2].values
        total_damaged_parts = (val_categories == 1).sum()
        total_undamaged_parts = (val_categories == 0).sum()
        print(f"Total damaged parts in validation: {total_damaged_parts}")
        print(f"Total undamaged parts in validation: {total_undamaged_parts}")
        for i, name in enumerate(LOCATION_NAMES):
            damaged_count = (val_categories[:, i] == 1).sum()
            print(f"  {name}: {damaged_count} damaged out of {len(val_df)} samples")
        print()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    train_dataset = CarDamageDataset(train_df, IMAGES_DIR, transform=transform)
    val_dataset = CarDamageDataset(val_df, IMAGES_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader

# ============================================================
# TRAINING / EVALUATION HELPERS
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print()

def train_model(
        model,
        criterion,
        model_name,
        train_loader,
        val_loader,
        num_epochs,
        learning_rate,
        train_dataset_len,
        val_dataset_len,
        category_threshold,
        severity_weight,
        severity_boost
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    train_cat_losses, train_sev_losses = [], []

    print("=" * 80)
    print(f"TRAINING MODEL: {model_name}")
    print("=" * 80)
    print(f"Category threshold: {category_threshold}")
    print(f"Severity weight: {severity_weight}x")
    print(f"Severity boost: {severity_boost}x for classified damaged parts")
    print("=" * 80)
    print()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_cat_loss = 0.0
        train_sev_loss = 0.0

        for images, categories, severities, _ in train_loader:
            images = images.to(device)
            categories = categories.to(device)
            severities = severities.to(device)

            optimizer.zero_grad()
            category_logits, severity_raw = model(images, apply_hierarchy=False)
            total_loss, cat_loss_val, sev_loss_val = criterion(
                category_logits, severity_raw, categories, severities
            )
            total_loss.backward()
            optimizer.step()

            batch_size_local = images.size(0)
            train_loss += total_loss.item() * batch_size_local
            train_cat_loss += cat_loss_val * batch_size_local
            train_sev_loss += sev_loss_val * batch_size_local

        train_loss /= train_dataset_len
        train_cat_loss /= train_dataset_len
        train_sev_loss /= train_dataset_len
        train_losses.append(train_loss)
        train_cat_losses.append(train_cat_loss)
        train_sev_losses.append(train_sev_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, categories, severities, _ in val_loader:
                images = images.to(device)
                categories = categories.to(device)
                severities = severities.to(device)
                category_logits, severity_raw = model(images, apply_hierarchy=False)
                total_loss, _, _ = criterion(category_logits, severity_raw, categories, severities)
                val_loss += total_loss.item() * images.size(0)

        val_loss /= val_dataset_len
        val_losses.append(val_loss)

        print(f'[{model_name}] Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Total Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}')
        print(f'  Category Loss: {train_cat_loss:.4f}, Severity Loss: {train_sev_loss:.4f}')
        print()

    return model, {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_cat_loss": train_cat_losses,
        "train_sev_loss": train_sev_losses
    }


def evaluate_model(
        model,
        model_name,
        val_loader,
        predictions_csv_prefix,
        category_threshold,
        save_predictions_csv=True,
        csv_suffix=""
):
    model.eval()
    all_category_logits = []
    all_severity_preds = []
    all_category_labels = []
    all_severity_labels = []
    all_image_ids = []

    with torch.no_grad():
        for images, categories, severities, image_ids in val_loader:
            images = images.to(device)
            category_logits, severity_preds = model(
                images, apply_hierarchy=True, threshold=category_threshold
            )
            all_category_logits.append(category_logits.cpu().numpy())
            all_severity_preds.append(severity_preds.cpu().numpy())
            all_category_labels.append(categories.numpy())
            all_severity_labels.append(severities.numpy())
            all_image_ids.extend(image_ids)

    all_category_logits = np.vstack(all_category_logits)
    all_severity_preds = np.vstack(all_severity_preds)
    all_category_labels = np.vstack(all_category_labels)
    all_severity_labels = np.vstack(all_severity_labels)

    all_category_probs = 1 / (1 + np.exp(-all_category_logits))
    all_category_binary = (all_category_probs > category_threshold).astype(int)
    all_severity_rounded = np.round(np.clip(all_severity_preds, 0, 5)).astype(int)

    print("\n" + "=" * 80)
    print(f"CATEGORY PREDICTION PERFORMANCE – {model_name}")
    print("=" * 80)

    category_accuracy = (all_category_binary == all_category_labels.astype(int)).mean() * 100
    print(f"Overall Category Accuracy: {category_accuracy:.2f}%")
    print()

    for i, name in enumerate(LOCATION_NAMES):
        true_labels = all_category_labels[:, i].astype(int)
        pred_labels = all_category_binary[:, i]
        acc = (pred_labels == true_labels).mean() * 100

        tp = ((pred_labels == 1) & (true_labels == 1)).sum()
        fp = ((pred_labels == 1) & (true_labels == 0)).sum()
        tn = ((pred_labels == 0) & (true_labels == 0)).sum()
        fn = ((pred_labels == 0) & (true_labels == 1)).sum()

        print(f"{name:<20} Accuracy: {acc:5.2f}% | TP:{tp:3d} FP:{fp:3d} TN:{tn:3d} FN:{fn:3d}")

    print("\n" + "-" * 80)
    print(f"SEVERITY PREDICTION PERFORMANCE (damaged parts only) – {model_name}")
    print("-" * 80)

    damage_mask = all_category_labels.astype(int) == 1
    correct_severities = 0
    total_damaged = 0
    per_location_sev_acc = []

    for i in range(NUM_PARTS):
        mask = damage_mask[:, i]
        if mask.sum() > 0:
            correct = (all_severity_rounded[:, i][mask] ==
                       all_severity_labels[:, i][mask].astype(int)).sum()
            total = mask.sum()
            correct_severities += correct
            total_damaged += total
            acc = correct / total * 100
            per_location_sev_acc.append(acc)
            print(f"{LOCATION_NAMES[i]:<20} Severity Accuracy: {acc:5.2f}% ({correct}/{total} damaged parts)")
        else:
            per_location_sev_acc.append(np.nan)
            print(f"{LOCATION_NAMES[i]:<20} Severity Accuracy: N/A (no damaged parts in validation)")

    overall_sev_acc = 0.0
    if total_damaged > 0:
        overall_sev_acc = correct_severities / total_damaged * 100
        print(f"\nOverall Severity Accuracy (damaged parts only): {overall_sev_acc:.2f}% "
              f"({correct_severities}/{total_damaged})")
    else:
        print(f"\nOverall Severity Accuracy: N/A (no damaged parts in validation)")

    if save_predictions_csv:
        predictions_df = pd.DataFrame({'Image_ID': all_image_ids})
        for i, name in enumerate(LOCATION_NAMES):
            predictions_df[f'True_Cat_{name}'] = all_category_labels[:, i].astype(int)
            predictions_df[f'Pred_Cat_{name}'] = all_category_binary[:, i]
            predictions_df[f'Pred_Prob_{name}'] = all_category_probs[:, i]
            predictions_df[f'True_Sev_{name}'] = all_severity_labels[:, i].astype(int)
            predictions_df[f'Pred_Sev_{name}'] = all_severity_rounded[:, i]
            predictions_df[f'Pred_Sev_Raw_{name}'] = all_severity_preds[:, i]

        csv_path = f"{predictions_csv_prefix}{csv_suffix}.csv"
        predictions_df.to_csv(csv_path, index=False)
        print(f"\nAll predictions for {model_name} saved to {csv_path}")

    y_true = all_category_labels.flatten().astype(int)
    y_scores = all_category_probs.flatten()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        "category_accuracy": category_accuracy,
        "severity_accuracy": overall_sev_acc,
        "per_location_severity_accuracy": per_location_sev_acc,
        "category_probs": all_category_probs,
        "category_labels": all_category_labels,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "all_image_ids": all_image_ids,
        "all_category_binary": all_category_binary,
        "all_severity_rounded": all_severity_rounded,
        "all_severity_labels": all_severity_labels
    }

# ============================================================
# BASELINE RUN: A + B + C + D
# ============================================================

# Prepare data for baseline
train_dataset, val_dataset, train_loader, val_loader = prepare_data(
    batch_size=BATCH_SIZE,
    random_seed=RANDOM_SEED,
    num_test_car_ids=NUM_TEST_CAR_IDS,
    verbose=True
)

results_baseline = {}
histories_baseline = {}

# Model A: Regression
model_reg = RegressionSeverityModel()
criterion_reg = RegressionLoss(category_weight=CATEGORY_WEIGHT, severity_weight=SEVERITY_WEIGHT)
model_reg, hist_reg = train_model(
    model_reg, criterion_reg, "Model A – Regression",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    train_dataset_len=len(train_dataset),
    val_dataset_len=len(val_dataset),
    category_threshold=CATEGORY_THRESHOLD,
    severity_weight=SEVERITY_WEIGHT,
    severity_boost=SEVERITY_BOOST
)
histories_baseline['Regression'] = hist_reg
results_baseline['Regression'] = evaluate_model(
    model_reg, "Model A – Regression", val_loader,
    PREDICTIONS_CSV_PREFIX_BASELINE,
    category_threshold=CATEGORY_THRESHOLD,
    csv_suffix="reg"
)

# Model B: Classification
model_cls = ClassificationSeverityModel()
criterion_cls = ClassificationLoss(category_weight=CATEGORY_WEIGHT, severity_weight=SEVERITY_WEIGHT)
model_cls, hist_cls = train_model(
    model_cls, criterion_cls, "Model B – Classification",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    train_dataset_len=len(train_dataset),
    val_dataset_len=len(val_dataset),
    category_threshold=CATEGORY_THRESHOLD,
    severity_weight=SEVERITY_WEIGHT,
    severity_boost=SEVERITY_BOOST
)
histories_baseline['Classification'] = hist_cls
results_baseline['Classification'] = evaluate_model(
    model_cls, "Model B – Classification", val_loader,
    PREDICTIONS_CSV_PREFIX_BASELINE,
    category_threshold=CATEGORY_THRESHOLD,
    csv_suffix="cls"
)

# Model C: Ordinal
model_ord = OrdinalSeverityModel()
criterion_ord = OrdinalLoss(category_weight=CATEGORY_WEIGHT, severity_weight=SEVERITY_WEIGHT)
model_ord, hist_ord = train_model(
    model_ord, criterion_ord, "Model C – Ordinal",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    train_dataset_len=len(train_dataset),
    val_dataset_len=len(val_dataset),
    category_threshold=CATEGORY_THRESHOLD,
    severity_weight=SEVERITY_WEIGHT,
    severity_boost=SEVERITY_BOOST
)
histories_baseline['Ordinal'] = hist_ord
results_baseline['Ordinal'] = evaluate_model(
    model_ord, "Model C – Ordinal", val_loader,
    PREDICTIONS_CSV_PREFIX_BASELINE,
    category_threshold=CATEGORY_THRESHOLD,
    csv_suffix="ord"
)

# Model D: ResNet Regression
model_resnet = ResNetHierarchicalRegression()
criterion_resnet = RegressionLoss(category_weight=CATEGORY_WEIGHT, severity_weight=SEVERITY_WEIGHT)
model_resnet, hist_resnet = train_model(
    model_resnet, criterion_resnet, "Model D – ResNet Regression",
    train_loader, val_loader,
    num_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    train_dataset_len=len(train_dataset),
    val_dataset_len=len(val_dataset),
    category_threshold=CATEGORY_THRESHOLD,
    severity_weight=SEVERITY_WEIGHT,
    severity_boost=SEVERITY_BOOST
)
histories_baseline['ResNet'] = hist_resnet
results_baseline['ResNet'] = evaluate_model(
    model_resnet, "Model D – ResNet Regression", val_loader,
    PREDICTIONS_CSV_PREFIX_BASELINE,
    category_threshold=CATEGORY_THRESHOLD,
    csv_suffix="resnet"
)

# ---- Baseline plots (all 4 models) ----

plt.figure(figsize=(10, 6))
epochs_range = range(1, NUM_EPOCHS + 1)
for name, hist in histories_baseline.items():
    plt.plot(epochs_range, hist["val_loss"], marker='o', label=f'{name} – Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Total Validation Loss')
plt.title('Validation Loss Comparison – All Models')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH_BASELINE)
plt.close()
print(f"Baseline loss comparison plot saved to {LOSS_PLOT_PATH_BASELINE}")

plt.figure(figsize=(8, 8))
for name, res in results_baseline.items():
    plt.plot(res["fpr"], res["tpr"], label=f'{name} (AUC = {res["roc_auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC Curves – Category Prediction (Baseline)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig(ROC_PLOT_PATH_BASELINE)
plt.close()
print(f"Baseline ROC curves plot saved to {ROC_PLOT_PATH_BASELINE}")

model_names_baseline = ['Regression', 'Classification', 'Ordinal', 'ResNet']
cat_accs_baseline = [results_baseline[m]['category_accuracy'] for m in model_names_baseline]
sev_accs_baseline = [results_baseline[m]['severity_accuracy'] for m in model_names_baseline]

x = np.arange(len(model_names_baseline))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width / 2, cat_accs_baseline, width, label='Category Accuracy')
rects2 = ax.bar(x + width / 2, sev_accs_baseline, width, label='Severity Accuracy (damaged parts)')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Category & Severity Accuracy by Model (Baseline)')
ax.set_xticks(x)
ax.set_xticklabels(model_names_baseline)
ax.legend()
ax.grid(axis='y')

def autolabel(rects, values):
    for rect, v in zip(rects, values):
        height = rect.get_height()
        ax.annotate(f'{v:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1, cat_accs_baseline)
autolabel(rects2, sev_accs_baseline)

fig.tight_layout()
plt.savefig(ACC_BAR_PLOT_PATH_BASELINE)
plt.close()
print(f"Baseline accuracy bar chart saved to {ACC_BAR_PLOT_PATH_BASELINE}")

# ---- Baseline final summary ----

print("\n" + "=" * 80)
print("BASELINE MODEL COMPARISON SUMMARY")
print("=" * 80)

summary_data_baseline = {
    'Model': model_names_baseline,
    'Category Acc (%)': [f"{results_baseline[m]['category_accuracy']:.2f}" for m in model_names_baseline],
    'Severity Acc (%)': [f"{results_baseline[m]['severity_accuracy']:.2f}" for m in model_names_baseline],
    'Category AUC': [f"{results_baseline[m]['roc_auc']:.4f}" for m in model_names_baseline]
}
summary_df_baseline = pd.DataFrame(summary_data_baseline)
print(summary_df_baseline.to_string(index=False))

print("\n" + "=" * 80)
print(f"Baseline outputs saved to '{BASELINE_DIR}'.")
print("=" * 80)

# ---- Extended sample predictions: Model A baseline ----

reg_res = results_baseline['Regression']
all_image_ids = reg_res['all_image_ids']
all_category_probs = reg_res['category_probs']
all_category_binary = reg_res['all_category_binary']
all_category_labels = reg_res['category_labels']
all_severity_rounded = reg_res['all_severity_rounded']
all_severity_labels = reg_res['all_severity_labels']

NUM_SAMPLES_TO_SHOW = min(5, len(val_dataset))

print("\n" + "=" * 80)
print(f"SAMPLE PREDICTIONS – MODEL A (REGRESSION, BASELINE)")
print(f"(Showing first {NUM_SAMPLES_TO_SHOW} validation samples)")
print(f"Category Threshold: {CATEGORY_THRESHOLD}")
print("=" * 80)

def format_sample_prediction(
        image_id, cat_prob, cat_pred, cat_true, sev_pred, sev_true, location_names
):
    output = f"Image: {image_id}\n"
    output += "Location             Damaged                   Severity                      \n"
    output += "---------------------------------------------------------------------------\n"

    for i, name in enumerate(location_names):
        cat_t = int(cat_true[i])
        cat_p = int(cat_pred[i])
        sev_p = int(sev_pred[i])
        sev_t = int(sev_true[i])
        prob = cat_prob[i]

        damage_status = f"T:{cat_t} P:{cat_p} ({prob:.2f})"

        checkmark = " "
        severity_detail = ""

        if cat_t == 0 and cat_p == 0:
            severity_detail = f"No damage (pred Sev:{sev_p})"
            checkmark = "✓"
        elif cat_t == 1 and cat_p == 1:
            severity_detail = f"T: {sev_t} P: {sev_p}"
            if sev_t == sev_p:
                checkmark = "✓"
            else:
                checkmark = "✗"
        elif cat_t == 0 and cat_p == 1:
            severity_detail = f"FP! (pred Sev: {sev_p})"
            checkmark = "✗"
        elif cat_t == 1 and cat_p == 0:
            severity_detail = f"FN! (True Sev: {sev_t})"
            checkmark = "✗"
        else:
            severity_detail = "Error"
            checkmark = "?"

        output += f"{name:<20} {damage_status:<25} {severity_detail:<30} {checkmark}\n"

    return output

for i in range(NUM_SAMPLES_TO_SHOW):
    image_id = all_image_ids[i]
    cat_prob = all_category_probs[i]
    cat_pred = all_category_binary[i]
    cat_true = all_category_labels[i]
    sev_pred = all_severity_rounded[i]
    sev_true = all_severity_labels[i]

    sample_output = format_sample_prediction(
        image_id, cat_prob, cat_pred, cat_true, sev_pred, sev_true, LOCATION_NAMES
    )
    print(sample_output)

print("=" * 80)

# ============================================================
# HYPERPARAMETER SWEEP: A AND D ONLY
# ============================================================

# Sweep grids
SWEEP_SEEDS = [44444, 33]
SWEEP_BATCH_SIZES = [16]
SWEEP_EPOCHS = [8, 14, 20]
SWEEP_LRS = [0.0003, 0.001]
SWEEP_SEVERITY_WEIGHTS = [2.5, 4.0]
SWEEP_THRESHOLDS = [0.35, 0.42, 0.5]
SWEEP_BOOSTS = [1.3, 2.0]

sweep_records = []

def safe_float_str(x):
    return str(x).replace('.', 'p')

def plot_run_AD(histA, histD, resA, resD, run_dir, num_epochs):
    # Loss curves
    plt.figure(figsize=(8, 5))
    epochs_range = range(1, num_epochs + 1)
    plt.plot(epochs_range, histA["val_loss"], marker='o', label='A – Regression')
    plt.plot(epochs_range, histD["val_loss"], marker='o', label='D – ResNet')
    plt.xlabel('Epoch')
    plt.ylabel('Total Validation Loss')
    plt.title('Validation Loss – Models A & D')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path_loss = os.path.join(run_dir, "loss_AD.png")
    plt.savefig(path_loss)
    plt.close()

    # ROC curves
    plt.figure(figsize=(8, 8))
    plt.plot(resA["fpr"], resA["tpr"], label=f'A (AUC = {resA["roc_auc"]:.4f})')
    plt.plot(resD["fpr"], resD["tpr"], label=f'D (AUC = {resD["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves – Models A & D')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    path_roc = os.path.join(run_dir, "roc_AD.png")
    plt.savefig(path_roc)
    plt.close()

    # Accuracy bar chart
    models = ['A_Regression', 'D_ResNet']
    cat_accs = [resA['category_accuracy'], resD['category_accuracy']]
    sev_accs = [resA['severity_accuracy'], resD['severity_accuracy']]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, cat_accs, width, label='Category Accuracy')
    rects2 = ax.bar(x + width / 2, sev_accs, width, label='Severity Accuracy')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Category & Severity Accuracy – Models A & D')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y')

    def autolabel_local(rects, values):
        for rect, v in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{v:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel_local(rects1, cat_accs)
    autolabel_local(rects2, sev_accs)

    fig.tight_layout()
    acc_path = os.path.join(run_dir, "accuracy_AD.png")
    plt.savefig(acc_path)
    plt.close()

    return {
        "loss_plot": path_loss,
        "roc_plot": path_roc,
        "accuracy_plot": acc_path
    }

print("\n" + "=" * 80)
print("STARTING HYPERPARAMETER SWEEP (Models A + D only)")
print("=" * 80)

for seed in SWEEP_SEEDS:
    for bs in SWEEP_BATCH_SIZES:
        for epochs in SWEEP_EPOCHS:
            for lr in SWEEP_LRS:
                for sw in SWEEP_SEVERITY_WEIGHTS:
                    for ct in SWEEP_THRESHOLDS:
                        for sb in SWEEP_BOOSTS:

                            run_name = (
                                f"seed{seed}"
                                f"_bs{bs}"
                                f"_ep{epochs}"
                                f"_lr{safe_float_str(lr)}"
                                f"_sw{safe_float_str(sw)}"
                                f"_ct{safe_float_str(ct)}"
                                f"_sb{safe_float_str(sb)}"
                            )
                            run_dir = os.path.join(SWEEP_ROOT_DIR, run_name)
                            os.makedirs(run_dir, exist_ok=True)

                            print("\n" + "-" * 80)
                            print(f"SWEEP RUN: {run_name}")
                            print("-" * 80)

                            # Prepare data for this run
                            train_dataset_s, val_dataset_s, train_loader_s, val_loader_s = prepare_data(
                                batch_size=bs,
                                random_seed=seed,
                                num_test_car_ids=NUM_TEST_CAR_IDS,
                                verbose=False
                            )

                            # Model A
                            model_A = RegressionSeverityModel()
                            crit_A = RegressionLoss(category_weight=CATEGORY_WEIGHT, severity_weight=sw)
                            model_A, hist_A = train_model(
                                model_A, crit_A, "Model A – Regression (Sweep)",
                                train_loader_s, val_loader_s,
                                num_epochs=epochs,
                                learning_rate=lr,
                                train_dataset_len=len(train_dataset_s),
                                val_dataset_len=len(val_dataset_s),
                                category_threshold=ct,
                                severity_weight=sw,
                                severity_boost=sb
                            )
                            preds_prefix_A = os.path.join(run_dir, "predictions_A_")
                            res_A = evaluate_model(
                                model_A,
                                "Model A – Regression (Sweep)",
                                val_loader_s,
                                preds_prefix_A,
                                category_threshold=ct,
                                csv_suffix="reg"
                            )

                            # Model D
                            model_D = ResNetHierarchicalRegression()
                            crit_D = RegressionLoss(category_weight=CATEGORY_WEIGHT, severity_weight=sw)
                            model_D, hist_D = train_model(
                                model_D, crit_D, "Model D – ResNet Regression (Sweep)",
                                train_loader_s, val_loader_s,
                                num_epochs=epochs,
                                learning_rate=lr,
                                train_dataset_len=len(train_dataset_s),
                                val_dataset_len=len(val_dataset_s),
                                category_threshold=ct,
                                severity_weight=sw,
                                severity_boost=sb
                            )
                            preds_prefix_D = os.path.join(run_dir, "predictions_D_")
                            res_D = evaluate_model(
                                model_D,
                                "Model D – ResNet Regression (Sweep)",
                                val_loader_s,
                                preds_prefix_D,
                                category_threshold=ct,
                                csv_suffix="resnet"
                            )

                            # Per-run plots
                            plot_paths = plot_run_AD(hist_A, hist_D, res_A, res_D, run_dir, epochs)

                            # Save summary.json for this run
                            summary_run = {
                                "run_name": run_name,
                                "seed": seed,
                                "batch_size": bs,
                                "num_epochs": epochs,
                                "learning_rate": lr,
                                "severity_weight": sw,
                                "category_threshold": ct,
                                "severity_boost": sb,
                                "A": {
                                    "auc": float(res_A["roc_auc"]),
                                    "category_acc": float(res_A["category_accuracy"]),
                                    "severity_acc": float(res_A["severity_accuracy"])
                                },
                                "D": {
                                    "auc": float(res_D["roc_auc"]),
                                    "category_acc": float(res_D["category_accuracy"]),
                                    "severity_acc": float(res_D["severity_accuracy"])
                                },
                                "plots": plot_paths
                            }
                            with open(os.path.join(run_dir, "summary.json"), "w") as f:
                                json.dump(summary_run, f, indent=2)

                            # Add record for global summary
                            sweep_records.append({
                                "run_name": run_name,
                                "seed": seed,
                                "batch_size": bs,
                                "num_epochs": epochs,
                                "learning_rate": lr,
                                "severity_weight": sw,
                                "category_threshold": ct,
                                "severity_boost": sb,
                                "A_auc": res_A["roc_auc"],
                                "A_cat_acc": res_A["category_accuracy"],
                                "A_sev_acc": res_A["severity_accuracy"],
                                "D_auc": res_D["roc_auc"],
                                "D_cat_acc": res_D["category_accuracy"],
                                "D_sev_acc": res_D["severity_accuracy"]
                            })

# ============================================================
# GLOBAL SWEEP SUMMARY + TOP-4 COMBOS (AVERAGE A AND D)
# ============================================================

if sweep_records:
    df_runs = pd.DataFrame(sweep_records)

    df_runs["mean_auc"] = (df_runs["A_auc"] + df_runs["D_auc"]) / 2.0
    df_runs["mean_category_acc"] = (df_runs["A_cat_acc"] + df_runs["D_cat_acc"]) / 2.0
    df_runs["mean_severity_acc"] = (df_runs["A_sev_acc"] + df_runs["D_sev_acc"]) / 2.0

    all_summary_path = os.path.join(GLOBAL_SUMMARY_DIR, "all_runs_summary.csv")
    df_runs.to_csv(all_summary_path, index=False)

    top4_auc = df_runs.sort_values("mean_auc", ascending=False).head(4)
    top4_cat = df_runs.sort_values("mean_category_acc", ascending=False).head(4)
    top4_sev = df_runs.sort_values("mean_severity_acc", ascending=False).head(4)

    top4_auc.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_auc.csv"), index=False)
    top4_cat.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_category_acc.csv"), index=False)
    top4_sev.to_csv(os.path.join(GLOBAL_SUMMARY_DIR, "top4_by_mean_severity_acc.csv"), index=False)

    report_lines = []

    def add_section(title, df_top, metric_col):
        report_lines.append(title)
        for i, row in df_top.iterrows():
            line = (
                f"- {row['run_name']}: "
                f"{metric_col}={row[metric_col]:.4f}, "
                f"A_auc={row['A_auc']:.4f}, D_auc={row['D_auc']:.4f}, "
                f"A_cat={row['A_cat_acc']:.2f}%, D_cat={row['D_cat_acc']:.2f}%, "
                f"A_sev={row['A_sev_acc']:.2f}%, D_sev={row['D_sev_acc']:.2f}%"
            )
            report_lines.append(line)
        report_lines.append("")

    add_section("Best runs by mean AUC (avg of A & D):", top4_auc, "mean_auc")
    add_section("Best runs by mean Category Accuracy (avg of A & D):", top4_cat, "mean_category_acc")
    add_section("Best runs by mean Severity Accuracy (avg of A & D):", top4_sev, "mean_severity_acc")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(GLOBAL_SUMMARY_DIR, "combined_top4_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 80)
    print("SWEEP GLOBAL SUMMARY")
    print("=" * 80)
    print(f"All runs summary saved to: {all_summary_path}")
    print(f"Top-4 by mean AUC saved to: {os.path.join(GLOBAL_SUMMARY_DIR, 'top4_by_mean_auc.csv')}")
    print(f"Top-4 by mean Category Accuracy saved to: {os.path.join(GLOBAL_SUMMARY_DIR, 'top4_by_mean_category_acc.csv')}")
    print(f"Top-4 by mean Severity Accuracy saved to: {os.path.join(GLOBAL_SUMMARY_DIR, 'top4_by_mean_severity_acc.csv')}")
    print(f"Combined report: {report_path}")
    print("=" * 80)
else:
    print("No sweep records collected – something went wrong before the sweep loop.")
