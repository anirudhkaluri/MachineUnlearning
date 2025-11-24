# -*- coding: utf-8 -*-
"""vgg_cifar100_unlearning.ipynb

VGG-16 Unlearning on CIFAR-100 with Time-Decay Method
Forget Class: Bicycle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import vgg16
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import pickle
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# HYPERPARAMETERS
# ================================
CLASSES = list(range(100))  # All 100 CIFAR-100 classes
FORGET_CLASS = 8  # Bicycle (you may need to verify the exact index)
BATCH_SIZE = 128
TRAIN_EPOCHS = 30  # VGG may need more epochs on CIFAR-100
TOP_K_FRACTION = 0.05
LAMBDA_DECAY = 0.1
DECAY_STEPS = 20
FINETUNE_LR = 0.0001
FINE_TUNE = True
FINE_TUNE_AFTER_EACH_STEP = True
FINE_TUNE_EPOCH_COUNT = 6
FINETUNE_FULL_BATCH = False
RANDOM_WEIGHT_DECAY = False
CALCULATE_IMPORTANCE_AGAIN = True
SUBSET_IMPORTANCE_COMPUTATION = True
CALIBRATION_SIZE = 500
CALCULATE_IMPORTANCE_ONLY_WITH_FORGET_SET = False

TRAINED_MODEL_PATH = './vgg_cifar100/vgg_trained_cifar100.pth'
IMPORTANCE_DICT_PATH = './vgg_cifar100/vgg_importance_dict.pkl'

# Create directories
for path in [TRAINED_MODEL_PATH, IMPORTANCE_DICT_PATH]:
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

# CIFAR-100 class names (fine labels)
CIFAR100_FINE_LABELS = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]

print(f"\nCIFAR-100 has {len(CIFAR100_FINE_LABELS)} fine-grained classes")
print(f"Forget Class: {CIFAR100_FINE_LABELS[FORGET_CLASS]} (class {FORGET_CLASS})")
print(f"Sample retain classes: {[CIFAR100_FINE_LABELS[i] for i in [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]]}")


# ================================
# 1. VGG-16 Architecture (Modified for CIFAR-100)
# ================================
def create_vgg16_cifar100(num_classes=100):
    """
    Create VGG-16 for CIFAR-100 (RGB, 32x32)
    Modify the architecture for smaller CIFAR-100 images
    """
    model = vgg16(pretrained=False)
    
    # CIFAR-100 images are 32x32 (not 224x224 like ImageNet)
    # We need to modify the first conv layer to avoid too much downsampling
    # Keep the original architecture but be aware of feature map sizes
    
    # Modify classifier for CIFAR-100
    # Original VGG classifier expects 7x7x512 features
    # For CIFAR-100 (32x32), after 5 maxpools (each 2x2), we get 1x1x512
    model.classifier = nn.Sequential(
        nn.Linear(512 * 1 * 1, 4096),  # Adjusted for CIFAR-100
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    
    return model


# ================================
# 2. Data Loading and Preparation
# ================================
def load_cifar100_subset(classes=list(range(100))):
    """Load CIFAR-100 dataset with only specified classes"""
    # CIFAR-100 normalization values
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    # Load full datasets (fine labels)
    train_dataset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100('./data', train=False, download=True, transform=transform_test)

    # Filter for specific classes
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in classes]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in classes]

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset


def create_forget_retain_splits(dataset, forget_class=8, is_train=True):
    """
    Split dataset into forget and retain sets

    Args:
        dataset: The dataset to split
        forget_class: The class to forget (8 = bicycle)
        is_train: Whether this is training data (True) or test data (False)

    Returns:
        forget_subset, retain_subset
    """
    forget_indices = []
    retain_indices = []

    for i, (_, label) in enumerate(dataset):
        if label == forget_class:
            forget_indices.append(i)
        else:
            retain_indices.append(i)

    forget_subset = Subset(dataset, forget_indices)
    retain_subset = Subset(dataset, retain_indices)

    if is_train:
        print(f"  Training - Forget class ({CIFAR100_FINE_LABELS[forget_class]}) samples: {len(forget_indices)}")
        print(f"  Training - Retain classes samples: {len(retain_indices)}")
    else:
        print(f"  Test - Forget class ({CIFAR100_FINE_LABELS[forget_class]}) samples: {len(forget_indices)}")
        print(f"  Test - Retain classes samples: {len(retain_indices)}")

    return forget_subset, retain_subset


# ================================
# 3. Training Functions
# ================================
def train_model(model, train_loader, num_epochs=30, lr=0.001):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1),
                                      'acc': 100. * correct / total,
                                      'lr': optimizer.param_groups[0]['lr']})

        scheduler.step()

    return model


def evaluate_model(model, data_loader, class_wise=False):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if class_wise:
                for label, pred in zip(labels, predicted):
                    label_item = label.item()
                    if label_item not in class_correct:
                        class_correct[label_item] = 0
                        class_total[label_item] = 0
                    class_total[label_item] += 1
                    if label == pred:
                        class_correct[label_item] += 1

    accuracy = 100. * correct / total

    if class_wise:
        class_accuracies = {k: 100. * class_correct[k] / class_total[k]
                            for k in class_correct.keys()}
        return accuracy, class_accuracies

    return accuracy


# ================================
# 4. Weight Importance Estimation (VGG-specific)
# ================================
def compute_weight_importance_vgg_subset(model, data_loader, calibration_size=500):
    """
    Compute importance using FIXED calibration set for VGG
    
    VGG has:
    - Features: Conv layers (13 conv layers in VGG-16)
    - Classifier: FC layers (3 FC layers)
    """
    print("\n" + "=" * 60)
    print("COMPUTING WEIGHT IMPORTANCE (VGG-16 on CIFAR-100)")
    print("=" * 60)

    model.eval()
    importance_dict = {}

    # ✅ EXTRACT FIXED CALIBRATION SET (ONCE!)
    dataset = data_loader.dataset
    dataset_size = len(dataset)
    calibration_size = min(calibration_size, dataset_size)

    print(f"\nDataset size: {dataset_size}")
    print(f"Calibration size: {calibration_size} ({100*calibration_size/dataset_size:.1f}%)")

    # ✅ Random sampling from dataset
    np.random.seed(42)
    torch.manual_seed(42)

    calibration_indices = np.random.choice(
        dataset_size,
        size=calibration_size,
        replace=False
    )

    print(f"Sampled {calibration_size} random indices")
    print(f"Sample indices: {calibration_indices[:20]}...")

    # ✅ Load samples ONCE
    print(f"\nLoading calibration set...")
    calibration_images = []
    calibration_labels = []

    for idx in tqdm(calibration_indices, desc="Loading"):
        img, label = dataset[idx]
        calibration_images.append(img)
        calibration_labels.append(label)

    # ✅ Convert to tensors (FIXED set for all tests)
    calibration_images = torch.stack(calibration_images).to(device)
    calibration_labels = torch.tensor(calibration_labels).to(device)

    print(f"Calibration set shape: {calibration_images.shape}")
    print(f"Calibration labels shape: {calibration_labels.shape}")

    # ✅ Baseline accuracy on FIXED calibration set
    with torch.no_grad():
        baseline_outputs = model(calibration_images)
        _, baseline_preds = baseline_outputs.max(1)
        baseline_correct = baseline_preds.eq(calibration_labels).sum().item()
        baseline_acc = 100. * baseline_correct / calibration_size

    print(f"\nBaseline accuracy on calibration set: {baseline_acc:.2f}%")

    # Track layers to analyze
    layers_to_analyze = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers_to_analyze.append((name, module, 'conv'))
        elif isinstance(module, nn.Linear):
            layers_to_analyze.append((name, module, 'linear'))

    print(f"\nTotal layers to analyze: {len(layers_to_analyze)}")

    # ✅ Test each filter/neuron on SAME calibration set
    for name, module, layer_type in layers_to_analyze:
        if layer_type == 'conv':
            print(f"\n[CONV] {name}: {module.weight.shape}")
            num_filters = module.weight.shape[0]

            for filter_idx in tqdm(range(num_filters), desc=f"  {name}"):
                original_filter = module.weight.data[filter_idx].clone()
                module.weight.data[filter_idx].zero_()

                # ✅ Evaluate on SAME fixed calibration set
                with torch.no_grad():
                    outputs = model(calibration_images)
                    _, preds = outputs.max(1)
                    correct = preds.eq(calibration_labels).sum().item()
                    new_acc = 100. * correct / calibration_size

                importance = baseline_acc - new_acc
                module.weight.data[filter_idx] = original_filter
                importance_dict[(name, filter_idx)] = importance

        elif layer_type == 'linear':
            print(f"\n[FC] {name}: {module.weight.shape}")
            num_neurons = module.weight.shape[0]

            for neuron_idx in tqdm(range(num_neurons), desc=f"  {name}"):
                original_weights = module.weight.data[neuron_idx].clone()
                original_bias = None
                if module.bias is not None:
                    original_bias = module.bias.data[neuron_idx].clone()

                module.weight.data[neuron_idx].zero_()
                if module.bias is not None:
                    module.bias.data[neuron_idx].zero_()

                # ✅ Evaluate on SAME fixed calibration set
                with torch.no_grad():
                    outputs = model(calibration_images)
                    _, preds = outputs.max(1)
                    correct = preds.eq(calibration_labels).sum().item()
                    new_acc = 100. * correct / calibration_size

                importance = baseline_acc - new_acc

                module.weight.data[neuron_idx] = original_weights
                if module.bias is not None:
                    module.bias.data[neuron_idx] = original_bias

                importance_dict[(name, neuron_idx)] = importance

    return importance_dict


def select_random_k_weights(importance_dict, k=0.05):
    """Select random k% of weights/filters (baseline comparison)"""
    all_weights = list(importance_dict.keys())
    num_to_select = max(1, int(len(all_weights) * k))
    random_weights = np.random.choice(len(all_weights), num_to_select, replace=False)
    random_k = [all_weights[i] for i in random_weights]

    print(f"\n{'=' * 60}")
    print(f"SELECTED RANDOM {k * 100:.1f}% WEIGHTS (BASELINE)")
    print(f"Total weights available: {len(importance_dict)}")
    print(f"Randomly selected: {num_to_select}")
    print(f"{'=' * 60}")
    print("\nFirst 20 randomly selected weights:")
    for i, key in enumerate(random_k[:20]):
        importance = importance_dict[key]
        print(f"  {i + 1}. {key[0]}[{key[1]}]: {importance:.4f}% importance")

    selected_importances = [importance_dict[key] for key in random_k]
    print(f"\nRandom Selection Statistics:")
    print(f"  Mean importance: {np.mean(selected_importances):.4f}%")
    print(f"  Median importance: {np.median(selected_importances):.4f}%")
    print(f"  Max importance: {np.max(selected_importances):.4f}%")
    print(f"  Min importance: {np.min(selected_importances):.4f}%")

    return random_k


def select_top_k_important(importance_dict, k=0.05):
    """Select top k% of most important weights/filters"""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    num_to_select = max(1, int(len(sorted_items) * k))
    top_k = [item[0] for item in sorted_items[:num_to_select]]

    print(f"\n{'=' * 60}")
    print(f"SELECTED TOP {k * 100:.1f}% IMPORTANT WEIGHTS")
    print(f"Total weights analyzed: {len(importance_dict)}")
    print(f"Selected: {num_to_select}")
    print(f"{'=' * 60}")
    print("\nTop 20 most important:")
    for i, (key, importance) in enumerate(sorted_items[:20]):
        print(f"  {i + 1}. {key[0]}[{key[1]}]: {importance:.4f}% accuracy drop")

    return top_k


# ================================
# 5. Time-Decay Unlearning
# ================================
def apply_time_decay(model, top_k_weights, lambda_decay=0.15, num_steps=20,
                     forget_test_loader=None, retain_train_loader=None, finetune_lr=0.0001, finetune=True,
                     finetune_after_each_step=True, fine_tune_epoch_count=10, finetune_full_batch=False, 
                     retain_test_loader=None):
    """Apply exponential time decay to selected weights over multiple steps"""
    print(f"\n{'=' * 60}")
    print(f"APPLYING TIME-DECAY UNLEARNING")
    print(f"Lambda: {lambda_decay}, Steps: {num_steps}")
    print(f"{'=' * 60}")

    history = {
        'step': [],
        'forget_acc': [],
        'retain_acc': []
    }

    # Group weights by layer for efficient access
    layer_dict = dict(model.named_modules())
    total_batches = len(retain_train_loader) if retain_train_loader else 0

    if not finetune_full_batch:
        batches_per_step = max(1, total_batches // num_steps) if total_batches > 0 else 0
        print(f"\nFinetuning strategy:")
        print(f"  Total retain batches: {total_batches}")
        print(f"  Batches per decay step: {batches_per_step}")
    else:
        batches_per_step = total_batches
        print(f"\nFinetuning strategy: Full batch ({total_batches} batches) per step")

    retain_iter = iter(retain_train_loader) if retain_train_loader else None

    # Main decay loop
    for step in range(num_steps):
        print(f"\n--- Decay Step {step + 1}/{num_steps} ---")

        # Apply decay to selected weights
        with torch.no_grad():
            for layer_name, idx in top_k_weights:
                layer = layer_dict[layer_name]

                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    # Apply decay: θ(t+Δt) = θ(t) * e^(-λΔt)
                    decay_factor = np.exp(-lambda_decay * 1.0)
                    layer.weight.data[idx] *= decay_factor

                    if layer.bias is not None and idx < layer.bias.shape[0]:
                        layer.bias.data[idx] *= decay_factor

        # Optional: Fine-tune on retain set to maintain performance
        if retain_train_loader and finetune and finetune_after_each_step:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
            criterion = nn.CrossEntropyLoss()

            batches_used = 0
            try:
                for _ in range(batches_per_step):
                    try:
                        inputs, labels = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_train_loader)
                        inputs, labels = next(retain_iter)

                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    batches_used += 1

                if batches_used > 0:
                    print(f"  Finetuned on {batches_used} batches")
            except Exception as e:
                print(f"  Warning: Exception during fine-tuning: {e}")

        # Evaluate after this decay step
        model.eval()

        if forget_test_loader is not None:
            forget_acc = evaluate_model(model, forget_test_loader)
            history['forget_acc'].append(forget_acc)
            print(f"  Forget Accuracy ({CIFAR100_FINE_LABELS[FORGET_CLASS]}): {forget_acc:.2f}%")

        if retain_test_loader is not None:
            retain_acc = evaluate_model(model, retain_test_loader)
            history['retain_acc'].append(retain_acc)
            print(f"  Retain Accuracy: {retain_acc:.2f}%")

        history['step'].append(step + 1)

        # Early stopping if forget accuracy is low enough
        if forget_test_loader is not None and forget_acc < 5.0:
            print(f"\n✓ Target forget accuracy (<5%) reached! Stopping early.")
            break

    # Final finetuning if requested
    if retain_train_loader and finetune and not finetune_after_each_step:
        print("\n" + "=" * 60)
        print("FINETUNING AFTER ALL DECAY STEPS")
        print("=" * 60)

        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(fine_tune_epoch_count):
            model.train()
            print(f"\n--- Finetune Epoch {epoch + 1}/{fine_tune_epoch_count} ---")

            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in retain_train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            model.eval()
            retain_acc = evaluate_model(model, retain_test_loader)
            forget_acc = evaluate_model(model, forget_test_loader)
            print(f"  Loss: {running_loss / len(retain_train_loader):.4f}")
            print(f"  Retain Accuracy: {retain_acc:.2f}%")
            print(f"  Forget Accuracy: {forget_acc:.2f}%")

    model.eval()
    return model, history


# ================================
# 6. Visualization Functions
# ================================
def plot_results(history, save_name='unlearning_results_vgg16_cifar100.png'):
    """Plot unlearning results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot forget accuracy
    ax1.plot(history['step'], history['forget_acc'], 'r-o', label=f'Forget Class ({CIFAR100_FINE_LABELS[FORGET_CLASS]})')
    ax1.set_xlabel('Decay Step')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Forget Class Accuracy (Should Decrease)')
    ax1.grid(True)
    ax1.legend()

    # Plot retain accuracy
    ax2.plot(history['step'], history['retain_acc'], 'b-o', label='Retain Classes')
    ax2.set_xlabel('Decay Step')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Retain Classes Accuracy (Should Stay High)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()

    print(f"\nPlot saved as '{save_name}'")


def visualize_weight_importance(importance_dict, top_n=30, save_name='weight_importance_vgg16_cifar100.png'):
    """Visualize top N most important weights"""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]

    labels = [f"{k[0]}[{k[1]}]" for k, v in top_items]
    values = [v for k, v in top_items]

    plt.figure(figsize=(14, 8))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.xlabel('Importance (Accuracy Drop %)')
    plt.title(
        f'Top {top_n} Most Important Weights for {CIFAR100_FINE_LABELS[FORGET_CLASS].title()} Class (VGG-16 on CIFAR-100)')
    plt.tight_layout()
    plt.savefig(save_name, dpi=150)
    plt.show()

    print(f"\nWeight importance plot saved as '{save_name}'")


# ================================
# 7. Main Execution Pipeline
# ================================
def main():
    print("=" * 60)
    print("TIME-DECAY UNLEARNING FOR VGG-16 ON CIFAR-100")
    print("=" * 60)

    print("\nHyperparameters:")
    print(f"Dataset: CIFAR-100")
    print(f"Classes: All 100 classes")
    print(f"Forget Class: {CIFAR100_FINE_LABELS[FORGET_CLASS]} (class {FORGET_CLASS})")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Training Epochs: {TRAIN_EPOCHS}")
    print(f"Top-K Fraction: {TOP_K_FRACTION}")
    print(f"Lambda Decay: {LAMBDA_DECAY}")
    print(f"Decay Steps: {DECAY_STEPS}")
    print(f"Finetune Learning Rate: {FINETUNE_LR}")
    print(f"Fine-tuning Enabled: {FINE_TUNE}")
    print(f"Fine-tune After Each Step: {FINE_TUNE_AFTER_EACH_STEP}")
    print(f"Fine-tune Epoch Count: {FINE_TUNE_EPOCH_COUNT}")
    print(f"Finetune Full Batch: {FINETUNE_FULL_BATCH}")
    print(f"Random Parameter Selection: {RANDOM_WEIGHT_DECAY}")

    # 1. Load data
    print("\n[STEP 1] Loading CIFAR-100 dataset...")
    train_dataset, test_dataset = load_cifar100_subset(CLASSES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    forget_train_dataset, retain_train_dataset = create_forget_retain_splits(
        train_dataset,
        forget_class=FORGET_CLASS,
        is_train=True
    )

    retain_train_loader = DataLoader(
        retain_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    forget_train_loader = DataLoader(
        forget_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    forget_test_dataset, retain_test_dataset = create_forget_retain_splits(
        test_dataset,
        forget_class=FORGET_CLASS,
        is_train=False
    )
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    retain_test_loader = DataLoader(retain_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Check if model exists
    if os.path.exists(TRAINED_MODEL_PATH):
        print(f"\n✓ Found existing trained model at '{TRAINED_MODEL_PATH}'")
        print(f"  Loading saved model...")
        model = create_vgg16_cifar100(num_classes=len(CLASSES)).to(device)
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=device))
        print(f"  Model loaded successfully!")
    else:
        # 2. Train model
        print(f"\n[STEP 2] Training VGG-16 on CIFAR-100...")
        model = create_vgg16_cifar100(num_classes=len(CLASSES)).to(device)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        model = train_model(model, train_loader, num_epochs=TRAIN_EPOCHS, lr=0.001)
        torch.save(model.state_dict(), TRAINED_MODEL_PATH)
        print(f"✓ Trained model saved to '{TRAINED_MODEL_PATH}'")

    # 3. Evaluate baseline
    print(f"\n[STEP 3] Evaluating baseline performance...")
    test_acc = evaluate_model(model, test_loader)
    forget_acc_before = evaluate_model(model, forget_test_loader)
    retain_acc_before = evaluate_model(model, retain_test_loader)

    print(f"\nBaseline Results:")
    print(f"  Overall Test Accuracy: {test_acc:.2f}%")
    print(f"  Forget Class ({CIFAR100_FINE_LABELS[FORGET_CLASS]}) Accuracy: {forget_acc_before:.2f}%")
    print(f"  Retain Classes Accuracy: {retain_acc_before:.2f}%")

    # Save original model
    original_model = copy.deepcopy(model)

    # 4. Compute weight importance
    print(f"\n[STEP 4] Computing weight importance...")
    if os.path.exists(IMPORTANCE_DICT_PATH) and not CALCULATE_IMPORTANCE_AGAIN:
        print(f"✓ Found existing importance dictionary at '{IMPORTANCE_DICT_PATH}'")
        print(f"  Loading saved importance scores...")
        with open(IMPORTANCE_DICT_PATH, 'rb') as f:
            importance_dict = pickle.load(f)
        print(f"  Loaded {len(importance_dict)} weight importance scores!")
    else:
        print(f"✗ No existing importance dictionary found. Computing from scratch...")
        print(f"Note: This will take a while for VGG-16...")
        importance_start_time = time.time()
        
        importance_data_loader = None
        if CALCULATE_IMPORTANCE_ONLY_WITH_FORGET_SET:
            importance_data_loader = forget_train_loader
        else:
            importance_data_loader = train_loader
            
        if SUBSET_IMPORTANCE_COMPUTATION:
            importance_dict = compute_weight_importance_vgg_subset(
                model, importance_data_loader, calibration_size=CALIBRATION_SIZE
            )
            print(f"Calibration size used: {CALIBRATION_SIZE}")
        else:
            importance_dict = compute_weight_importance_vgg_subset(model, importance_data_loader)
            
        importance_end_time = time.time()
        importance_elapsed_seconds = importance_end_time - importance_start_time
        importance_elapsed_minutes = importance_elapsed_seconds / 60.0
        print(f"IMPORTANCE COMPUTATION TIME")
        print(f"Total time: {importance_elapsed_seconds:.2f} seconds")
        print(f"Total time: {importance_elapsed_minutes:.2f} minutes")
        print(f"Time per weight: {importance_elapsed_seconds/len(importance_dict):.4f} seconds")
        print(f"Weights analyzed: {len(importance_dict)}")

        # Save importance dictionary
        with open(IMPORTANCE_DICT_PATH, 'wb') as f:
            pickle.dump(importance_dict, f)
        print(f"✓ Importance dictionary saved to '{IMPORTANCE_DICT_PATH}'")

    # Visualize importance
    visualize_weight_importance(importance_dict, top_n=30)

    # 5. Select top-k weights
    if RANDOM_WEIGHT_DECAY:
        print(f"\n[STEP 5] Selecting RANDOM {TOP_K_FRACTION * 100}% weights (baseline)...")
        top_k_weights = select_random_k_weights(importance_dict, k=TOP_K_FRACTION)
    else:
        print(f"\n[STEP 5] Selecting TOP {TOP_K_FRACTION * 100}% important weights...")
        top_k_weights = select_top_k_important(importance_dict, k=TOP_K_FRACTION)

    # 6. Apply time-decay unlearning
    print(f"\n[STEP 6] Applying time-decay unlearning...")

    model, history = apply_time_decay(
        model,
        top_k_weights,
        lambda_decay=LAMBDA_DECAY,
        num_steps=DECAY_STEPS,
        forget_test_loader=forget_test_loader,
        retain_train_loader=retain_train_loader,
        finetune_lr=FINETUNE_LR,
        finetune=FINE_TUNE,
        finetune_after_each_step=FINE_TUNE_AFTER_EACH_STEP,
        fine_tune_epoch_count=FINE_TUNE_EPOCH_COUNT,
        finetune_full_batch=FINETUNE_FULL_BATCH,
        retain_test_loader=retain_test_loader
    )

    # Add baseline to history for plotting
    history['step'] = [0] + history['step']
    history['forget_acc'] = [forget_acc_before] + history['forget_acc']
    history['retain_acc'] = [retain_acc_before] + history['retain_acc']

    # 7. Final evaluation
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print(f"{'=' * 60}")

    forget_acc_after = evaluate_model(model, forget_test_loader)
    retain_acc_after = evaluate_model(model, retain_test_loader)
    test_acc_after = evaluate_model(model, test_loader)

    print(f"\nBefore Unlearning:")
    print(f"  Forget Class ({CIFAR100_FINE_LABELS[FORGET_CLASS]}) Accuracy: {forget_acc_before:.2f}%")
    print(f"  Retain Classes Accuracy: {retain_acc_before:.2f}%")
    print(f"  Overall Test Accuracy: {test_acc:.2f}%")

    print(f"\nAfter Unlearning:")
    print(f"  Forget Class ({CIFAR100_FINE_LABELS[FORGET_CLASS]}) Accuracy: {forget_acc_after:.2f}%")
    print(f"  Retain Classes Accuracy: {retain_acc_after:.2f}%")
    print(f"  Overall Test Accuracy: {test_acc_after:.2f}%")

    print(f"\nChanges:")
    print(f"  Forget Class: {forget_acc_before:.2f}% → {forget_acc_after:.2f}% (Δ {forget_acc_after - forget_acc_before:+.2f}%)")
    print(f"  Retain Classes: {retain_acc_before:.2f}% → {retain_acc_after:.2f}% (Δ {retain_acc_after - retain_acc_before:+.2f}%)")

    # 8. Plot results
    plot_results(history)

    # 9. Save model
    torch.save(model.state_dict(), './vgg_cifar100/unlearned_model_vgg16_cifar100.pth')

    print(f"\nModels saved:")
    print(f"  - unlearned_model_vgg16_cifar100.pth")
    print(f"  - vgg_trained_cifar100.pth")

    # 7.5. Per-class evaluation on sample retain classes
    print(f"\n{'=' * 60}")
    print("PER-CLASS ACCURACY ON SAMPLE RETAIN CLASSES")
    print(f"{'=' * 60}")

    # Sample 10 retain classes for detailed analysis
    sample_retain_classes = [i for i in range(20) if i != FORGET_CLASS][:10]
    class_accuracies_before = {}
    class_accuracies_after = {}

    for cls in sample_retain_classes:
        # Create loader for this specific class
        class_indices = [i for i, (_, label) in enumerate(test_dataset) if label == cls]
        class_dataset = Subset(test_dataset, class_indices)
        class_loader = DataLoader(class_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Evaluate before and after
        acc_before = evaluate_model(original_model, class_loader)
        acc_after = evaluate_model(model, class_loader)

        class_accuracies_before[CIFAR100_FINE_LABELS[cls]] = acc_before
        class_accuracies_after[CIFAR100_FINE_LABELS[cls]] = acc_after

        print(f"{CIFAR100_FINE_LABELS[cls]:15s}: {acc_before:.2f}% → {acc_after:.2f}% (Δ {acc_after - acc_before:+.2f}%)")

    # Plot per-class accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(sample_retain_classes))
    width = 0.35

    bars1 = ax.bar(x - width/2, [class_accuracies_before[CIFAR100_FINE_LABELS[c]] for c in sample_retain_classes],
                  width, label='Before Unlearning', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, [class_accuracies_after[CIFAR100_FINE_LABELS[c]] for c in sample_retain_classes],
                  width, label='After Unlearning', color='green', alpha=0.7)

    ax.set_xlabel('Retain Classes (Sample)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy on Sample Retain Classes (Before vs After Unlearning)')
    ax.set_xticks(x)
    ax.set_xticklabels([CIFAR100_FINE_LABELS[c] for c in sample_retain_classes], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    suffix = "_random" if RANDOM_WEIGHT_DECAY else "_topk"
    plt.savefig(f'per_class_accuracy_comparison_vgg_cifar100{suffix}.png', dpi=150)
    plt.show()

    print(f"\nPer-class accuracy plot saved as 'per_class_accuracy_comparison_vgg_cifar100{suffix}.png'")

    return model, original_model, history


if __name__ == "__main__":
    model, original_model, history = main()
