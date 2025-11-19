import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ================================
# 1. LeNet-256 Architecture
# ================================
class LeNet256(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet256, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 28x28 -> 28x28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 14x14 -> 10x10
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 10x10 -> 5x5

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 256)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(256, 128)
        self.relu4 = nn.ReLU()

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


# ================================
# 2. Data Loading and Preparation
# ================================
def load_mnist_subset(classes=[0, 1, 2, 3]):
    """Load MNIST dataset with only specified classes"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    # Filter for specific classes
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in classes]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in classes]

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    return train_subset, test_subset


def create_forget_retain_splits(dataset, forget_class=0, is_train=True):
    """
    Split dataset into forget and retain sets

    Args:
        dataset: The dataset to split
        forget_class: The class to forget
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
        print(f"  Training - Forget class samples: {len(forget_indices)}")
        print(f"  Training - Retain classes samples: {len(retain_indices)}")
    else:
        print(f"  Test - Forget class samples: {len(forget_indices)}")
        print(f"  Test - Retain classes samples: {len(retain_indices)}")

    return forget_subset, retain_subset


# ================================
# 3. Training Functions
# ================================
def train_model(model, train_loader, num_epochs=10, lr=0.001):
    """Train the model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

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
                                      'acc': 100. * correct / total})

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
# 4. Weight Importance Estimation
# ================================
def compute_weight_importance(model, forget_loader):
    """
    Compute importance of each weight/filter by measuring accuracy drop
    when that weight/filter is removed.

    Returns:
        importance_dict: Dictionary mapping (layer_name, index) -> importance_score
    """
    print("\n" + "=" * 60)
    print("COMPUTING WEIGHT IMPORTANCE FOR FORGET CLASS")
    print("=" * 60)

    model.eval()
    importance_dict = {}

    # Get baseline accuracy on forget class
    baseline_acc = evaluate_model(model, forget_loader)
    print(f"\nBaseline accuracy on forget class: {baseline_acc:.2f}%")

    # Iterate through all layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(f"\n[CONV LAYER] {name}: {module.weight.shape}")
            # For convolutional layers, treat each filter as a unit
            num_filters = module.weight.shape[0]

            for filter_idx in tqdm(range(num_filters), desc=f"  Evaluating {name}"):
                # Save original filter
                original_filter = module.weight.data[filter_idx].clone()

                # Zero out the filter
                module.weight.data[filter_idx].zero_()

                # Measure accuracy drop
                new_acc = evaluate_model(model, forget_loader)
                importance = baseline_acc - new_acc

                # Restore filter
                module.weight.data[filter_idx] = original_filter

                # Store importance
                importance_dict[(name, filter_idx)] = importance

        elif isinstance(module, nn.Linear):
            print(f"\n[FC LAYER] {name}: {module.weight.shape}")
            # For fully connected layers, we can do individual weights or neurons
            # Let's treat each output neuron (row) as a unit for efficiency
            num_neurons = module.weight.shape[0]

            for neuron_idx in tqdm(range(num_neurons), desc=f"  Evaluating {name}"):
                # Save original neuron weights
                original_weights = module.weight.data[neuron_idx].clone()
                if module.bias is not None:
                    original_bias = module.bias.data[neuron_idx].clone()

                # Zero out the neuron
                module.weight.data[neuron_idx].zero_()
                if module.bias is not None:
                    module.bias.data[neuron_idx].zero_()

                # Measure accuracy drop
                new_acc = evaluate_model(model, forget_loader)
                importance = baseline_acc - new_acc

                # Restore neuron
                module.weight.data[neuron_idx] = original_weights
                if module.bias is not None:
                    module.bias.data[neuron_idx] = original_bias

                # Store importance
                importance_dict[(name, neuron_idx)] = importance

    return importance_dict


def select_top_k_important(importance_dict, k=0.1):
    """
    Select top k% of most important weights/filters

    Args:
        importance_dict: Dictionary of importance scores
        k: Fraction of weights to select (0 to 1)

    Returns:
        List of (layer_name, index) tuples for top k weights
    """
    # Sort by importance (descending)
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top k%
    num_to_select = max(1, int(len(sorted_items) * k))
    top_k = [item[0] for item in sorted_items[:num_to_select]]

    print(f"\n{'=' * 60}")
    print(f"SELECTED TOP {k * 100:.1f}% IMPORTANT WEIGHTS")
    print(f"Total weights analyzed: {len(importance_dict)}")
    print(f"Selected: {num_to_select}")
    print(f"{'=' * 60}")
    print("\nTop 10 most important:")
    for i, (key, importance) in enumerate(sorted_items[:10]):
        print(f"  {i + 1}. {key[0]}[{key[1]}]: {importance:.4f}% accuracy drop")

    return top_k


# ================================
# 5. Time-Decay Unlearning
# ================================

# ================================
# 5. Time-Decay Unlearning (REFACTORED)
# ================================
def apply_time_decay(model, top_k_weights, lambda_decay=0.1, num_steps=10,
                     forget_loader=None, retain_loader=None, finetune_lr=0.0001,finetune=False,
                     finetune_after_each_step=True,fine_tune_epoch_count=5):
    """
    Apply exponential time decay to selected weights over multiple steps

    Args:
        model: Neural network model
        top_k_weights: List of (layer_name, index) tuples
        lambda_decay: Decay rate
        num_steps: Number of decay steps to perform
        forget_loader: DataLoader for forget set (for evaluation)
        retain_loader: DataLoader for retain set (for fine-tuning)
        finetune_lr: Learning rate for fine-tuning

    Returns:
        model: Updated model
        history: Dictionary with decay history
    """
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

    # Main decay loop - ALL LOOPING HAPPENS HERE
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
        if retain_loader and finetune and finetune_after_each_step:
            model.train()
            optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
            criterion = nn.CrossEntropyLoss()

            # One pass through retain set
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Evaluate after this decay step
        model.eval()

        if forget_loader is not None:
            forget_acc = evaluate_model(model, forget_loader)
            history['forget_acc'].append(forget_acc)
            print(f"  Forget Accuracy: {forget_acc:.2f}%")

        if retain_loader is not None:
            retain_acc = evaluate_model(model, retain_loader)
            history['retain_acc'].append(retain_acc)
            print(f"  Retain Accuracy: {retain_acc:.2f}%")

        history['step'].append(step + 1)

        # Early stopping if forget accuracy is low enough
        if forget_loader is not None and forget_acc < 5.0:
            print(f"\n✓ Target forget accuracy (<5%) reached! Stopping early.")
            break

    if retain_loader and finetune and not finetune_after_each_step:
        print("=" * 60)
        print("FINETUNING AFTER ALL DECAY STEPS")
        print("=" * 60)


        optimizer = optim.Adam(model.parameters(), lr=finetune_lr)
        criterion = nn.CrossEntropyLoss()
        for step in range(fine_tune_epoch_count):
            model.train()
            print(f"\n--- Finetune Epoch {step + 1}/{fine_tune_epoch_count} ---")
            # One pass through retain set
            for inputs, labels in retain_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            retain_acc = evaluate_model(model, retain_loader)
            print(f"  Retain Accuracy: {retain_acc:.2f}%")


    model.eval()
    return model, history


# ================================
# 6. Visualization Functions
# ================================
def plot_results(history):
    """Plot unlearning results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot forget accuracy
    ax1.plot(history['step'], history['forget_acc'], 'r-o', label='Forget Class')
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
    plt.savefig('unlearning_results.png', dpi=150)
    plt.show()

    print(f"\nPlot saved as 'unlearning_results.png'")


def visualize_weight_importance(importance_dict, top_n=20):
    """Visualize top N most important weights"""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_n]

    labels = [f"{k[0]}[{k[1]}]" for k, v in top_items]
    values = [v for k, v in top_items]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.xlabel('Importance (Accuracy Drop %)')
    plt.title(f'Top {top_n} Most Important Weights for Forget Class')
    plt.tight_layout()
    plt.savefig('weight_importance.png', dpi=150)
    plt.show()

    print(f"\nWeight importance plot saved as 'weight_importance.png'")


# ================================
# 7. Main Execution Pipeline
# ================================
def main():
    print("=" * 60)
    print("TIME-DECAY UNLEARNING FOR LENET-256 ON MNIST")
    print("=" * 60)

    # Hyperparameters
    CLASSES = [0, 1, 2, 3]
    FORGET_CLASS = 0
    BATCH_SIZE = 64
    TRAIN_EPOCHS = 10
    TOP_K_FRACTION = 0.1  # Top 10% most important weights
    LAMBDA_DECAY = 0.2
    DECAY_STEPS = 15
    FINETUNE_LR = 0.0001
    FINE_TUNE=True
    FINE_TUNE_AFTER_EACH_STEP=False
    FINE_TUNE_EPOCH_COUNT=5
    print("\nHyperparameters:")
    print(f"Classes: {CLASSES}")
    print(f"Forget Class: {FORGET_CLASS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Training Epochs: {TRAIN_EPOCHS}")
    print(f"Top-K Fraction: {TOP_K_FRACTION}")
    print(f"Lambda Decay: {LAMBDA_DECAY}")
    print(f"Decay Steps: {DECAY_STEPS}")
    print(f"Finetune Learning Rate: {FINETUNE_LR}")
    print(f"Fine-tuning Enabled: {FINE_TUNE}")
    print(f"Fine-tune After Each Step: {FINE_TUNE_AFTER_EACH_STEP}")
    print(f"Fine-tune Epoch Count (if not after each step): {FINE_TUNE_EPOCH_COUNT}")
    # 1. Load data
    print("\n[STEP 1] Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist_subset(CLASSES)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    forget_train_dataset, retain_train_dataset = create_forget_retain_splits(
        train_dataset,
        forget_class=FORGET_CLASS,
        is_train=True
    )

    retain_train_loader = DataLoader(
        retain_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    forget_train_loader = DataLoader(
        forget_train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Create forget/retain splits

    forget_test_dataset, retain_test_dataset = create_forget_retain_splits(
        test_dataset,
        forget_class=FORGET_CLASS,
        is_train=False
    )
    forget_test_loader = DataLoader(forget_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    retain_test_loader = DataLoader(retain_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Forget class train samples ({FORGET_CLASS}) samples: {len(forget_train_dataset)}")
    print(f"Retain classes train samples: {len(retain_train_dataset)}")

    # 2. Train model
    print(f"\n[STEP 2] Training LeNet-256 on classes {CLASSES}...")
    model = LeNet256(num_classes=len(CLASSES)).to(device)
    model = train_model(model, train_loader, num_epochs=TRAIN_EPOCHS)

    # 3. Evaluate baseline
    print(f"\n[STEP 3] Evaluating baseline performance...")
    test_acc = evaluate_model(model, test_loader)
    forget_acc_before = evaluate_model(model, forget_test_loader)
    retain_acc_before = evaluate_model(model, retain_test_loader)

    print(f"\nBaseline Results:")
    print(f"  Overall Test Accuracy: {test_acc:.2f}%")
    print(f"  Forget Class (0) Accuracy: {forget_acc_before:.2f}%")
    print(f"  Retain Classes (1,2,3) Accuracy: {retain_acc_before:.2f}%")

    # Save original model
    original_model = copy.deepcopy(model)

    # 4. Compute weight importance
    print(f"\n[STEP 4] Computing weight importance...")
    importance_dict = compute_weight_importance(model, forget_train_loader)

    # Visualize importance
    visualize_weight_importance(importance_dict)

    # 5. Select top-k weights
    print(f"\n[STEP 5] Selecting top {TOP_K_FRACTION * 100}% important weights...")
    top_k_weights = select_top_k_important(importance_dict, k=TOP_K_FRACTION)

    # 6. Apply time-decay unlearning
    print(f"\n[STEP 6] Applying time-decay unlearning...")

    # Apply one step of decay
    model, history = apply_time_decay(
        model,
        top_k_weights,
        lambda_decay=LAMBDA_DECAY,
        num_steps=DECAY_STEPS,
        forget_loader=forget_test_loader,
        retain_loader=retain_train_loader,
        finetune_lr=FINETUNE_LR,
        finetune=FINE_TUNE,
        finetune_after_each_step=FINE_TUNE_AFTER_EACH_STEP,
        fine_tune_epoch_count=FINE_TUNE_EPOCH_COUNT
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
    print(f"  Forget Class Accuracy: {forget_acc_before:.2f}%")
    print(f"  Retain Classes Accuracy: {retain_acc_before:.2f}%")
    print(f"  Overall Test Accuracy: {test_acc:.2f}%")

    print(f"\nAfter Unlearning:")
    print(f"  Forget Class Accuracy: {forget_acc_after:.2f}%")
    print(f"  Retain Classes Accuracy: {retain_acc_after:.2f}%")
    print(f"  Overall Test Accuracy: {test_acc_after:.2f}%")

    print(f"\nChanges:")
    print(
        f"  Forget Class: {forget_acc_before:.2f}% → {forget_acc_after:.2f}% (Δ {forget_acc_after - forget_acc_before:+.2f}%)")
    print(
        f"  Retain Classes: {retain_acc_before:.2f}% → {retain_acc_after:.2f}% (Δ {retain_acc_after - retain_acc_before:+.2f}%)")

    # 8. Plot results
    plot_results(history)

    # 9. Save model
    torch.save(model.state_dict(), 'unlearned_model.pth')
    print(f"\nUnlearned model saved as 'unlearned_model.pth'")

    return model, original_model, history


if __name__ == "__main__":
    model, original_model, history = main()