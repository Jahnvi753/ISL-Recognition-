import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from collections import Counter

from training.dataset import ISLDataset
from models.stgcn.model import STGCN


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset_normalized" / "train"
VAL_DIR = BASE_DIR / "dataset_normalized" / "val"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "stgcn"
    / "best_model.pt"
)

NUM_FRAMES = 100

BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 0.001

NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASSES
# ============================================================

CLASSES = [
    "Hello",
    "Help",
    "No",
    "Yes",
    "goodbye"
]


# ============================================================
# DATASETS
# ============================================================

train_dataset = ISLDataset(
    root=TRAIN_DIR,
    num_frames=NUM_FRAMES
)

val_dataset = ISLDataset(
    root=VAL_DIR,
    num_frames=NUM_FRAMES
)


print("========================================")
print("DATASET")
print("========================================")

print(
    f"Training samples:   {len(train_dataset)}"
)

print(
    f"Validation samples: {len(val_dataset)}"
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = [
    sample[-1]
    for sample in train_dataset.samples
]

class_counts = Counter(train_labels)

print("\nClass counts:")

for class_idx, class_name in enumerate(CLASSES):

    print(
        f"{class_name:8s}: "
        f"{class_counts[class_idx]}"
    )


# ============================================================
# BALANCED SAMPLING
# ============================================================

sample_weights = []

for label in train_labels:

    class_count = class_counts[label]

    weight = 1.0 / class_count

    sample_weights.append(weight)


sample_weights = torch.DoubleTensor(
    sample_weights
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = STGCN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=7
)


# ============================================================
# BEST MODEL
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# TRAINING INFO
# ============================================================

print("\n========================================")
print("TRAINING")
print("========================================")

print(f"Device: {DEVICE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Frames: {NUM_FRAMES}")


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            * inputs.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    train_loss /= train_total

    train_accuracy = (
        train_correct
        / train_total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item()
                * inputs.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    val_loss /= val_total

    val_accuracy = (
        val_correct
        / val_total
    )


    # ========================================================
    # LEARNING RATE
    # ========================================================

    scheduler.step(
        val_accuracy
    )

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        marker = " <-- BEST"

    else:

        marker = ""


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.3f} | "
        f"LR: {current_lr:.6f}"
        f"{marker}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Saved model: {MODEL_PATH}"
)