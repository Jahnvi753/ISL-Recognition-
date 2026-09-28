import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from training.dataset import ISLDataset
from models.stgcn.model import STGCN


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = "dataset_normalized/test"
CHECKPOINT_PATH = "models/stgcn/best_model.pt"

NUM_FRAMES = 100
BATCH_SIZE = 1

CLASSES = [
    "Hello",
    "Help",
    "No",
    "Yes",
    "goodbye"
]


# ============================================================
# MAIN
# ============================================================

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Classes:", CLASSES)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    test_dataset = ISLDataset(
        root=TEST_DIR,
        num_frames=NUM_FRAMES
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Test samples:", len(test_dataset))

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    print("Loaded:", CHECKPOINT_PATH)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = STGCN(
        num_classes=len(CLASSES)
    ).to(device)

    # Support either:
    # 1. checkpoint containing {"model_state_dict": ...}
    # 2. checkpoint containing {"state_dict": ...}
    # 3. raw state_dict

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Assume the checkpoint itself is the state dict
            state_dict = checkpoint

    else:
        raise RuntimeError("Unsupported checkpoint format.")

    model.load_state_dict(state_dict)

    model.eval()

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    correct = 0
    total = 0

    class_correct = {name: 0 for name in CLASSES}
    class_total = {name: 0 for name in CLASSES}

    print()
    print("=" * 40)
    print("TEST RESULTS")
    print("=" * 40)

    with torch.no_grad():

        for batch in test_loader:

            # Dataset may return:
            # (landmarks, label)
            # or
            # (landmarks, mask, label)

            if len(batch) == 2:
                inputs, labels = batch

            elif len(batch) == 3:
                inputs, _, labels = batch

            else:
                raise RuntimeError(
                    f"Unexpected batch format with {len(batch)} elements."
                )

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            probabilities = torch.softmax(outputs, dim=1)

            confidence, predictions = torch.max(
                probabilities,
                dim=1
            )

            for i in range(len(labels)):

                true_idx = labels[i].item()
                pred_idx = predictions[i].item()

                true_class = CLASSES[true_idx]
                pred_class = CLASSES[pred_idx]

                conf = confidence[i].item()

                print(
                    f"True: {true_class:<8} | "
                    f"Predicted: {pred_class:<8} | "
                    f"Confidence: {conf:.3f}"
                )

                total += 1
                class_total[true_class] += 1

                if pred_idx == true_idx:
                    correct += 1
                    class_correct[true_class] += 1

    # --------------------------------------------------------
    # Overall accuracy
    # --------------------------------------------------------

    accuracy = correct / total if total > 0 else 0.0

    print()
    print("=" * 40)
    print("OVERALL")
    print("=" * 40)

    print(
        f"Accuracy: {correct}/{total} = {accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print()
    print("Per-class:")

    for class_name in CLASSES:

        total_class = class_total[class_name]
        correct_class = class_correct[class_name]

        if total_class > 0:
            class_acc = correct_class / total_class
        else:
            class_acc = 0.0

        print(
            f"{class_name:<8} "
            f"{correct_class}/{total_class} "
            f"({class_acc:.2%})"
        )


if __name__ == "__main__":
    main()