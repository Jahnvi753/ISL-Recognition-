import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from training.dataset import ISLDataset
from models.stgcn.model import STGCN
from models.dtw import dtw_distance


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/stgcn/best_model.pt"
TRAIN_ROOT = "dataset_normalized/train"
VAL_ROOT = "dataset_normalized/val"
TEST_ROOT = "dataset_normalized/test"

NUM_FRAMES = 100

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Load DTW training sequences
# --------------------------------------------------

def load_dtw_sequences(root, classes):

    sequences = []

    for class_name in classes:

        class_dir = os.path.join(root, class_name)

        for filename in sorted(os.listdir(class_dir)):

            if not filename.endswith(".npy"):
                continue

            if filename.endswith("_mask.npy"):
                continue

            path = os.path.join(class_dir, filename)

            sequence = np.load(path).astype(np.float32)

            sequences.append(
                (class_name, sequence)
            )

    return sequences


# --------------------------------------------------
# DTW prediction
# --------------------------------------------------

def dtw_predict(sequence, training_sequences):

    best_distance = float("inf")
    best_class = None

    for class_name, template in training_sequences:

        distance = dtw_distance(
            sequence,
            template
        )

        if distance < best_distance:

            best_distance = distance
            best_class = class_name

    return best_class, best_distance


# --------------------------------------------------
# Get ST-GCN prediction
# --------------------------------------------------

def stgcn_predict(model, x, classes):

    x = x.to(device)

    with torch.no_grad():

        outputs = model(x)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        prediction_index = probabilities.argmax(
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction_index
        ].item()

    prediction = classes[prediction_index]

    return prediction, confidence


# --------------------------------------------------
# Evaluate hybrid
# --------------------------------------------------

def evaluate_hybrid(
    model,
    dataset,
    dtw_sequences,
    classes,
    threshold
):

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    correct = 0
    total = 0

    details = []

    with torch.no_grad():

        for x, y in loader:

            true_class = classes[y.item()]

            # ST-GCN
            stgcn_class, confidence = stgcn_predict(
                model,
                x,
                classes
            )

            # DTW
            sequence = x[0].numpy()

            dtw_class, dtw_distance_value = dtw_predict(
                sequence,
                dtw_sequences
            )

            # Hybrid decision
            if confidence >= threshold:

                final_class = stgcn_class
                method = "ST-GCN"

            else:

                final_class = dtw_class
                method = "DTW"

            is_correct = (
                final_class == true_class
            )

            if is_correct:
                correct += 1

            total += 1

            details.append({
                "true": true_class,
                "stgcn": stgcn_class,
                "confidence": confidence,
                "dtw": dtw_class,
                "dtw_distance": dtw_distance_value,
                "final": final_class,
                "method": method,
                "correct": is_correct
            })

    accuracy = (
        correct / total
        if total > 0
        else 0
    )

    return accuracy, details


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Device:", device)

    # --------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    classes = checkpoint["classes"]

    print("Classes:", classes)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = STGCN(
        num_classes=len(classes)
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded:", MODEL_PATH)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    val_dataset = ISLDataset(
        root=VAL_ROOT,
        num_frames=NUM_FRAMES
    )

    test_dataset = ISLDataset(
        root=TEST_ROOT,
        num_frames=NUM_FRAMES
    )

    # --------------------------------------------------
    # Load DTW reference sequences
    # --------------------------------------------------

    dtw_sequences = load_dtw_sequences(
        TRAIN_ROOT,
        classes
    )

    print(
        "DTW training sequences:",
        len(dtw_sequences)
    )

    # --------------------------------------------------
    # Tune threshold on validation ONLY
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD SEARCH")
    print("=" * 60)

    best_threshold = None
    best_val_accuracy = -1

    thresholds = np.arange(
        0.20,
        0.91,
        0.05
    )

    for threshold in thresholds:

        accuracy, _ = evaluate_hybrid(
            model,
            val_dataset,
            dtw_sequences,
            classes,
            float(threshold)
        )

        print(
            f"Threshold {threshold:.2f}"
            f" -> Validation Accuracy: "
            f"{accuracy:.4f}"
        )

        if accuracy > best_val_accuracy:

            best_val_accuracy = accuracy
            best_threshold = float(threshold)

    print(
        f"\nBest threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # --------------------------------------------------
    # Final TEST evaluation
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL HYBRID TEST")
    print("=" * 60)

    test_accuracy, details = evaluate_hybrid(
        model,
        test_dataset,
        dtw_sequences,
        classes,
        best_threshold
    )

    for item in details:

        print(
            f"True: {item['true']:8s} | "
            f"ST-GCN: {item['stgcn']:8s} | "
            f"Conf: {item['confidence']:.3f} | "
            f"DTW: {item['dtw']:8s} | "
            f"Final: {item['final']:8s} | "
            f"Used: {item['method']}"
        )

    correct = sum(
        item["correct"]
        for item in details
    )

    total = len(details)

    print("\n" + "=" * 60)
    print("HYBRID RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{correct}/{total} = {test_accuracy:.4f}"
    )


if __name__ == "__main__":
    main()