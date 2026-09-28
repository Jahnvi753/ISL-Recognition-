import numpy as np
from pathlib import Path

from models.dtw import dtw_distance


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset_normalized" / "train"
TEST_DIR = BASE_DIR / "dataset_normalized" / "test"

CLASSES = [
    "goodbye",
    "Hello",
    "Help",
    "No",
    "Yes",
]


# ============================================================
# LOAD TRAINING SEQUENCES
# ============================================================

training_data = {}

print("\n========================================")
print("LOADING DTW TRAINING SEQUENCES")
print("========================================")

for class_name in CLASSES:

    class_dir = TRAIN_DIR / class_name

    files = sorted(
        class_dir.glob("*.npy")
    )

    # Ignore masks
    files = [
        f for f in files
        if not f.name.endswith("_mask.npy")
    ]

    training_data[class_name] = []

    for file_path in files:

        sequence = np.load(file_path)

        training_data[class_name].append(
            sequence
        )

    print(
        f"{class_name}: "
        f"{len(training_data[class_name])} sequences"
    )


# ============================================================
# TEST
# ============================================================

correct = 0
total = 0

class_correct = {
    class_name: 0
    for class_name in CLASSES
}

class_total = {
    class_name: 0
    for class_name in CLASSES
}


print("\n========================================")
print("DTW NEAREST-NEIGHBOR TEST")
print("========================================")


for actual_class in CLASSES:

    test_dir = TEST_DIR / actual_class

    test_files = sorted(
        test_dir.glob("*.npy")
    )

    test_files = [
        f for f in test_files
        if not f.name.endswith("_mask.npy")
    ]

    print(
        f"\n{actual_class}: "
        f"{len(test_files)} test samples"
    )

    for test_file in test_files:

        test_sequence = np.load(test_file)

        best_distance = float("inf")
        prediction = None
        best_training_file = None

        # ----------------------------------------------------
        # Compare against EVERY training sequence
        # ----------------------------------------------------

        for candidate_class in CLASSES:

            candidate_dir = TRAIN_DIR / candidate_class

            candidate_files = sorted(
                candidate_dir.glob("*.npy")
            )

            candidate_files = [
                f for f in candidate_files
                if not f.name.endswith("_mask.npy")
            ]

            for training_file in candidate_files:

                training_sequence = np.load(
                    training_file
                )

                distance = dtw_distance(
                    test_sequence,
                    training_sequence
                )

                if distance < best_distance:

                    best_distance = distance

                    prediction = candidate_class

                    best_training_file = (
                        training_file.name
                    )

        # ----------------------------------------------------
        # Accuracy
        # ----------------------------------------------------

        total += 1

        class_total[actual_class] += 1

        if prediction == actual_class:

            correct += 1
            class_correct[actual_class] += 1

        print(
            f"Actual: {actual_class:<8} | "
            f"Predicted: {prediction:<8} | "
            f"Distance: {best_distance:10.2f} | "
            f"Test: {test_file.name:<15} | "
            f"Match: {best_training_file}"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("DTW RESULTS")
print("========================================")

print(
    f"Overall Accuracy: "
    f"{correct}/{total} = "
    f"{correct / total:.4f}"
)

print("\nPer-class accuracy:")

for class_name in CLASSES:

    c = class_correct[class_name]
    t = class_total[class_name]

    accuracy = c / t if t > 0 else 0

    print(
        f"{class_name:<8}: "
        f"{c}/{t} = {accuracy:.4f}"
    )

print("\n========================================")