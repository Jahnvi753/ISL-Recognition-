import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = BASE_DIR / "dataset_normalized" / "train"

OUTPUT_FILE = BASE_DIR / "models" / "dtw_templates.npz"

CLASSES = [
    "goodbye",
    "Hello",
    "Help",
    "No",
    "Yes",
]


# ============================================================
# RESAMPLE SEQUENCE
# ============================================================

def resample_sequence(sequence, target_frames=100):
    """
    Resample a landmark sequence to a fixed number of frames.

    Input:
        (T, 75, 3)

    Output:
        (target_frames, 75, 3)
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.ndim != 3:
        raise ValueError(
            f"Expected (T,75,3), got {sequence.shape}"
        )

    if sequence.shape[1:] != (75, 3):
        raise ValueError(
            f"Expected (T,75,3), got {sequence.shape}"
        )

    old_frames = sequence.shape[0]

    if old_frames == target_frames:
        return sequence

    old_indices = np.linspace(
        0,
        old_frames - 1,
        old_frames
    )

    new_indices = np.linspace(
        0,
        old_frames - 1,
        target_frames
    )

    output = np.zeros(
        (target_frames, 75, 3),
        dtype=np.float32
    )

    for landmark in range(75):

        for coordinate in range(3):

            output[:, landmark, coordinate] = np.interp(
                new_indices,
                old_indices,
                sequence[:, landmark, coordinate]
            )

    return output


# ============================================================
# BUILD ONE TEMPLATE
# ============================================================

def build_template(files):

    sequences = []

    for file in files:

        sequence = np.load(file)

        sequence = resample_sequence(
            sequence,
            target_frames=100
        )

        sequences.append(sequence)

    if not sequences:
        raise ValueError("No sequences found.")

    # Average the aligned sequences
    template = np.mean(
        np.stack(sequences),
        axis=0
    )

    return template.astype(np.float32)


# ============================================================
# MAIN
# ============================================================

def main():

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{TRAIN_DIR}"
        )

    templates = {}

    print("\n========================================")
    print("BUILDING DTW TEMPLATES")
    print("========================================")

    for class_name in CLASSES:

        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Class directory not found:\n{class_dir}"
            )

        files = sorted(
            f for f in class_dir.glob("*.npy")
            if not f.name.endswith("_mask.npy")
        )

        print(
            f"\n{class_name}: "
            f"{len(files)} training sequences"
        )

        if len(files) == 0:
            raise ValueError(
                f"No landmark files found for {class_name}"
            )

        template = build_template(files)

        templates[class_name] = template

        print(
            f"Template shape: {template.shape}"
        )

    # --------------------------------------------------------
    # Save templates
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        OUTPUT_FILE,
        **templates
    )

    print("\n========================================")
    print("DTW TEMPLATES CREATED")
    print("========================================")

    print(f"Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()