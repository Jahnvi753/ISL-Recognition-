import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "normalized"


# MediaPipe Pose landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


# ============================================================
# NORMALIZE ONE SEQUENCE
# ============================================================

def normalize_sequence(sequence, mask):

    normalized = np.zeros_like(sequence)

    num_frames = sequence.shape[0]

    successful_frames = 0

    for t in range(num_frames):

        frame = sequence[t]
        frame_mask = mask[t]

        # ----------------------------------------------------
        # Check whether both shoulders were detected
        # ----------------------------------------------------

        left_shoulder_present = (
            frame_mask[LEFT_SHOULDER] > 0
        )

        right_shoulder_present = (
            frame_mask[RIGHT_SHOULDER] > 0
        )

        if not (
            left_shoulder_present
            and right_shoulder_present
        ):
            # We cannot reliably normalize this frame.
            # Keep it as zeros for now.
            continue

        # ----------------------------------------------------
        # Get shoulder coordinates
        # ----------------------------------------------------

        left_shoulder = frame[LEFT_SHOULDER]
        right_shoulder = frame[RIGHT_SHOULDER]

        # ----------------------------------------------------
        # Calculate body center
        # ----------------------------------------------------

        center = (
            left_shoulder + right_shoulder
        ) / 2.0

        # ----------------------------------------------------
        # Calculate body scale
        # ----------------------------------------------------

        shoulder_distance = np.linalg.norm(
            left_shoulder - right_shoulder
        )

        # Avoid division by zero
        if shoulder_distance < 1e-6:
            continue

        # ----------------------------------------------------
        # Normalize only detected landmarks
        # ----------------------------------------------------

        detected = frame_mask > 0

        normalized[t, detected] = (
            frame[detected] - center
        ) / shoulder_distance

        successful_frames += 1

    return normalized, successful_frames


# ============================================================
# FIND LANDMARK FILES
# ============================================================

landmark_files = sorted(
    INPUT_DIR.rglob("*.npy")
)

# Don't accidentally process mask files
landmark_files = [
    path
    for path in landmark_files
    if not path.name.endswith("_mask.npy")
]


print("\n========================================")
print("ISL LANDMARK NORMALIZATION")
print("========================================")

print(f"Files found: {len(landmark_files)}")


# ============================================================
# PROCESS
# ============================================================

total = 0
successful = 0
failed = 0


for landmark_path in landmark_files:

    print("\n----------------------------------------")
    print(f"Processing: {landmark_path}")

    try:

        # ----------------------------------------------------
        # Load sequence
        # ----------------------------------------------------

        sequence = np.load(
            landmark_path
        )

        # ----------------------------------------------------
        # Find corresponding mask
        # ----------------------------------------------------

        mask_path = landmark_path.with_name(
            landmark_path.stem + "_mask.npy"
        )

        if not mask_path.exists():

            print("ERROR: Mask not found")
            failed += 1
            continue

        mask = np.load(mask_path)

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        normalized, valid_frames = (
            normalize_sequence(
                sequence,
                mask
            )
        )

        # ----------------------------------------------------
        # Create output path
        # ----------------------------------------------------

        relative_path = landmark_path.relative_to(
            INPUT_DIR
        )

        output_path = (
            OUTPUT_DIR / relative_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        np.save(
            output_path,
            normalized
        )

        print(
            f"Original shape:   {sequence.shape}"
        )

        print(
            f"Normalized shape: {normalized.shape}"
        )

        print(
            f"Valid frames:     "
            f"{valid_frames}/{len(sequence)}"
        )

        print(
            f"Saved: {output_path}"
        )

        successful += 1

    except Exception as e:

        print("ERROR:", e)
        failed += 1

    total += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("NORMALIZATION COMPLETE")
print("========================================")

print(f"Total:      {total}")
print(f"Successful: {successful}")
print(f"Failed:     {failed}")

print("\nOutput:")
print(OUTPUT_DIR)

