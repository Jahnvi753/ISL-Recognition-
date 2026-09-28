from pathlib import Path
import random
import shutil


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_DIR = BASE_DIR / "dataset_original"
OUTPUT_DIR = BASE_DIR / "dataset_split"

SPLIT = {
    "train": 0.50,
    "val": 0.25,
    "test": 0.25,
}

random.seed(42)


def main():

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {SOURCE_DIR}"
        )

    class_dirs = [
        d for d in SOURCE_DIR.iterdir()
        if d.is_dir()
        and d.name not in {"train", "val", "test"}
    ]

    if not class_dirs:
        raise ValueError(
            f"No class folders found in {SOURCE_DIR}"
        )

    print("Classes found:")
    for class_dir in class_dirs:
        print(f"  {class_dir.name}")

    # Remove old split
    if OUTPUT_DIR.exists():
        print("\nRemoving old dataset_split...")
        shutil.rmtree(OUTPUT_DIR)

    # Create output directories
    for split in SPLIT:
        (OUTPUT_DIR / split).mkdir(
            parents=True,
            exist_ok=True
        )

    # -------------------------------------------------------
    # Process each class
    # -------------------------------------------------------

    for class_dir in class_dirs:

        # Find landmark files recursively
        landmark_files = sorted(
            f for f in class_dir.rglob("*.npy")
            if not f.name.endswith("_mask.npy")
        )

        random.shuffle(landmark_files)

        total = len(landmark_files)

        if total == 0:
            print(
                f"\nWARNING: {class_dir.name}: "
                "no landmark files found"
            )
            continue

        train_end = int(total * SPLIT["train"])
        val_end = train_end + int(
            total * SPLIT["val"]
        )

        split_files = {
            "train": landmark_files[:train_end],
            "val": landmark_files[train_end:val_end],
            "test": landmark_files[val_end:],
        }

        print(
            f"\n{class_dir.name}: "
            f"{total} landmark samples"
        )

        # ---------------------------------------------------
        # Copy landmark + corresponding mask
        # ---------------------------------------------------

        for split, selected_files in split_files.items():

            destination = (
                OUTPUT_DIR
                / split
                / class_dir.name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            for landmark_file in selected_files:

                # Copy landmark file
                shutil.copy2(
                    landmark_file,
                    destination / landmark_file.name
                )

                # Find corresponding mask
                mask_file = landmark_file.with_name(
                    landmark_file.stem + "_mask.npy"
                )

                if mask_file.exists():

                    shutil.copy2(
                        mask_file,
                        destination / mask_file.name
                    )

                else:

                    print(
                        f"WARNING: mask not found for "
                        f"{landmark_file.name}"
                    )

            print(
                f"  {split}: "
                f"{len(selected_files)} landmark samples"
            )

    print("\n========================================")
    print("DATASET SPLIT COMPLETE")
    print("========================================")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()