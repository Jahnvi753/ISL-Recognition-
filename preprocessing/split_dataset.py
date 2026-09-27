"""
Splits normalized landmark .npy files into train/val/test folders.

BEFORE RUNNING:
1. Change SOURCE_DIR to wherever your normalized files actually live.
2. Change OUTPUT_DIR to wherever you want the split dataset to go.
3. Make sure SIGNS matches your actual folder names exactly (case-sensitive).

WHAT IT DOES:
For each sign, shuffles that sign's videos and splits them roughly
70% train / 15% val / 15% test, then copies (not moves) the .npy
files into a new folder structure:

    dataset/
        train/Hello/..., train/Goodbye/..., etc.
        val/Hello/..., etc.
        test/Hello/..., etc.

NOTE ON SIGNER-INDEPENDENCE:
This does a random split, NOT a signer-independent split. If your
filenames encode who the signer is (e.g. "signerA_hello_1.npy"),
tell Claude the naming pattern and this script can be upgraded to
group by signer instead of splitting randomly. If you only have
1-2 signers total, a random split is a reasonable fallback — just
say so honestly in the report.
"""

import os
import random
import shutil

# ---- EDIT THESE THREE THINGS ----
SOURCE_DIR = "C:/ISL-Recognition-/normalized"
OUTPUT_DIR = "C:/ISL-Recognition-/dataset"
SIGNS = ["Goodbye", "Hello", "Yes", "No", "Help"]
# ----------------------------------

SPLIT = {"train": 0.50, "val": 0.25, "test": 0.25}
random.seed(42)  # fixed seed so the split is reproducible for the team

for split_name in SPLIT:
    for sign in SIGNS:
        os.makedirs(os.path.join(OUTPUT_DIR, split_name, sign), exist_ok=True)

print("=" * 50)
print("SPLITTING DATASET")
print("=" * 50)

total_train, total_val, total_test = 0, 0, 0

for sign in SIGNS:
    sign_dir = os.path.join(SOURCE_DIR, sign)

    if not os.path.isdir(sign_dir):
        print(f"WARNING: {sign_dir} not found, skipping {sign}")
        continue

    files = [f for f in os.listdir(sign_dir) if f.endswith(".npy")]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    for f in train_files:
        shutil.copy(os.path.join(sign_dir, f), os.path.join(OUTPUT_DIR, "train", sign, f))
    for f in val_files:
        shutil.copy(os.path.join(sign_dir, f), os.path.join(OUTPUT_DIR, "val", sign, f))
    for f in test_files:
        shutil.copy(os.path.join(sign_dir, f), os.path.join(OUTPUT_DIR, "test", sign, f))

    total_train += len(train_files)
    total_val += len(val_files)
    total_test += len(test_files)

    print(f"{sign}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test  (total: {n})")

print("=" * 50)
print(f"TOTAL: {total_train} train, {total_val} val, {total_test} test")
print(f"Output written to: {OUTPUT_DIR}")
print("=" * 50)
