import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ISLDataset(Dataset):
    def __init__(self, root="data/processed", num_frames=100):
        self.root = root
        self.num_frames = num_frames

        self.classes = sorted(
            [
                d for d in os.listdir(root)
                if os.path.isdir(os.path.join(root, d))
            ]
        )

        self.class_to_idx = {
            name: i for i, name in enumerate(self.classes)
        }

        self.samples = []

        for class_name in self.classes:
            class_dir = os.path.join(root, class_name)

            for path in sorted(glob.glob(os.path.join(class_dir, "*.npy"))):
                # Ignore mask files
                if path.endswith("_mask.npy"):
                    continue

                self.samples.append(
                    (path, self.class_to_idx[class_name])
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        data = np.load(path).astype(np.float32)

        # Expected shape: (T, 75, 3)
        if data.ndim != 3 or data.shape[1:] != (75, 3):
            raise ValueError(
                f"Unexpected shape {data.shape} in {path}"
            )

        # Normalize sequence length to num_frames
        T = data.shape[0]

        if T >= self.num_frames:
            indices = np.linspace(
                0,
                T - 1,
                self.num_frames
            ).astype(int)

            data = data[indices]

        else:
            padded = np.zeros(
                (self.num_frames, 75, 3),
                dtype=np.float32
            )

            padded[:T] = data

            data = padded

        return (
            torch.from_numpy(data),
            torch.tensor(label, dtype=torch.long)
        )