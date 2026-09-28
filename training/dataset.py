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
            name: i
            for i, name in enumerate(self.classes)
        }

        self.samples = []

        for class_name in self.classes:

            class_dir = os.path.join(
                root,
                class_name
            )

            for path in sorted(
                glob.glob(
                    os.path.join(class_dir, "*.npy")
                )
            ):

                if path.endswith("_mask.npy"):
                    continue

                mask_path = path.replace(
                    ".npy",
                    "_mask.npy"
                )

                if not os.path.exists(mask_path):
                    print(
                        f"Warning: missing mask for {path}"
                    )
                    continue

                self.samples.append(
                    (
                        path,
                        mask_path,
                        self.class_to_idx[class_name]
                    )
                )

    def __len__(self):
        return len(self.samples)

    def _resample_sequence(self, data, mask):

        T = data.shape[0]

        if T == self.num_frames:
            return data, mask

        # New temporal positions
        old_positions = np.arange(T)

        new_positions = np.linspace(
            0,
            T - 1,
            self.num_frames
        )

        # --------------------------------------------------
        # Landmark interpolation
        # --------------------------------------------------

        resized_data = np.zeros(
            (self.num_frames, 75, 3),
            dtype=np.float32
        )

        for landmark in range(75):

            for coordinate in range(3):

                resized_data[:, landmark, coordinate] = np.interp(
                    new_positions,
                    old_positions,
                    data[:, landmark, coordinate]
                )

        # --------------------------------------------------
        # Mask interpolation
        #
        # Use nearest-neighbour behavior so that the mask
        # remains effectively 0/1.
        # --------------------------------------------------

        mask_indices = np.rint(
            new_positions
        ).astype(np.int32)

        mask_indices = np.clip(
            mask_indices,
            0,
            T - 1
        )

        resized_mask = mask[mask_indices]

        return resized_data, resized_mask

    def __getitem__(self, index):

        path, mask_path, label = self.samples[index]

        data = np.load(path).astype(np.float32)

        mask = np.load(mask_path).astype(np.float32)

        # --------------------------------------------------
        # Validate landmark shape
        # --------------------------------------------------

        if data.ndim != 3 or data.shape[1:] != (75, 3):

            raise ValueError(
                f"Unexpected landmark shape "
                f"{data.shape} in {path}"
            )

        # --------------------------------------------------
        # Validate mask shape
        # --------------------------------------------------

        if mask.ndim != 2 or mask.shape[1] != 75:

            raise ValueError(
                f"Unexpected mask shape "
                f"{mask.shape} in {mask_path}"
            )

        if data.shape[0] != mask.shape[0]:

            raise ValueError(
                f"Landmark/mask frame mismatch: "
                f"{data.shape[0]} vs {mask.shape[0]} "
                f"in {path}"
            )

        # --------------------------------------------------
        # Resample every sequence to exactly num_frames
        # --------------------------------------------------

        data, mask = self._resample_sequence(
            data,
            mask
        )

        # --------------------------------------------------
        # Remove undetected landmarks
        # --------------------------------------------------

        data = data * mask[..., None]

        return (
            torch.from_numpy(data),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )