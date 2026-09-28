from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT_DIR / "dataset" / "train" / "Goodbye" / "Goodbye_1.mp4"
MODEL_PATH = ROOT_DIR / "models" / "holistic_landmarker.task"
OUTPUT_PATH = ROOT_DIR / "processed" / "Goodbye_1.npy"

for required_path in (VIDEO_PATH, MODEL_PATH):
    if not required_path.exists():
        raise FileNotFoundError(f"Required file not found: {required_path}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE MEDIAPIPE HOLISTIC LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=str(MODEL_PATH)
)

options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
)

landmarker = vision.HolisticLandmarker.create_from_options(options)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video FPS: {fps}")


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

sequence = []

frame_index = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # OpenCV uses BGR.
    # MediaPipe expects RGB.
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert NumPy image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    # Timestamp in milliseconds
    timestamp_ms = int((frame_index / fps) * 1000)

    # Run MediaPipe
    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # --------------------------------------------------------
    # Pose: 33 landmarks
    # --------------------------------------------------------

    pose = np.zeros((33, 3), dtype=np.float32)

    if result.pose_landmarks:

        for i, landmark in enumerate(result.pose_landmarks):

            pose[i] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

    # --------------------------------------------------------
    # Left hand: 21 landmarks
    # --------------------------------------------------------

    left_hand = np.zeros((21, 3), dtype=np.float32)

    if result.left_hand_landmarks:

        for i, landmark in enumerate(result.left_hand_landmarks):

            left_hand[i] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

    # --------------------------------------------------------
    # Right hand: 21 landmarks
    # --------------------------------------------------------

    right_hand = np.zeros((21, 3), dtype=np.float32)

    if result.right_hand_landmarks:

        for i, landmark in enumerate(result.right_hand_landmarks):

            right_hand[i] = [
                landmark.x,
                landmark.y,
                landmark.z
            ]

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    frame_landmarks = np.concatenate(
        [
            pose,
            left_hand,
            right_hand
        ],
        axis=0
    )

    # Shape:
    # (75, 3)

    sequence.append(frame_landmarks)

    frame_index += 1


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
landmarker.close()


# ============================================================
# SAVE
# ============================================================

sequence = np.array(sequence, dtype=np.float32)

print("\nExtraction complete!")
print("Number of frames:", sequence.shape[0])
print("Landmarks per frame:", sequence.shape[1])
print("Coordinates:", sequence.shape[2])
print("Final shape:", sequence.shape)

np.save(OUTPUT_PATH, sequence)

print(f"\nSaved to:\n{OUTPUT_PATH}")
