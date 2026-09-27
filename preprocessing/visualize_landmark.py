from pathlib import Path

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

VIDEO_PATH = ROOT_DIR / "ISL_Dataset" / "Goodbye" / "Goodbye_1.mp4"
LANDMARK_PATH = ROOT_DIR / "processed" / "Goodbye_001.npy"
OUTPUT_PATH = ROOT_DIR / "processed" / "goodbye_001_preview.mp4"

if not LANDMARK_PATH.exists():
    raise FileNotFoundError(f"Landmark file not found: {LANDMARK_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")


# ============================================================
# LOAD LANDMARKS
# ============================================================

landmarks = np.load(LANDMARK_PATH)

print("Loaded landmarks")
print("Shape:", landmarks.shape)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")


fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video FPS:", fps)
print("Video size:", width, "x", height)


# ============================================================
# CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

out = cv2.VideoWriter(
    str(OUTPUT_PATH),
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PROCESS FRAMES
# ============================================================

frame_index = 0

while True:

    success, frame = cap.read()

    if not success:
        break

    # Stop if we somehow have more video frames than landmarks
    if frame_index >= len(landmarks):
        break

    current_landmarks = landmarks[frame_index]

    # --------------------------------------------------------
    # Draw all 75 landmarks
    # --------------------------------------------------------

    for landmark_index, landmark in enumerate(current_landmarks):

        x = landmark[0]
        y = landmark[1]

        # Skip missing landmarks
        if x == 0 and y == 0:
            continue

        # MediaPipe x/y are normalized from 0 to 1
        pixel_x = int(x * width)
        pixel_y = int(y * height)

        # Keep coordinates inside the image
        pixel_x = max(0, min(width - 1, pixel_x))
        pixel_y = max(0, min(height - 1, pixel_y))

        # Different marker size for body vs hands
        if landmark_index < 33:
            radius = 4
        else:
            radius = 3

        cv2.circle(
            frame,
            (pixel_x, pixel_y),
            radius,
            (0, 255, 0),
            -1
        )

    # --------------------------------------------------------
    # Add frame number
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_index}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Write frame
    # --------------------------------------------------------

    out.write(frame)

    frame_index += 1


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
out.release()

print("\nPreview created!")
print(OUTPUT_PATH)
