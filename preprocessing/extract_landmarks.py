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

DATASET_DIR = ROOT_DIR / "ISL_Dataset"
OUTPUT_DIR = ROOT_DIR / "processed"
MODEL_PATH = ROOT_DIR / "models" / "holistic_landmarker.task"

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CREATE MEDIAPIPE HOLISTIC LANDMARKER
# ============================================================

def create_landmarker():
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )

    return vision.HolisticLandmarker.create_from_options(options)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(video_path, output_path):
    landmarker = create_landmarker()

    print("\n----------------------------------------")
    print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    sequence = []
    presence_sequence = []

    frame_index = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # OpenCV BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        timestamp_ms = int(
            (frame_index / fps) * 1000
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # ----------------------------------------------------
        # Create empty arrays
        # ----------------------------------------------------

        frame_landmarks = np.zeros(
            (75, 3),
            dtype=np.float32
        )

        frame_presence = np.zeros(
            75,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # POSE: 33 landmarks
        # ----------------------------------------------------

        if result.pose_landmarks:

            for i, landmark in enumerate(result.pose_landmarks):

                frame_landmarks[i] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

                frame_presence[i] = 1.0

        # ----------------------------------------------------
        # LEFT HAND: 21 landmarks
        # ----------------------------------------------------

        if result.left_hand_landmarks:

            for i, landmark in enumerate(result.left_hand_landmarks):

                index = 33 + i

                frame_landmarks[index] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

                frame_presence[index] = 1.0

        # ----------------------------------------------------
        # RIGHT HAND: 21 landmarks
        # ----------------------------------------------------

        if result.right_hand_landmarks:

            for i, landmark in enumerate(result.right_hand_landmarks):

                index = 54 + i

                frame_landmarks[index] = [
                    landmark.x,
                    landmark.y,
                    landmark.z
                ]

                frame_presence[index] = 1.0

        # ----------------------------------------------------
        # Save frame
        # ----------------------------------------------------

        sequence.append(frame_landmarks)
        presence_sequence.append(frame_presence)

        frame_index += 1

    cap.release()
    landmarker.close()

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    presence_sequence = np.asarray(
        presence_sequence,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        output_path,
        sequence
    )

    # Save detection/presence information separately
    mask_path = output_path.with_name(
        output_path.stem + "_mask.npy"
    )

    np.save(
        mask_path,
        presence_sequence
    )

    print(f"Frames: {sequence.shape[0]}")
    print(f"Landmarks: {sequence.shape[1]}")
    print(f"Coordinates: {sequence.shape[2]}")
    print(f"Shape: {sequence.shape}")

    print(f"Saved: {output_path.name}")
    print(f"Mask:  {mask_path.name}")

    return True


# ============================================================
# FIND ALL VIDEOS
# ============================================================

video_extensions = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
}

video_files = []

for path in DATASET_DIR.rglob("*"):

    if path.is_file() and path.suffix.lower() in video_extensions:
        video_files.append(path)


video_files.sort()


print("\n========================================")
print("ISL LANDMARK EXTRACTION")
print("========================================")

print(f"Dataset: {DATASET_DIR}")
print(f"Videos found: {len(video_files)}")


# ============================================================
# PROCESS ALL VIDEOS
# ============================================================

successful = 0
failed = 0

for video_path in video_files:

    # Find the sign/class name.
    #
    # Example:
    #
    # C:\ISL-Dataset\Goodbye\goodbye_001.mp4
    #
    # parent.name = Goodbye

    sign_name = video_path.parent.name

    # Create matching output directory

    output_class_dir = OUTPUT_DIR / sign_name

    output_filename = video_path.stem + ".npy"

    output_path = (
        output_class_dir /
        output_filename
    )

    try:

        success = process_video(
            video_path,
            output_path
        )

        if success:
            successful += 1
        else:
            failed += 1

    except Exception as e:

        print("\nERROR:")
        print(video_path)
        print(e)

        failed += 1


# ============================================================
# CLEAN UP
# ============================================================


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("EXTRACTION COMPLETE")
print("========================================")

print(f"Total videos: {len(video_files)}")
print(f"Successful:   {successful}")
print(f"Failed:       {failed}")

print("\nOutput directory:")
print(OUTPUT_DIR)
