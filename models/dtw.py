import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ============================================================
# LANDMARK GROUPS
# ============================================================

# MediaPipe Pose landmarks
POSE_INDICES = list(range(33))

# MediaPipe left hand landmarks
LEFT_HAND_INDICES = list(range(33, 54))

# MediaPipe right hand landmarks
RIGHT_HAND_INDICES = list(range(54, 75))


# Give hands more importance because hand motion is
# especially important for sign recognition.
POSE_WEIGHT = 1.0
HAND_WEIGHT = 2.0


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_sequence(sequence):
    """
    Convert landmark sequence into weighted
    position + velocity features.

    Input:
        (T, 75, 3)

    Output:
        (T, 450)
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.ndim != 3:
        raise ValueError(
            f"Expected 3D sequence, got {sequence.shape}"
        )

    if sequence.shape[1:] != (75, 3):
        raise ValueError(
            f"Expected (T,75,3), got {sequence.shape}"
        )

    # --------------------------------------------------------
    # Copy so original data is never modified
    # --------------------------------------------------------

    position = sequence.copy()

    # --------------------------------------------------------
    # Weight hands more strongly
    # --------------------------------------------------------

    position[:, POSE_INDICES, :] *= POSE_WEIGHT

    position[:, LEFT_HAND_INDICES, :] *= HAND_WEIGHT
    position[:, RIGHT_HAND_INDICES, :] *= HAND_WEIGHT

    # --------------------------------------------------------
    # Calculate velocity
    # --------------------------------------------------------

    velocity = np.zeros_like(position)

    velocity[1:] = (
        position[1:] - position[:-1]
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    position = position.reshape(
        position.shape[0],
        -1
    )

    velocity = velocity.reshape(
        velocity.shape[0],
        -1
    )

    # --------------------------------------------------------
    # Combine position + velocity
    # --------------------------------------------------------

    features = np.concatenate(
        [position, velocity],
        axis=1
    )

    return features


# ============================================================
# DTW DISTANCE
# ============================================================

def dtw_distance(sequence_a, sequence_b):

    a = prepare_sequence(sequence_a)
    b = prepare_sequence(sequence_b)

    distance, _ = fastdtw(
        a,
        b,
        dist=euclidean
    )

    return float(distance)


# ============================================================
# DTW PREDICTION
# ============================================================

def predict_with_dtw(
    sequence,
    templates,
    classes
):

    distances = {}

    for class_name in classes:

        distances[class_name] = dtw_distance(
            sequence,
            templates[class_name]
        )

    prediction = min(
        distances,
        key=distances.get
    )

    return prediction, distances