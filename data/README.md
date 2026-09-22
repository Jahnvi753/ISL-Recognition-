# ISL Dataset

The raw MP4 videos are stored separately from this GitHub repository because the dataset contains large video files.

## Dataset location

The complete dataset is available in our shared Google Drive:

https://drive.google.com/drive/folders/13UEQAB3Rl1x2kt_iLgMLAec7cjo02QKh

## Directory structure

```text
ISL_Dataset/
├── HELLO/
│   ├── hello_001.mp4
│   ├── hello_002.mp4
│   └── ...
├── THANK_YOU/
│   ├── thank_you_001.mp4
│   └── ...
├── HOW/
│   └── ...
└── ...
```

## Dataset format

* Format: MP4
* Input type: RGB video
* Source: Self-recorded ISL gestures
* Signers: 4
* Each folder represents one sign/class

## Preprocessing

The videos are converted into MediaPipe landmark sequences before training.

Expected processed format:

```text
(T, 75, 3)
```

where:

* `T` = number of frames
* `75` = body + left hand + right hand landmarks
* `3` = x, y, z coordinates

