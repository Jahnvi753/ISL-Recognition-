# ISL Recognition

Indian Sign Language (ISL) recognition system. Converts signed video input into text, with the eventual goal of continuous (multi-sign) recognition rather than single isolated signs. Also serves as the base project for a parallel Design and Analysis of Algorithms (DAA) report.
Drive link: https://drive.google.com/drive/folders/13UEQAB3Rl1x2kt_iLgMLAec7cjo02QKh

## Project status

**Completed**
- MediaPipe landmark extraction (75 landmarks/frame: 33 pose + 21 left-hand + 21 right-hand)
- Shoulder-relative landmark normalization (verified working on initial 20-video set)

**In progress**
- Dataset expansion to ~50 videos across multiple signers
- Signer-independent train/val/test split
- ST-GCN model implementation
- DTW-based sequence comparison

**Not yet started**
- ST-GCN vs ST-GCN+DTW evaluation
- Continuous (multi-sign) segmentation
- Real-time / application integration

Do not treat anything in "in progress" or "not yet started" as complete when writing reports.

## Setup

```bash
git clone <repo-url>
cd ISL-Recognition
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Data (raw videos, extracted/normalized `.npy` files) is **not** stored in this repo — see the shared Drive link pinned in the team chat. Download it and place it locally so your folder matches:

```
ISL-Recognition/
    processed/        <- extracted landmarks (from Drive)
    normalized/        <- normalized landmarks (from Drive)
```

## Folder structure (code)

```
ISL-Recognition/
    data_pipeline/     <- extraction + normalization scripts (Member 1)
    stgcn/              <- ST-GCN model + training (Member 2)
    dtw/                <- DTW comparison module (Member 3)
    integration/        <- demo, evaluation metrics, final pipeline (Member 4)
    requirements.txt
    README.md
```

Work inside your own folder. Changes to shared files (e.g. a `config.py` with data paths) should be flagged to the group before merging.

## Team roles

| Member | Owns |
|---|---|
| 1 | Dataset recording, extraction, normalization, signer-independent splits |
| 2 | ST-GCN architecture and training |
| 3 | DTW implementation and ST-GCN vs ST-GCN+DTW comparison |
| 4 | Evaluation metrics, integration, demo |

## Git workflow

- Never commit directly to `main`.
- Create a branch per workstream: `git checkout -b stgcn-model`
- Open a Pull Request to merge into `main` once your piece runs end-to-end.
- `.gitignore` already excludes `venv/`, `*.npy`, `*.mp4`, `__pycache__/` — data never goes in Git.

## Key design decisions

- **ST-GCN vs ST-GCN+DTW is not assumed** — it's being tested experimentally, not treated as settled.
- Dataset splits are **signer-independent** (e.g. signers 1-7 train, 8 val, 9-10 test), not a random shuffle, so the model isn't just memorizing one person's signing style.
- Normalization is shoulder-relative: center = midpoint of pose landmarks 11 & 12, scaled by shoulder distance.
