# EgoSSL: Sensor-like Time-Series Abstraction for Egocentric Video

This repository contains a lightweight experimental pipeline for exploring
whether egocentric video can be abstracted into **sensor-like time-series
representations**, and to what extent such low-dimensional temporal signals
can support verb-centric egocentric tasks.

The current code and results focus on **a minimal, reproducible setup**
used for visualization and boundary analysis on EPIC-KITCHENS.

Legacy experimental code (earlier phases) is preserved in the repository
but is **not required** to reproduce the results reported here.

---

## 1. Dataset Preparation (Official EPIC-KITCHENS Downloader)

This project uses **EPIC-KITCHENS-100**.

Please follow the official instructions to download videos and annotations:

https://github.com/epic-kitchens/epic-kitchens-download-scripts



Only a **subset of participants and videos** is sufficient to run the
experiments (e.g., P01 and P02).

---

## 2. Quick Egocentric Time-Series Visualization Pipeline

The main entry point is:

epic-tfc/quick_egossl_viz.py

This script performs the following steps:

1. Loads EPIC-KITCHENS action annotations (verb labels + timestamps)
2. Verifies video integrity and availability
3. Extracts **sensor-like temporal signals** from video segments:
   - RGB mean statistics
   - Motion magnitude proxy
4. Builds fixed-length time-series segments
5. Produces:
   - Quantitative metrics (majority baseline, linear probe)
   - Embedding visualization (PCA)
   - Temporal motion profiles for frequent verbs
   - Qualitative random segment visualization

---

## 3. Running the Script

Activate your environment first (example):

```bash
conda activate TF-C
//Then run:

python epic-tfc/quick_egossl_viz.py \
  --actions_csv annotations/EPIC-Kitchens-100-Annotations/EPIC_100_train.csv \
  --max_segments 800 \
  --sample_hz 4 \
  --max_len 32
  
//You may adjust max_segments (e.g., 200 or 800) to study scale effects.