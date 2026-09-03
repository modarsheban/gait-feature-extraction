# Gait Feature Extraction from Smartphone Skeleton Data

This repository contains a Python pipeline for extracting interpretable gait and sit-to-stand features from smartphone-derived 2D skeleton sequences.

The pipeline includes optional skeleton preprocessing, automatic gait-phase segmentation, segment-level feature extraction, and debug visualization.

## Repository structure

```text
gait-feature-extraction/
├── README.md
├── requirements.txt
├── .gitignore
├── check_repository.py
├── helper.py
├── segmentation.py
├── features_registry.py
├── features.py
├── scripts/
│   └── blur_demo_video.py
├── examples/
│   ├── README.md
│   ├── demo_video_blurred.mp4
│   └── demo_skeleton_2d_pixels.csv
├── notebooks/
│   ├── segmentation_debug_notebook.ipynb
│   └── feature_extraction_debug_notebook.ipynb
├── data/
│   └── README.md
└── outputs/
    └── .gitkeep
```

## Main files

- `helper.py`: landmark definitions, coordinate extraction, signal smoothing, derivative computation, safe statistics, angle computation, video FPS reading, and optional correction of sudden foot-landmark jumps.
- `segmentation.py`: automatic detection of sit-to-stand, lateral walking, turning, frontal walking, and frame-level phase labels.
- `features_registry.py`: registry of implemented features, including feature ID, output column, group, phase, and source.
- `features.py`: segment-level feature extraction, step detection, gait descriptors, sit-to-stand features, frame-level debug signals, and total walking summary.
- `scripts/blur_demo_video.py`: utility script used to create a blurred public demo video from a private raw video.
- `notebooks/`: debugging notebooks for validating segmentation and feature extraction.

## Demo data

The `examples/` folder contains a blurred demo video and its matching skeleton CSV:

```text
examples/demo_video_blurred.mp4
examples/demo_skeleton_2d_pixels.csv
```

These files allow users to run the debug notebooks without access to private raw recordings.

## Input CSV format

The input CSV must contain one row per frame. The required frame column is:

```text
Frame
```

Each landmark should have x, y, and visibility columns:

```text
<Landmark>_x
<Landmark>_y
<Landmark>_vis
```

Example:

```text
LeftAnkle_x
LeftAnkle_y
LeftAnkle_vis
RightAnkle_x
RightAnkle_y
RightAnkle_vis
```

The expected landmark names are defined in the `POINTS` dictionary in `helper.py`.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\\Scripts\\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Open the notebooks and run them cell by cell:

```text
notebooks/segmentation_debug_notebook.ipynb
notebooks/feature_extraction_debug_notebook.ipynb
```

Recommended workflow:

```text
1. Run the segmentation debug notebook.
2. Visually check the detected sit-to-stand, lateral, turning, and frontal phases.
3. Run the feature extraction debug notebook.
4. Check feature values, missing values, step events, and frame-level signals.
5. Export final features only after visual validation.
```

## Walking distance settings

Metric gait speed, step length, and stride length require a known walking distance. In the feature extraction notebook, set:

```python
LATERAL_DISTANCE_M = 3.0
FRONTAL_DISTANCE_M = 3.0
```

Adjust these values according to the acquisition protocol.

## Outputs

Generated files are saved in `outputs/`.

When the feature extraction notebook is executed, the main generated files are:

```text
corrected_skeleton.csv
features_debug.csv
signals_debug.csv
foot_anomaly_debug.csv

The `outputs/` folder is ignored by Git except for `.gitkeep`.

## Data privacy

Do not commit raw participant videos, hospital data, or identifiable recordings.

Private files should remain local in:

```text
data/private/
```

Public demo files should be placed in:

```text
examples/
```

## Limitations

- Distance-based features are computed from 2D image coordinates unless metric calibration is added.
- Some distance features should be interpreted as image-coordinate or normalized proxies.
- Automatic segmentation should be visually checked for each new recording.
- Step detection may require parameter tuning for recordings with occlusion, low visibility, or atypical gait patterns.
- Sit-to-stand detection depends on visible hip landmarks and a clear vertical displacement.
