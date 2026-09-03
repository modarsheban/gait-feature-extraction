# Data folder

This folder is reserved for local raw data used during development.

Do not commit real participant, patient, hospital, or identifiable video data to the public repository.

## Recommended local structure

```text
data/
└── private/
    ├── original_video.mp4
    └── original_skeleton_2d_pixels.csv
```

The `data/private/` folder is ignored by Git and should stay only on your computer.

## Public demo files

The public files used by the notebooks should be stored in:

```text
examples/
├── demo_video_blurred.mp4
└── demo_skeleton_2d_pixels.csv
```

## Expected CSV format

The skeleton CSV should contain one row per frame.

Required frame column:

```text
Frame
```

Each landmark should have:

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
