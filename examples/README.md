# Examples

This folder contains public demo files that allow users to test the pipeline.

## Files

```text
demo_video_blurred.mp4
demo_skeleton_2d_pixels.csv
```

The video is a blurred version of the original recording. The CSV contains the matching frame-level 2D skeleton landmarks which was extracted in real-time through the developed app using mediapipe.

## Usage in notebooks

The debug notebooks should use these paths:

```python
CSV_PATH = PROJECT_ROOT / "examples" / "demo_skeleton_2d_pixels.csv"
VIDEO_PATH = PROJECT_ROOT / "examples" / "demo_video_blurred.mp4"
```

## Notes

The demo video and skeleton CSV must correspond to the same recording and the same frame numbering.

The demo files are provided only to demonstrate the workflow. For new recordings, replace these paths with the corresponding local video and skeleton CSV.
