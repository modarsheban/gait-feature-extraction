from pathlib import Path
import cv2
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "private" / "SAFE_GAIT_3D_20260708_170453_2D_Pixels.csv"
VIDEO_PATH = PROJECT_ROOT / "data" / "private" / "SAFE_GAIT_3D_20260708_170453.mp4"

OUTPUT_PATH = PROJECT_ROOT / "examples" / "demo_video_blurred.mp4"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

FACE_POINTS = [
    "Nose",
    "LeftEye",
    "RightEye",
    "LeftEar",
    "RightEar",
]

PADDING = 35              # smaller box; increase to 45 or 50 if face is still visible
VISIBILITY_THRESHOLD = 0  # keep low so the box does not disappear
SMOOTHING_WINDOW = 9      # smooth box movement across frames


# ============================================================
# Face box utilities
# ============================================================

def compute_face_box_from_row(row, frame_width, frame_height, padding=PADDING):
    points = []

    for point in FACE_POINTS:
        x_col = f"{point}_x"
        y_col = f"{point}_y"
        v_col = f"{point}_vis"

        if x_col not in row.index or y_col not in row.index:
            continue

        x = row[x_col]
        y = row[y_col]

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if v_col in row.index:
            visibility = row[v_col]
            if np.isfinite(visibility) and visibility < VISIBILITY_THRESHOLD:
                continue

        # Convert normalized coordinates to pixels if needed
        if 0 <= x <= 1 and 0 <= y <= 1:
            x = x * frame_width
            y = y * frame_height

        points.append((float(x), float(y)))

    if len(points) < 1:
        return None

    points = np.asarray(points, dtype=float)

    x_min = np.nanmin(points[:, 0]) - padding
    x_max = np.nanmax(points[:, 0]) + padding
    y_min = np.nanmin(points[:, 1]) - padding
    y_max = np.nanmax(points[:, 1]) + padding

    x_min = max(0, int(round(x_min)))
    y_min = max(0, int(round(y_min)))
    x_max = min(frame_width, int(round(x_max)))
    y_max = min(frame_height, int(round(y_max)))

    if x_max <= x_min or y_max <= y_min:
        return None

    return x_min, y_min, x_max, y_max


def build_interpolated_face_boxes(df, frame_count, frame_width, frame_height):
    """
    Build one face box for every video frame.

    Missing boxes are interpolated, so the blur does not disappear
    when face landmarks are temporarily missing.
    """

    rows = []

    for _, row in df.iterrows():
        frame = int(row["Frame"])
        box = compute_face_box_from_row(row, frame_width, frame_height)

        if box is None:
            continue

        x_min, y_min, x_max, y_max = box

        rows.append({
            "Frame": frame,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        })

    boxes_df = pd.DataFrame(rows)

    if len(boxes_df) == 0:
        raise ValueError("No valid face boxes were detected from the skeleton CSV.")

    boxes_df = (
        boxes_df
        .drop_duplicates("Frame")
        .set_index("Frame")
        .reindex(range(frame_count))
    )

    # Fill missing frames between detected boxes
    boxes_df = boxes_df.interpolate(limit_direction="both")

    # Smooth box movement to avoid jumping
    boxes_df = boxes_df.rolling(
        window=SMOOTHING_WINDOW,
        center=True,
        min_periods=1,
    ).mean()

    boxes_df = boxes_df.round().astype(int)

    # Keep boxes inside the video frame
    boxes_df["x_min"] = boxes_df["x_min"].clip(0, frame_width - 1)
    boxes_df["x_max"] = boxes_df["x_max"].clip(1, frame_width)
    boxes_df["y_min"] = boxes_df["y_min"].clip(0, frame_height - 1)
    boxes_df["y_max"] = boxes_df["y_max"].clip(1, frame_height)

    return boxes_df


def blur_region(frame, box):
    x_min, y_min, x_max, y_max = box

    region = frame[y_min:y_max, x_min:x_max]

    if region.size == 0:
        return frame

    # Strong blur
    blurred = cv2.GaussianBlur(region, (99, 99), 30)
    frame[y_min:y_max, x_min:x_max] = blurred

    return frame


# ============================================================
# Main script
# ============================================================

df = pd.read_csv(CSV_PATH)
df = df.sort_values("Frame").reset_index(drop=True)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise ValueError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video FPS:", fps)
print("Video frame count:", frame_count)
print("Video size:", width, "x", height)

face_boxes = build_interpolated_face_boxes(
    df=df,
    frame_count=frame_count,
    frame_width=width,
    frame_height=height,
)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

frame_index = 0

while True:
    ok, frame = cap.read()

    if not ok:
        break

    box_row = face_boxes.iloc[frame_index]

    box = (
        int(box_row["x_min"]),
        int(box_row["y_min"]),
        int(box_row["x_max"]),
        int(box_row["y_max"]),
    )

    frame = blur_region(frame, box)

    writer.write(frame)
    frame_index += 1

cap.release()
writer.release()

print(f"Blurred video saved to: {OUTPUT_PATH}")