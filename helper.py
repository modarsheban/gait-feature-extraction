import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ============================================================
# Landmark definitions for the iPhone / named-column CSV format
# ============================================================

POINTS = {
    "nose": "Nose",

    "left_eye": "LeftEye",
    "right_eye": "RightEye",

    "left_ear": "LeftEar",
    "right_ear": "RightEar",

    "left_shoulder": "LeftShoulder",
    "right_shoulder": "RightShoulder",

    "left_elbow": "LeftElbow",
    "right_elbow": "RightElbow",

    "left_wrist": "LeftWrist",
    "right_wrist": "RightWrist",

    "left_hip": "LeftHip",
    "right_hip": "RightHip",

    "left_knee": "LeftKnee",
    "right_knee": "RightKnee",

    "left_ankle": "LeftAnkle",
    "right_ankle": "RightAnkle",

    "left_heel": "LeftHeel",
    "right_heel": "RightHeel",

    "left_foot_index": "LeftToe",
    "right_foot_index": "RightToe",
}


def get_xyv(df, point):
    """
    Read x, y, and visibility for one landmark from the new CSV format.

    Expected columns:
        LeftAnkle_x
        LeftAnkle_y
        LeftAnkle_vis

    Example:
        get_xyv(df, POINTS["left_ankle"])
        reads:
            LeftAnkle_x, LeftAnkle_y, LeftAnkle_vis
    """

    prefix = point

    x_col = f"{prefix}_x"
    y_col = f"{prefix}_y"
    v_col = f"{prefix}_vis"

    missing = [
        col for col in [x_col, y_col, v_col]
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns for landmark '{prefix}': {missing}"
        )

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(float)

    return x, y, v

def _get_xyzv(df, point):
    """
    Get x, y, z, visibility for a named landmark.

    Expected columns:
        LeftAnkle_x
        LeftAnkle_y
        LeftAnkle_z
        LeftAnkle_vis
    """

    x_col = f"{point}_x"
    y_col = f"{point}_y"
    z_col = f"{point}_z"
    v_col = f"{point}_vis"

    missing = [
        col for col in [x_col, y_col, z_col, v_col]
        if col not in df.columns
    ]

    if missing:
        raise KeyError(f"Missing 3D columns for {point}: {missing}")

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    z = pd.to_numeric(df[z_col], errors="coerce").to_numpy(float)
    v = pd.to_numeric(df[v_col], errors="coerce").to_numpy(float)

    return x, y, z, v


def distance_2d(df, p1, p2):
    """
    Compute 2D distance between two landmarks for every frame.
    """

    x1, y1, _ = get_xyv(df, p1)
    x2, y2, _ = get_xyv(df, p2)

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def mean_visibility(df, p1, p2):
    """
    Compute mean visibility of two landmarks.
    """

    _, _, v1 = get_xyv(df, p1)
    _, _, v2 = get_xyv(df, p2)

    return np.nanmean((v1 + v2) / 2)

# ============================================================
# Video / data helpers
# ============================================================

def read_fps_from_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, frame_count



# ============================================================
# Signal processing helpers
# ============================================================

def smooth_signal(signal, window=11, polyorder=2):
    signal = (
        pd.Series(signal, dtype="float64")
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    if len(signal) < 7:
        return signal

    window = min(window, len(signal) - 1)

    if window % 2 == 0:
        window -= 1

    if window < 5:
        return signal

    return savgol_filter(
        signal,
        window_length=window,
        polyorder=min(polyorder, window - 1)
    )


def derivative_abs_per_sec(signal, fps):
    signal = np.asarray(signal, dtype=float)
    out = np.full(len(signal), np.nan)

    if len(signal) >= 2:
        out[1:] = np.abs(np.diff(signal)) * fps

    return out


def acceleration_abs_per_sec2(trajectory, fps):
    trajectory = np.asarray(trajectory, dtype=float)
    out = np.full(len(trajectory), np.nan)

    if len(trajectory) >= 3:
        out[2:] = np.abs(np.diff(trajectory[1:])) * fps

    return out


# ============================================================
# Safe statistics helpers
# ============================================================

def safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(~np.isfinite(x)) else np.nanmean(x)


def safe_nanmax(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(~np.isfinite(x)) else np.nanmax(x)


def safe_nanmin(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(~np.isfinite(x)) else np.nanmin(x)


def safe_nanvar(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(~np.isfinite(x)) else np.nanvar(x)


def safe_nanstd(x):
    x = np.asarray(x, dtype=float)
    return np.nan if np.all(~np.isfinite(x)) else np.nanstd(x)


def safe_range(x):
    x = np.asarray(x, dtype=float)

    if np.all(~np.isfinite(x)):
        return np.nan

    return np.nanmax(x) - np.nanmin(x)


def signal_summary(prefix, signal):
    return {
        f"{prefix}_mean": safe_nanmean(signal),
        f"{prefix}_max": safe_nanmax(signal),
        f"{prefix}_min": safe_nanmin(signal),
        f"{prefix}_variance": safe_nanvar(signal),
        f"{prefix}_range": safe_range(signal),
    }


# ============================================================
# Geometry / angle helpers
# ============================================================

def angle_2d_from_points(df, a, b, c):
    """
    Angle ABC in degrees.
    Example:
        hip-knee-ankle gives knee angle.
    """

    ax, ay, _ = get_xyv(df, a)
    bx, by, _ = get_xyv(df, b)
    cx, cy, _ = get_xyv(df, c)

    BA = np.column_stack([ax - bx, ay - by])
    BC = np.column_stack([cx - bx, cy - by])

    dot = np.sum(BA * BC, axis=1)

    norm_ba = np.linalg.norm(BA, axis=1)
    norm_bc = np.linalg.norm(BC, axis=1)
    norm = norm_ba * norm_bc

    cosang = np.full(len(df), np.nan)

    valid = norm > 1e-12
    cosang[valid] = dot[valid] / norm[valid]

    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def spine_angle_2d(df):
    """
    Approximate spine/trunk angle using mid-shoulder and mid-hip.
    """

    lsx, lsy, _ = get_xyv(df, POINTS["left_shoulder"])
    rsx, rsy, _ = get_xyv(df, POINTS["right_shoulder"])
    lhx, lhy, _ = get_xyv(df, POINTS["left_hip"])
    rhx, rhy, _ = get_xyv(df, POINTS["right_hip"])

    mid_shoulder_x = (lsx + rsx) / 2
    mid_shoulder_y = (lsy + rsy) / 2
    mid_hip_x = (lhx + rhx) / 2
    mid_hip_y = (lhy + rhy) / 2

    dx = mid_shoulder_x - mid_hip_x
    dy = mid_shoulder_y - mid_hip_y

    return np.degrees(np.arctan2(dx, -dy))


# ============================================================
# Preprocessing helpers
# ============================================================

def correct_foot_group_misdetections(
    df,
    threshold_px=40,
    frames_before=2,
    frames_after=2,
    interpolation_method="spline",
    spline_order=3,
):
    """
    Detect sudden foot-landmark jumps and correct a neighborhood of frames.

    If any of ankle/heel/toe exceeds the threshold on one side,
    all three landmarks of that side are set to NaN for a small frame window,
    then interpolated.

    Example:
        abnormal frame = 153
        frames_before = 2
        frames_after = 2

        corrected frames = 151, 152, 153, 154, 155

    """

    corrected = df.copy()

    foot_groups = {
        "left": [
            POINTS["left_ankle"],
            POINTS["left_heel"],
            POINTS["left_foot_index"],
        ],
        "right": [
            POINTS["right_ankle"],
            POINTS["right_heel"],
            POINTS["right_foot_index"],
        ],
    }

    debug = pd.DataFrame({
        "Frame": corrected["Frame"].to_numpy(int),
    })

    n = len(corrected)

    for side, points in foot_groups.items():
        side_abnormal = np.zeros(n, dtype=bool)
        side_max_jump = np.full(n, np.nan)

        # ------------------------------------------------------------
        # 1. Detect abnormal jump frames
        # ------------------------------------------------------------
        for point in points:
            for axis in ["x", "y"]:
                col = f"{point}_{axis}"

                if col not in corrected.columns:
                    continue

                values = pd.to_numeric(corrected[col], errors="coerce")
                jump = values.diff()
                abs_jump = np.abs(jump.to_numpy(float))

                abnormal = abs_jump > threshold_px
                abnormal = np.nan_to_num(abnormal, nan=False)

                side_abnormal |= abnormal

                if np.all(np.isnan(side_max_jump)):
                    side_max_jump = abs_jump
                else:
                    side_max_jump = np.nanmax(
                        np.vstack([side_max_jump, abs_jump]),
                        axis=0,
                    )

                debug[f"{col}_jump_px"] = jump
                debug[f"{col}_is_abnormal_jump"] = abnormal

        # ------------------------------------------------------------
        # 2. Expand abnormal frames into a correction window
        # ------------------------------------------------------------
        side_correction_mask = np.zeros(n, dtype=bool)

        abnormal_indices = np.where(side_abnormal)[0]

        for idx in abnormal_indices:
            start_idx = max(0, idx - frames_before)
            end_idx = min(n - 1, idx + frames_after)

            side_correction_mask[start_idx:end_idx + 1] = True

        debug[f"{side}_foot_max_jump_px"] = side_max_jump
        debug[f"{side}_foot_is_misdetection"] = side_abnormal
        debug[f"{side}_foot_is_corrected_window"] = side_correction_mask

        # ------------------------------------------------------------
        # 3. Delete ankle + heel + toe for the whole correction window
        # ------------------------------------------------------------
        for point in points:
            for axis in ["x", "y"]:
                col = f"{point}_{axis}"

                if col in corrected.columns:
                    corrected.loc[side_correction_mask, col] = np.nan

        # Optional: also lower visibility during corrected frames
        for point in points:
            vis_col = f"{point}_vis"
            if vis_col in corrected.columns:
                corrected.loc[side_correction_mask, vis_col] = np.nan

    debug["any_foot_misdetection"] = (
        debug["left_foot_is_misdetection"]
        | debug["right_foot_is_misdetection"]
    )

    debug["any_foot_corrected_window"] = (
        debug["left_foot_is_corrected_window"]
        | debug["right_foot_is_corrected_window"]
    )

    # ------------------------------------------------------------
    # 4. Interpolate deleted values
    # ------------------------------------------------------------
    all_points = [
        POINTS["left_ankle"],
        POINTS["left_heel"],
        POINTS["left_foot_index"],
        POINTS["right_ankle"],
        POINTS["right_heel"],
        POINTS["right_foot_index"],
    ]

    for point in all_points:
        for suffix in ["x", "y", "vis"]:
            col = f"{point}_{suffix}"

            if col not in corrected.columns:
                continue

            series = pd.to_numeric(corrected[col], errors="coerce")

            if suffix == "vis":
                corrected[col] = series.interpolate(limit_direction="both")
                continue

            if (
                interpolation_method == "spline"
                and series.notna().sum() > spline_order + 1
            ):
                corrected[col] = (
                    series
                    .interpolate(
                        method="spline",
                        order=spline_order,
                        limit_direction="both",
                    )
                    .interpolate(limit_direction="both")
                )
            else:
                corrected[col] = series.interpolate(limit_direction="both")

    return corrected, debug