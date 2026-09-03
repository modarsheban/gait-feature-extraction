"""
segmentation.py

Automatic gait-phase segmentation utilities.

It estimates:
    - walking start and stop
    - turn/cornering start and end
    - sit-to-stand start and end
"""

import numpy as np
import pandas as pd
from helper import (
        POINTS,
        get_xyv,
        distance_2d,
        smooth_signal,
    )


# ============================================================
# Small internal helpers
# ============================================================

FRAME_COL = "Frame"


def _frames(df):
    """
    Return frame numbers.

    The final repository expects the CSV frame column to be named 'Frame'.
    """

    if FRAME_COL not in df.columns:
        raise ValueError("Dataframe must contain a 'Frame' column.")

    return df[FRAME_COL].to_numpy(int)

def _find_start_stop_from_motion_mask(
    moving,
    min_true_frames=5,
    max_false_gap_frames=30,
):
    moving = np.asarray(moving, dtype=bool)

    start_idx = None

    for i in range(0, len(moving) - min_true_frames + 1):
        if np.all(moving[i:i + min_true_frames]):
            start_idx = i
            break

    if start_idx is None:
        return None, None

    last_true_idx = start_idx
    false_count = 0

    for i in range(start_idx, len(moving)):
        if moving[i]:
            last_true_idx = i
            false_count = 0
        else:
            false_count += 1

        if false_count > max_false_gap_frames:
            break

    return start_idx, last_true_idx

# ============================================================
# Turn / transition detection
# ============================================================

def estimate_turn_region(
    df,
    change_threshold_ratio=0.30,
    smooth_window=11,
):
    """
    Estimate the turn/cornering region using apparent body width.

    Idea:
        shoulder_width = distance(left_shoulder, right_shoulder)
        hip_width      = distance(left_hip, right_hip)
        body_width     = shoulder_width + hip_width

    During the transition from lateral view to frontal view, the apparent
    body width changes strongly. The maximum change in smoothed body width
    is used as the transition frame.

    Returns:
        transition_frame, turn_start_frame, turn_end_frame, debug_df
    """

    frames = _frames(df)

    shoulder_width = distance_2d(
        df,
        POINTS["left_shoulder"],
        POINTS["right_shoulder"],
    )

    hip_width = distance_2d(
        df,
        POINTS["left_hip"],
        POINTS["right_hip"],
    )

    body_width = shoulder_width + hip_width
    body_width_smooth = smooth_signal(
        body_width,
        window=smooth_window,
        polyorder=2,
    )

    change = np.abs(np.diff(body_width_smooth))

    if len(change) == 0 or np.all(~np.isfinite(change)):
        raise ValueError("Could not estimate turn region from body width.")

    peak_change_idx = int(np.nanargmax(change))
    transition_idx = min(peak_change_idx + 1, len(frames) - 1)
    transition_frame = int(frames[transition_idx])

    threshold = np.nanmax(change) * change_threshold_ratio

    left_change_idx = peak_change_idx
    while left_change_idx > 0 and change[left_change_idx - 1] >= threshold:
        left_change_idx -= 1

    right_change_idx = peak_change_idx
    while right_change_idx < len(change) - 1 and change[right_change_idx + 1] >= threshold:
        right_change_idx += 1

    turn_start_idx = max(0, left_change_idx)
    turn_end_idx = min(len(frames) - 1, right_change_idx + 1)

    turn_start_frame = int(frames[turn_start_idx])
    turn_end_frame = int(frames[turn_end_idx])

    debug_df = pd.DataFrame({
        "Frame": frames,
        "shoulder_width": shoulder_width,
        "hip_width": hip_width,
        "body_width": body_width,
        "body_width_smooth": body_width_smooth,
        "body_width_change": np.r_[np.nan, change],
        "body_width_change_threshold": threshold,
        "is_transition_frame": frames == transition_frame,
        "is_turn_region": (frames >= turn_start_frame) & (frames <= turn_end_frame),
    })

    return transition_frame, turn_start_frame, turn_end_frame, debug_df


# ============================================================
# Walking start / stop detection
# ============================================================

def estimate_walking_start_stop_from_feet(
    df,
    fps,
    search_start_frame=None,
    visibility_threshold=0.5,
    smooth_window_sec=0.20,
    low_percentile=45,
    max_gap_sec=1.50,
    min_bout_sec=0.30,
    use_velocity=True,
):
    """
    Estimate walking start and stop from lower-limb movement.

    Candidate landmarks:
        - left/right foot index
        - left/right heel
        - left/right ankle

    The motion signal is computed from frame-to-frame foot displacement.
    A sustained-motion rule is used to detect the beginning and end of walking.
    """

    frames = _frames(df)

    if len(frames) < 2:
        debug_df = pd.DataFrame({
            "Frame": frames,
            "foot_motion_smooth": np.full(len(frames), np.nan),
            "is_walking_bout": np.full(len(frames), False),
        })
        return int(frames[0]), int(frames[-1]), debug_df

    candidate_points = [
        POINTS["left_foot_index"],
        POINTS["right_foot_index"],
        POINTS["left_heel"],
        POINTS["right_heel"],
        POINTS["left_ankle"],
        POINTS["right_ankle"],
    ]

    frame_gap = np.diff(frames).astype(float)
    dt = frame_gap / float(fps)
    dt[dt <= 0] = np.nan

    motion_signals = []

    for point_id in candidate_points:
        x, y, v = get_xyv(df, point_id)

        displacement = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        if use_velocity:
            motion = displacement / dt
        else:
            motion = displacement

        visibility_pair = np.minimum(v[:-1], v[1:])
        motion[visibility_pair < visibility_threshold] = np.nan

        motion_signals.append(motion)

    motion_stack = np.vstack(motion_signals)

    all_nan_columns = np.all(~np.isfinite(motion_stack), axis=0)
    foot_motion = np.full(motion_stack.shape[1], np.nan)

    valid_columns = ~all_nan_columns

    if np.any(valid_columns):
        # Median is more robust than max because one wrong landmark
        # does not dominate the complete foot-motion signal.
        foot_motion[valid_columns] = np.nanmedian(
            motion_stack[:, valid_columns],
            axis=0,
        )

    smooth_window = max(5, int(smooth_window_sec * fps))
    if smooth_window % 2 == 0:
        smooth_window += 1

    foot_motion_smooth = smooth_signal(
        foot_motion,
        window=smooth_window,
        polyorder=2,
    )

    valid_motion = foot_motion_smooth[np.isfinite(foot_motion_smooth)]

    if len(valid_motion) == 0:
        debug_df = pd.DataFrame({
            "Frame": frames[1:],
            "foot_motion": foot_motion,
            "foot_motion_smooth": foot_motion_smooth,
            "low_threshold": np.nan,
            "is_moving": False,
            "is_walking_bout": False,
        })
        return int(frames[0]), int(frames[-1]), debug_df

    motion_frames = frames[1:]

    if search_start_frame is None:
        valid_search = np.full(len(motion_frames), True)
    else:
        valid_search = motion_frames >= search_start_frame

    valid_motion = foot_motion_smooth[
        valid_search & np.isfinite(foot_motion_smooth)
    ]

    if len(valid_motion) == 0:
        raise ValueError("Could not detect walking: no valid foot motion after sit-to-stand.")

    low_threshold = np.nanpercentile(valid_motion, low_percentile)

    moving = (foot_motion_smooth >= low_threshold) & valid_search

    max_gap_frames = max(1, int(max_gap_sec * fps))
    min_motion_frames = max(1, int(min_bout_sec * fps))

    start_idx, stop_idx = _find_start_stop_from_motion_mask(
        moving,
        min_true_frames=min_motion_frames,
        max_false_gap_frames=max_gap_frames,
    )

    if start_idx is None or stop_idx is None:
        raise ValueError(
            "Could not detect walking start/stop from foot motion. "
            "Use manual walking frames if needed."
        )

    walking_start_frame = int(frames[start_idx + 1])
    walking_stop_frame = int(frames[stop_idx + 1])

    is_walking_bout = np.zeros(len(frames) - 1, dtype=bool)
    is_walking_bout[start_idx:stop_idx + 1] = True

    debug_df = pd.DataFrame({
        "Frame": frames[1:],
        "foot_motion": foot_motion,
        "foot_motion_smooth": foot_motion_smooth,
        "low_threshold": low_threshold,
        "is_moving": moving,
        "is_walking_bout": is_walking_bout,
    })

    return walking_start_frame, walking_stop_frame, debug_df

# ============================================================
# Sit-to-stand detection
# ============================================================
def estimate_sit_to_stand_region(
    df,
    fps,
    search_end_frame=None,
    manual_sts_start_frame=None,
    manual_sts_end_frame=None,
    high_velocity_ratio=0.25,
    boundary_velocity_ratio=0.05,
    min_duration_sec=0.20,
    min_y_change_px=20,
):
    """
    Detect sit-to-stand from the major vertical displacement of the midhip.

    Logic:
        - Compute midhip_y.
        - Smooth it.
        - Find the strongest vertical change before walking/turning.
        - Expand around that change to get STS start and end.
    """

    frames = _frames(df)

    _, left_hip_y, _ = get_xyv(df, POINTS["left_hip"])
    _, right_hip_y, _ = get_xyv(df, POINTS["right_hip"])

    midhip_y = (left_hip_y + right_hip_y) / 2
    midhip_y_smooth = smooth_signal(midhip_y, window=11, polyorder=2)

    vertical_speed = np.r_[
        np.nan,
        np.diff(midhip_y_smooth) * fps,
    ]

    vertical_speed_abs = np.abs(vertical_speed)

    if search_end_frame is None:
        valid_search = np.full(len(frames), True)
    else:
        valid_search = frames < search_end_frame

    # Manual override
    if manual_sts_start_frame is not None and manual_sts_end_frame is not None:
        sts_start_frame = int(manual_sts_start_frame)
        sts_end_frame = int(manual_sts_end_frame)

        debug_df = pd.DataFrame({
            "Frame": frames,
            "midhip_y": midhip_y,
            "midhip_y_smooth": midhip_y_smooth,
            "vertical_speed": vertical_speed,
            "vertical_speed_abs": vertical_speed_abs,
            "sts_high_threshold": np.nan,
            "sts_boundary_threshold": np.nan,
            "is_sts_region": (frames >= sts_start_frame) & (frames <= sts_end_frame),
            "sts_detection_status": "manual",
        })

        return sts_start_frame, sts_end_frame, debug_df

    y_search = midhip_y_smooth[valid_search]

    if len(y_search) < 3 or np.all(~np.isfinite(y_search)):
        raise ValueError("Could not detect sit-to-stand: no valid midhip_y signal.")

    total_y_change = np.nanmax(y_search) - np.nanmin(y_search)

    if total_y_change < min_y_change_px:
        raise ValueError(
            "Could not detect sit-to-stand: no major midhip_y change was found. "
            "Use manual STS frames if needed."
        )

    speed_for_search = vertical_speed_abs.copy()
    speed_for_search[~valid_search] = np.nan

    if np.all(~np.isfinite(speed_for_search)):
        raise ValueError("Could not detect sit-to-stand: no valid vertical speed.")

    peak_idx = int(np.nanargmax(speed_for_search))
    peak_speed = vertical_speed_abs[peak_idx]

    high_threshold = peak_speed * high_velocity_ratio
    boundary_threshold = peak_speed * boundary_velocity_ratio

    if not np.isfinite(peak_speed) or peak_speed <= 0:
        raise ValueError("Could not detect sit-to-stand: vertical speed is too small.")

    # Start from the strongest vertical movement and expand around it
    sts_start_idx = peak_idx
    while (
        sts_start_idx > 0
        and valid_search[sts_start_idx]
        and vertical_speed_abs[sts_start_idx] >= boundary_threshold
    ):
        sts_start_idx -= 1

    sts_end_idx = peak_idx
    while (
        sts_end_idx < len(frames) - 1
        and valid_search[sts_end_idx]
        and vertical_speed_abs[sts_end_idx] >= boundary_threshold
    ):
        sts_end_idx += 1

    # Safety: require minimum duration
    min_frames = max(1, int(min_duration_sec * fps))
    if sts_end_idx - sts_start_idx + 1 < min_frames:
        raise ValueError("Detected sit-to-stand duration is too short.")

    sts_start_frame = int(frames[sts_start_idx])
    sts_end_frame = int(frames[sts_end_idx])

    debug_df = pd.DataFrame({
        "Frame": frames,
        "midhip_y": midhip_y,
        "midhip_y_smooth": midhip_y_smooth,
        "vertical_speed": vertical_speed,
        "vertical_speed_abs": vertical_speed_abs,
        "sts_high_threshold": high_threshold,
        "sts_boundary_threshold": boundary_threshold,
        "is_sts_region": (frames >= sts_start_frame) & (frames <= sts_end_frame),
        "sts_detection_status": "automatic_midhip_y_change",
    })

    return sts_start_frame, sts_end_frame, debug_df

# ============================================================
# Segment resolution
# ============================================================

def resolve_gait_segments(
    df,
    fps,
    manual_lateral_start_frame=None,
    manual_turn_start_frame=None,
    manual_turn_end_frame=None,
    manual_frontal_end_frame=None,
    run_sit_to_stand=False,
    manual_sts_start_frame=None,
    manual_sts_end_frame=None,
):
    """
    Estimate all main protocol segments and apply manual overrides.

    Detection order:
        1. Detect turning region.
        2. Detect sit-to-stand before the turn.
        3. Detect walking start/stop after sit-to-stand.
        4. Define lateral, turning, and frontal phases.

    Returns:
        segmentation_result, turn_debug_df, foot_motion_debug_df, sts_debug_df
    """

    frames = _frames(df)

    first_frame = int(frames[0])
    last_frame = int(frames[-1])

    # ------------------------------------------------------------
    # 1. Detect turn / cornering region
    # ------------------------------------------------------------
    transition_frame, auto_turn_start, auto_turn_end, turn_debug_df = estimate_turn_region(
        df
    )

    if manual_turn_start_frame is not None and manual_turn_end_frame is not None:
        turn_start = int(manual_turn_start_frame)
        turn_end = int(manual_turn_end_frame)
        turn_source = "manual"
    else:
        turn_start = int(auto_turn_start)
        turn_end = int(auto_turn_end)
        turn_source = "automatic"

    # ------------------------------------------------------------
    # 2. Detect sit-to-stand before walking/turning
    # ------------------------------------------------------------
    sts_debug_df = None
    sit_to_stand_start = None
    sit_to_stand_end = None
    sts_source = "not_run"

    if run_sit_to_stand:
        sit_to_stand_start, sit_to_stand_end, sts_debug_df = estimate_sit_to_stand_region(
            df,
            fps=fps,
            search_end_frame=turn_start,
            manual_sts_start_frame=manual_sts_start_frame,
            manual_sts_end_frame=manual_sts_end_frame,
        )

        if manual_sts_start_frame is not None or manual_sts_end_frame is not None:
            sts_source = "manual"
        else:
            sts_source = "automatic"

    # ------------------------------------------------------------
    # 3. Detect walking only after sit-to-stand
    # ------------------------------------------------------------
    if sit_to_stand_end is not None:
        walking_search_start = int(sit_to_stand_end + 1)
    else:
        walking_search_start = first_frame

    walking_start, walking_stop, foot_motion_debug_df = estimate_walking_start_stop_from_feet(
        df,
        fps=fps,
        search_start_frame=walking_search_start,
    )

    # ------------------------------------------------------------
    # 4. Apply manual walking overrides
    # ------------------------------------------------------------
    if manual_lateral_start_frame is not None:
        lateral_start = int(manual_lateral_start_frame)
        lateral_start_source = "manual"
    else:
        lateral_start = int(walking_start)
        lateral_start_source = "automatic_walking_start"

    lateral_end = int(turn_start - 1)
    frontal_start = int(turn_end + 1)

    if manual_frontal_end_frame is not None:
        frontal_end = int(manual_frontal_end_frame)
        frontal_end_source = "manual"
    else:
        frontal_end = int(walking_stop)
        frontal_end_source = "automatic_walking_stop"

    # ------------------------------------------------------------
    # 5. Safety checks
    # ------------------------------------------------------------
    if lateral_start >= lateral_end:
        raise ValueError(
            "Invalid lateral segment: "
            f"lateral_start={lateral_start}, lateral_end={lateral_end}. "
            "Set MANUAL_LATERAL_START_FRAME or check sit-to-stand/walking detection."
        )

    if frontal_start >= frontal_end:
        raise ValueError(
            "Invalid frontal segment: "
            f"frontal_start={frontal_start}, frontal_end={frontal_end}. "
            "Set MANUAL_FRONTAL_END_FRAME or check walking-stop detection."
        )

    # ------------------------------------------------------------
    # 6. Store final segmentation result
    # ------------------------------------------------------------
    segmentation_result = {
        "fps": fps,
        "first_frame": first_frame,
        "last_frame": last_frame,

        "transition_frame_peak": transition_frame,

        "auto_turn_start_frame": auto_turn_start,
        "auto_turn_end_frame": auto_turn_end,

        "turn_start": turn_start,
        "turn_end": turn_end,
        "turn_source": turn_source,

        "walking_start_frame": walking_start,
        "walking_stop_frame": walking_stop,

        "lateral_start": lateral_start,
        "lateral_end": lateral_end,
        "lateral_start_source": lateral_start_source,

        "frontal_start": frontal_start,
        "frontal_end": frontal_end,
        "frontal_end_source": frontal_end_source,

        "cornering_time_sec": (turn_end - turn_start + 1) / fps,

        "lateral_range": f"{lateral_start}-{lateral_end}",
        "turning_range": f"{turn_start}-{turn_end}",
        "frontal_range": f"{frontal_start}-{frontal_end}",

        "sit_to_stand_start": sit_to_stand_start,
        "sit_to_stand_end": sit_to_stand_end,
        "sit_to_stand_source": sts_source,
    }

    return segmentation_result, turn_debug_df, foot_motion_debug_df, sts_debug_df

# ============================================================
# Frame-level phase labels for plotting/debugging
# ============================================================

def create_phase_dataframe(df, segmentation_result):
    """
    Create one row per frame with the assigned protocol phase.

    Useful for plotting the full signals with colored/labelled regions.
    """

    frames = _frames(df)

    lateral_start = segmentation_result["lateral_start"]
    lateral_end = segmentation_result["lateral_end"]
    turn_start = segmentation_result["turn_start"]
    turn_end = segmentation_result["turn_end"]
    frontal_start = segmentation_result["frontal_start"]
    frontal_end = segmentation_result["frontal_end"]
    sts_start = segmentation_result.get("sit_to_stand_start")
    sts_end = segmentation_result.get("sit_to_stand_end")

    phase = []

    for frame in frames:
        if sts_start is not None and sts_end is not None and sts_start <= frame <= sts_end:
            phase.append("sit_to_stand")
        elif lateral_start <= frame <= lateral_end:
            phase.append("lateral")
        elif turn_start <= frame <= turn_end:
            phase.append("turning")
        elif frontal_start <= frame <= frontal_end:
            phase.append("frontal")
        else:
            phase.append("outside")

    return pd.DataFrame({
        "Frame": frames,
        "phase": phase,
        "is_lateral": (frames >= lateral_start) & (frames <= lateral_end),
        "is_turning": (frames >= turn_start) & (frames <= turn_end),
        "is_frontal": (frames >= frontal_start) & (frames <= frontal_end),
        "is_sit_to_stand": (
            (frames >= sts_start) & (frames <= sts_end)
            if sts_start is not None and sts_end is not None
            else np.full(len(frames), False)
        ),
    })
