import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from helper import (
    POINTS,
    get_xyv,
    distance_2d,
    mean_visibility,
    smooth_signal,
    safe_nanmean,
    safe_nanmax,
    safe_nanmin,
    safe_nanvar,
    safe_nanstd,
    safe_range,
    derivative_abs_per_sec,
    acceleration_abs_per_sec2,
    angle_2d_from_points,
    spine_angle_2d,
)

from features_registry import (
    FEATURE_REGISTRY,
    feature_registry_dataframe,
    feature_values_dataframe,
    default_feature_values,
)

# ============================================================
# Small feature helpers
# ============================================================

def _empty_signal_dataframe():
    return pd.DataFrame(columns=[
        "Frame",
        "segment",
        "step_signal_raw",
        "body_scale",
        "step_signal_smooth",
        "is_step_event",
        "step_event_source",
    ])


def _select_segment(df, start_frame, end_frame):
    return (
        df[
            (df["Frame"] >= start_frame)
            & (df["Frame"] <= end_frame)
        ]
        .copy()
        .reset_index(drop=True)
    )


# ============================================================
# Step signal selection and step detection
# ============================================================

def get_step_signal_by_name(df, signal_name):
    """
    Return the requested lower-limb distance signal.
    """

    if signal_name == "foot_index_distance":
        return distance_2d(
            df,
            POINTS["left_foot_index"],
            POINTS["right_foot_index"]
        )

    if signal_name == "heel_distance":
        return distance_2d(
            df,
            POINTS["left_heel"],
            POINTS["right_heel"]
        )

    if signal_name == "ankle_distance":
        return distance_2d(
            df,
            POINTS["left_ankle"],
            POINTS["right_ankle"]
        )

    return distance_2d(
        df,
        POINTS["left_ankle"],
        POINTS["right_ankle"]
    )


def choose_step_signal(df, label=None):
    """
    Choose the lower-limb signal used for step counting.

    For frontal view, candidate signals are normalized by body scale.
    """

    candidates = {
        "foot_index_distance": {
            "p1": POINTS["left_foot_index"],
            "p2": POINTS["right_foot_index"],
        },
        "heel_distance": {
            "p1": POINTS["left_heel"],
            "p2": POINTS["right_heel"],
        },
        "ankle_distance": {
            "p1": POINTS["left_ankle"],
            "p2": POINTS["right_ankle"],
        },
    }

    # Body scale for perspective normalization
    shoulder_width = distance_2d(
        df,
        POINTS["left_shoulder"],
        POINTS["right_shoulder"]
    )

    hip_width = distance_2d(
        df,
        POINTS["left_hip"],
        POINTS["right_hip"]
    )

    body_scale = shoulder_width + hip_width
    body_scale = smooth_signal(body_scale, window=11, polyorder=2).copy()
    body_scale[body_scale <= 1e-8] = np.nan

    scored = []

    for name, pts in candidates.items():
        raw_signal = distance_2d(df, pts["p1"], pts["p2"])

        if label == "frontal":
            signal_for_detection = raw_signal / body_scale
        else:
            signal_for_detection = raw_signal

        signal_smooth = smooth_signal(
            signal_for_detection,
            window=11,
            polyorder=2
        )

        visibility = mean_visibility(df, pts["p1"], pts["p2"])
        score = safe_nanstd(signal_smooth) * visibility

        scored.append((score, visibility, name, signal_for_detection))

    scored.sort(reverse=True, key=lambda x: x[0])

    best_score, best_visibility, best_name, best_signal = scored[0]

    return best_name, best_signal, best_visibility


def count_steps(df, fps, label=None):
    """
    Count steps in a segment or detection window.

    Frontal:
        detect peaks from foot_index, heel, and ankle signals,
        then merge close peaks.

    Lateral / turning:
        choose one best signal and detect peaks from it.
    """

    empty_sources = pd.DataFrame(
        columns=["Frame", "step_event_source"]
    )

    if len(df) < 5:
        return 0, np.array([]), np.array([]), "none", np.nan, empty_sources

    frames = df["Frame"].to_numpy(int)

    if label == "frontal":
        shoulder_width = distance_2d(
            df,
            POINTS["left_shoulder"],
            POINTS["right_shoulder"]
        )

        hip_width = distance_2d(
            df,
            POINTS["left_hip"],
            POINTS["right_hip"]
        )

        body_scale = shoulder_width + hip_width
        body_scale = smooth_signal(body_scale, window=11, polyorder=2).copy()
        body_scale[body_scale <= 1e-8] = np.nan

        reference_scale = np.nanmedian(body_scale)

        candidates = {
            "foot_index_distance": (
                POINTS["left_foot_index"],
                POINTS["right_foot_index"]
            ),
            "heel_distance": (
                POINTS["left_heel"],
                POINTS["right_heel"]
            ),
            "ankle_distance": (
                POINTS["left_ankle"],
                POINTS["right_ankle"]
            ),
        }

        all_peak_rows = []

        for signal_name, (p1, p2) in candidates.items():
            raw_signal = distance_2d(df, p1, p2)
            # Why used here also the reference_scale not just the body_scale
            corrected_signal = raw_signal * reference_scale / body_scale

            signal_smooth = smooth_signal(
                corrected_signal,
                window=11,
                polyorder=2
            )

            visibility = mean_visibility(df, p1, p2)

            p5 = np.nanpercentile(signal_smooth, 5)
            p95 = np.nanpercentile(signal_smooth, 95)
            amplitude = p95 - p5

            #This prevents counting tiny repeated bumps as multiple steps
            min_peak_distance = max(1, int(0.30 * fps))
            prominence = max(amplitude * 0.07, 1e-8)
            min_height = p5 + 0.02 * amplitude

            peaks, properties = find_peaks(
                signal_smooth,
                distance=min_peak_distance,
                prominence=prominence,
                height=min_height,
            )

            for i, peak_idx in enumerate(peaks):
                all_peak_rows.append({
                    "Frame": int(frames[peak_idx]),
                    "signal_name": signal_name,
                    "peak_height": float(signal_smooth[peak_idx]),
                    "prominence": float(properties["prominences"][i]),
                    "visibility": float(visibility),
                    "score": float(properties["prominences"][i] * visibility),
                })

        if len(all_peak_rows) == 0:
            return (
                0,
                np.array([]),
                np.array([]),
                "multi_signal_frontal",
                np.nan,
                empty_sources,
            )

        peaks_df = pd.DataFrame(all_peak_rows)
        peaks_df = peaks_df.sort_values("Frame").reset_index(drop=True)
       
        # Merge peaks that happen close together
        merge_window_frames = max(1, int(0.15 * fps))

        final_rows = []
        current_group = [0]

        for i in range(1, len(peaks_df)):
            previous_frame = peaks_df.loc[current_group[-1], "Frame"]
            current_frame = peaks_df.loc[i, "Frame"]

            if current_frame - previous_frame <= merge_window_frames:
                current_group.append(i)
            else:
                group_df = peaks_df.loc[current_group]
                best_idx = group_df["score"].idxmax()
                final_rows.append(peaks_df.loc[best_idx])
                current_group = [i]

        group_df = peaks_df.loc[current_group]
        best_idx = group_df["score"].idxmax()
        final_rows.append(peaks_df.loc[best_idx])

        final_peaks_df = pd.DataFrame(final_rows)
        step_frames = final_peaks_df["Frame"].astype(int).to_numpy()

        step_sources_df = final_peaks_df[["Frame", "signal_name"]].copy()
        step_sources_df = step_sources_df.rename(
            columns={"signal_name": "step_event_source"}
        )

        # Representative signal for generic plotting/debug column
        raw_signal = distance_2d(
            df,
            POINTS["left_heel"],
            POINTS["right_heel"]
        )

        representative_signal = raw_signal * reference_scale / body_scale

        signal_smooth = smooth_signal(
            representative_signal,
            window=11,
            polyorder=2
        )

        visibility = np.nanmean(final_peaks_df["visibility"])

        return (
            len(step_frames),
            step_frames,
            signal_smooth,
            "multi_signal_frontal",
            visibility,
            step_sources_df,
        )

    # Lateral / turning
    signal_name, signal, visibility = choose_step_signal(df, label=label)
    signal_smooth = smooth_signal(signal, window=11, polyorder=2)

    p5 = np.nanpercentile(signal_smooth, 5)
    p95 = np.nanpercentile(signal_smooth, 95)
    amplitude = p95 - p5

    min_peak_distance = max(1, int(0.35 * fps))
    prominence = max(amplitude * 0.08, 1e-8)
    min_height = p5 + 0.02 * amplitude

    peaks, properties = find_peaks(
        signal_smooth,
        distance=min_peak_distance,
        prominence=prominence,
        height=min_height,
    )

    step_frames = frames[peaks]

    step_sources_df = pd.DataFrame({
        "Frame": step_frames,
        "step_event_source": signal_name,
    })

    return (
        len(step_frames),
        step_frames,
        signal_smooth,
        signal_name,
        visibility,
        step_sources_df,
    )


# ============================================================
# Feature computation blocks
# ============================================================

def compute_basic_coordinates(sub):
    left_ankle_x, left_ankle_y, _ = get_xyv(sub, POINTS["left_ankle"])
    right_ankle_x, right_ankle_y, _ = get_xyv(sub, POINTS["right_ankle"])

    left_knee_x, left_knee_y, _ = get_xyv(sub, POINTS["left_knee"])
    right_knee_x, right_knee_y, _ = get_xyv(sub, POINTS["right_knee"])

    left_hip_x, left_hip_y, _ = get_xyv(sub, POINTS["left_hip"])
    right_hip_x, right_hip_y, _ = get_xyv(sub, POINTS["right_hip"])

    ankle_x_distance = np.abs(right_ankle_x - left_ankle_x)
    knee_x_distance = np.abs(right_knee_x - left_knee_x)

    midhip_x = (left_hip_x + right_hip_x) / 2
    midhip_y = (left_hip_y + right_hip_y) / 2

    return {
        "left_ankle_x": left_ankle_x,
        "left_ankle_y": left_ankle_y,
        "right_ankle_x": right_ankle_x,
        "right_ankle_y": right_ankle_y,
        "left_knee_x": left_knee_x,
        "left_knee_y": left_knee_y,
        "right_knee_x": right_knee_x,
        "right_knee_y": right_knee_y,
        "left_hip_x": left_hip_x,
        "left_hip_y": left_hip_y,
        "right_hip_x": right_hip_x,
        "right_hip_y": right_hip_y,
        "ankle_x_distance": ankle_x_distance,
        "knee_x_distance": knee_x_distance,
        "midhip_x": midhip_x,
        "midhip_y": midhip_y,
    }


def compute_spatiotemporal_features(
    label,
    gait_time_sec,
    walking_distance_m,
    step_frames,
    fps,
    step_signal_used,
    step_signal_visibility_mean,
):
    n_steps = len(step_frames)

    features = {
        "A01_gait_speed_m_s": np.nan,
        "A02_number_of_steps": n_steps,
        "A03_step_length_m": np.nan,
        "A04_stride_length_m": np.nan,
        "A05_cadence_steps_per_min": np.nan,
        "mean_step_time_sec": np.nan,
        "step_time_std_sec": np.nan,
        "A06_step_time_cv_percent": np.nan,
        "step_time_values_sec": "",
        "n_step_intervals": 0,
        "A07_step_signal_used": step_signal_used,
        "step_signal_visibility_mean": step_signal_visibility_mean,
    }

    if label == "sit_to_stand":
        return features

    if gait_time_sec > 0:
        features["A05_cadence_steps_per_min"] = (n_steps / gait_time_sec) * 60.0

        if np.isfinite(walking_distance_m):
            features["A01_gait_speed_m_s"] = walking_distance_m / gait_time_sec

    if n_steps > 0 and np.isfinite(walking_distance_m):
        features["A03_step_length_m"] = walking_distance_m / n_steps
        features["A04_stride_length_m"] = 2 * features["A03_step_length_m"]

    if n_steps >= 2:
        step_time_values = np.diff(step_frames) / fps

        mean_step_time_sec = np.mean(step_time_values)

        if len(step_time_values) >= 2:
            step_time_std_sec = np.std(step_time_values, ddof=1)
        else:
            step_time_std_sec = 0.0

        if mean_step_time_sec > 0:
            step_time_cv_percent = (
                step_time_std_sec / mean_step_time_sec
            ) * 100.0
        else:
            step_time_cv_percent = np.nan

        features["mean_step_time_sec"] = mean_step_time_sec
        features["step_time_std_sec"] = step_time_std_sec
        features["A06_step_time_cv_percent"] = step_time_cv_percent
        features["step_time_values_sec"] = ";".join(
            f"{value:.4f}" for value in step_time_values
        )
        features["n_step_intervals"] = len(step_time_values)

    return features


def compute_body_position_features(coords):
    return {
        "P22_midhip_x_mean": safe_nanmean(coords["midhip_x"]),
        "P23_midhip_x_variance": safe_nanvar(coords["midhip_x"]),
        "P24_midhip_y_mean": safe_nanmean(coords["midhip_y"]),
        "P25_midhip_y_variance": safe_nanvar(coords["midhip_y"]),
    }


def compute_step_width_features(label, coords):
    if label == "frontal":
        return {
            "P04_step_width_norm": safe_nanmean(coords["ankle_x_distance"])
        }

    return {
        "P04_step_width_norm": np.nan
    }


def compute_distance_features(label, coords, fps):
    n = len(coords["ankle_x_distance"])

    features = {
        "P05_ankle_distance_acceleration_max_norm_per_sec2": np.nan,
        "P06_ankle_distance_mean_norm": np.nan,
        "P07_ankle_distance_max_norm": np.nan,
        "P08_ankle_distance_variance_norm": np.nan,
        "P09_ankle_distance_trajectory_max_norm_per_sec": np.nan,
        "P10_ankle_distance_trajectory_mean_norm_per_sec": np.nan,
        "P11_knee_distance_acceleration_max_norm_per_sec2": np.nan,
        "P12_knee_distance_mean_norm": np.nan,
        "P13_knee_distance_max_norm": np.nan,
        "P14_knee_distance_variance_norm": np.nan,
        "P15_knee_distance_trajectory_max_norm_per_sec": np.nan,
        "P16_knee_distance_trajectory_mean_norm_per_sec": np.nan,

        # support signals
        "_ankle_distance_for_paper": np.full(n, np.nan),
        "_ankle_distance_smooth": np.full(n, np.nan),
        "_ankle_distance_trajectory": np.full(n, np.nan),
        "_ankle_distance_acceleration": np.full(n, np.nan),
        "_knee_distance_for_paper": np.full(n, np.nan),
        "_knee_distance_smooth": np.full(n, np.nan),
        "_knee_distance_trajectory": np.full(n, np.nan),
        "_knee_distance_acceleration": np.full(n, np.nan),
    }

    if label != "lateral":
        return features

    ankle_distance_for_paper = coords["ankle_x_distance"]
    ankle_distance_smooth = smooth_signal(
        ankle_distance_for_paper,
        window=11,
        polyorder=2
    )
    ankle_distance_trajectory = derivative_abs_per_sec(
        ankle_distance_smooth,
        fps
    )
    ankle_distance_acceleration = acceleration_abs_per_sec2(
        ankle_distance_trajectory,
        fps
    )

    knee_distance_for_paper = coords["knee_x_distance"]
    knee_distance_smooth = smooth_signal(
        knee_distance_for_paper,
        window=11,
        polyorder=2
    )
    knee_distance_trajectory = derivative_abs_per_sec(
        knee_distance_smooth,
        fps
    )
    knee_distance_acceleration = acceleration_abs_per_sec2(
        knee_distance_trajectory,
        fps
    )

    features.update({
        "P05_ankle_distance_acceleration_max_norm_per_sec2": safe_nanmax(
            ankle_distance_acceleration
        ),
        "P06_ankle_distance_mean_norm": safe_nanmean(
            ankle_distance_for_paper
        ),
        "P07_ankle_distance_max_norm": safe_nanmax(
            ankle_distance_for_paper
        ),
        "P08_ankle_distance_variance_norm": safe_nanvar(
            ankle_distance_for_paper
        ),
        "P09_ankle_distance_trajectory_max_norm_per_sec": safe_nanmax(
            ankle_distance_trajectory
        ),
        "P10_ankle_distance_trajectory_mean_norm_per_sec": safe_nanmean(
            ankle_distance_trajectory
        ),

        "P11_knee_distance_acceleration_max_norm_per_sec2": safe_nanmax(
            knee_distance_acceleration
        ),
        "P12_knee_distance_mean_norm": safe_nanmean(
            knee_distance_for_paper
        ),
        "P13_knee_distance_max_norm": safe_nanmax(
            knee_distance_for_paper
        ),
        "P14_knee_distance_variance_norm": safe_nanvar(
            knee_distance_for_paper
        ),
        "P15_knee_distance_trajectory_max_norm_per_sec": safe_nanmax(
            knee_distance_trajectory
        ),
        "P16_knee_distance_trajectory_mean_norm_per_sec": safe_nanmean(
            knee_distance_trajectory
        ),

        "_ankle_distance_for_paper": ankle_distance_for_paper,
        "_ankle_distance_smooth": ankle_distance_smooth,
        "_ankle_distance_trajectory": ankle_distance_trajectory,
        "_ankle_distance_acceleration": ankle_distance_acceleration,
        "_knee_distance_for_paper": knee_distance_for_paper,
        "_knee_distance_smooth": knee_distance_smooth,
        "_knee_distance_trajectory": knee_distance_trajectory,
        "_knee_distance_acceleration": knee_distance_acceleration,
    })

    return features


def compute_joint_angle_features(sub, label):
    n = len(sub)

    features = {
        "P01_ankle_joint_rom_deg": np.nan,
        "P02_ankle_joint_angle_max_deg": np.nan,
        "P03_ankle_joint_angle_min_deg": np.nan,
        "P17_hip_joint_angle_mean_deg": np.nan,
        "P18_hip_joint_angle_variance_deg": np.nan,
        "P19_knee_joint_rom_deg": np.nan,
        "P20_knee_joint_angle_max_deg": np.nan,
        "P21_knee_joint_angle_min_deg": np.nan,
        "left_knee_rom_deg": np.nan,
        "right_knee_rom_deg": np.nan,
        "A10_mean_knee_rom_deg": np.nan,
        "A17_ankle_angle_mean_deg": np.nan,
        "A18_ankle_angle_variance_deg": np.nan,
        "A19_knee_angle_mean_deg": np.nan,
        "A20_knee_angle_variance_deg": np.nan,
        "A21_hip_angle_rom_deg": np.nan,
        "A22_hip_angle_max_deg": np.nan,
        "A23_hip_angle_min_deg": np.nan,

        # support signals
        "_left_knee_angle": np.full(n, np.nan),
        "_right_knee_angle": np.full(n, np.nan),
        "_left_hip_angle": np.full(n, np.nan),
        "_right_hip_angle": np.full(n, np.nan),
        "_left_ankle_angle": np.full(n, np.nan),
        "_right_ankle_angle": np.full(n, np.nan),
    }

    if label != "lateral":
        return features

    left_knee_angle = angle_2d_from_points(
        sub,
        POINTS["left_hip"],
        POINTS["left_knee"],
        POINTS["left_ankle"]
    )

    right_knee_angle = angle_2d_from_points(
        sub,
        POINTS["right_hip"],
        POINTS["right_knee"],
        POINTS["right_ankle"]
    )

    left_hip_angle = angle_2d_from_points(
        sub,
        POINTS["left_shoulder"],
        POINTS["left_hip"],
        POINTS["left_knee"]
    )

    right_hip_angle = angle_2d_from_points(
        sub,
        POINTS["right_shoulder"],
        POINTS["right_hip"],
        POINTS["right_knee"]
    )

    left_ankle_angle = angle_2d_from_points(
        sub,
        POINTS["left_knee"],
        POINTS["left_ankle"],
        POINTS["left_foot_index"]
    )

    right_ankle_angle = angle_2d_from_points(
        sub,
        POINTS["right_knee"],
        POINTS["right_ankle"],
        POINTS["right_foot_index"]
    )

    knee_angle_all = np.r_[left_knee_angle, right_knee_angle]
    hip_angle_all = np.r_[left_hip_angle, right_hip_angle]
    ankle_angle_all = np.r_[left_ankle_angle, right_ankle_angle]

    left_knee_rom_deg = safe_range(left_knee_angle)
    right_knee_rom_deg = safe_range(right_knee_angle)

    features.update({
        "P01_ankle_joint_rom_deg": safe_range(ankle_angle_all),
        "P02_ankle_joint_angle_max_deg": safe_nanmax(ankle_angle_all),
        "P03_ankle_joint_angle_min_deg": safe_nanmin(ankle_angle_all),
        "P17_hip_joint_angle_mean_deg": safe_nanmean(hip_angle_all),
        "P18_hip_joint_angle_variance_deg": safe_nanvar(hip_angle_all),
        "P19_knee_joint_rom_deg": safe_range(knee_angle_all),
        "P20_knee_joint_angle_max_deg": safe_nanmax(knee_angle_all),
        "P21_knee_joint_angle_min_deg": safe_nanmin(knee_angle_all),
        
        "left_knee_rom_deg": left_knee_rom_deg,
        "right_knee_rom_deg": right_knee_rom_deg,
        "A10_mean_knee_rom_deg": safe_nanmean([
            left_knee_rom_deg,
            right_knee_rom_deg,
        ]),
        "A17_ankle_angle_mean_deg": safe_nanmean(ankle_angle_all),
        "A18_ankle_angle_variance_deg": safe_nanvar(ankle_angle_all),

        "A19_knee_angle_mean_deg": safe_nanmean(knee_angle_all),
        "A20_knee_angle_variance_deg": safe_nanvar(knee_angle_all),

        "A21_hip_angle_rom_deg": safe_range(hip_angle_all),
        "A22_hip_angle_max_deg": safe_nanmax(hip_angle_all),
        "A23_hip_angle_min_deg": safe_nanmin(hip_angle_all),

        "_left_knee_angle": left_knee_angle,
        "_right_knee_angle": right_knee_angle,
        "_left_hip_angle": left_hip_angle,
        "_right_hip_angle": right_hip_angle,
        "_left_ankle_angle": left_ankle_angle,
        "_right_ankle_angle": right_ankle_angle,

    })

    return features


def compute_spine_walking_features(sub, label):
    spine_angle = spine_angle_2d(sub)

    features = {
        "A09_spine_angle_lateral_mean_deg": np.nan,
        "spine_angle_lateral_sd_deg": np.nan,
        "spine_angle_lateral_max_deg": np.nan,
        "spine_angle_frontal_mean_deg": np.nan,
        "spine_angle_frontal_sd_deg": np.nan,
        "spine_angle_frontal_max_deg": np.nan,
        "_spine_angle": spine_angle,
    }

    if label == "lateral":
        features["A09_spine_angle_lateral_mean_deg"] = safe_nanmean(spine_angle)
        features["spine_angle_lateral_sd_deg"] = safe_nanstd(spine_angle)
        features["spine_angle_lateral_max_deg"] = safe_nanmax(spine_angle)

    elif label == "frontal":
        features["spine_angle_frontal_mean_deg"] = safe_nanmean(spine_angle)
        features["spine_angle_frontal_sd_deg"] = safe_nanstd(spine_angle)
        features["spine_angle_frontal_max_deg"] = safe_nanmax(spine_angle)

    return features


def compute_sit_to_stand_features(
    df,
    sub,
    label,
    fps,
    start_frame,
    end_frame,
    sitting_window_sec=1.0,
):
    n = len(sub)

    features = {
        "P26_spine_angle_in_stand_up_max_deg": np.nan,
        "P27_spine_angle_in_stand_up_mean_deg": np.nan,
        "P28_spine_angle_in_sitting_mean_deg": np.nan,
        "A11_sit_to_stand_time_sec": np.nan,
        "A12_spine_angle_standup_sd_deg": np.nan,
        "A13_spine_angle_standup_rom_deg": np.nan,
        "A14_spine_angle_velocity_max_deg_per_sec": np.nan,
        "A15_midhip_y_displacement_norm": np.nan,
        "A16_midhip_vertical_speed_max_norm_per_sec": np.nan,

        # support signals
        "_spine_angle_standup_smooth": np.full(n, np.nan),
        "_spine_angle_velocity": np.full(n, np.nan),
        "_midhip_y_smooth": np.full(n, np.nan),
        "_midhip_upward_speed": np.full(n, np.nan),
    }

    if label != "sit_to_stand":
        return features

    frames = sub["Frame"].to_numpy(int)

    spine_angle_standup = spine_angle_2d(sub)
    spine_angle_standup_smooth = smooth_signal(
        spine_angle_standup,
        window=11,
        polyorder=2
    )

    spine_angle_velocity = derivative_abs_per_sec(
        spine_angle_standup_smooth,
        fps
    )

    sitting_window_frames = int(sitting_window_sec * fps)

    sitting_start_frame = max(
        int(df["Frame"].min()),
        start_frame - sitting_window_frames
    )

    sitting_end_frame = start_frame - 1

    sitting_sub = (
        df[
            (df["Frame"] >= sitting_start_frame)
            & (df["Frame"] <= sitting_end_frame)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if len(sitting_sub) >= 2:
        sitting_spine_angle = spine_angle_2d(sitting_sub)
        spine_angle_in_sitting_deg = safe_nanmean(sitting_spine_angle)
    else:
        spine_angle_in_sitting_deg = np.nan


    _, left_hip_y, _ = get_xyv(sub, POINTS["left_hip"])
    _, right_hip_y, _ = get_xyv(sub, POINTS["right_hip"])

    midhip_y = (left_hip_y + right_hip_y) / 2
    midhip_y_smooth = smooth_signal(midhip_y, window=11, polyorder=2)

    midhip_vertical_speed = np.r_[
        np.nan,
        np.abs(np.diff(midhip_y_smooth)) * fps
    ]

    midhip_y_displacement_norm = abs(midhip_y_smooth[-1] - midhip_y_smooth[0])

    features.update({
        "P26_spine_angle_in_stand_up_max_deg": safe_nanmax(
            spine_angle_standup_smooth
        ),
        "P27_spine_angle_in_stand_up_mean_deg": safe_nanmean(
            spine_angle_standup_smooth
        ),
        "P28_spine_angle_in_sitting_mean_deg": spine_angle_in_sitting_deg,
        "A11_sit_to_stand_time_sec": (frames[-1] - frames[0] + 1) / fps,
        "A12_spine_angle_standup_sd_deg": safe_nanstd(
            spine_angle_standup_smooth
        ),
        "A13_spine_angle_standup_rom_deg": safe_range(
            spine_angle_standup_smooth
        ),
        "A14_spine_angle_velocity_max_deg_per_sec": safe_nanmax(
            spine_angle_velocity
        ),
        "A15_midhip_y_displacement_norm": midhip_y_displacement_norm,
        "A16_midhip_vertical_speed_max_norm_per_sec": safe_nanmax(
            midhip_vertical_speed
        ),
        "_spine_angle_standup_smooth": spine_angle_standup_smooth,
        "_spine_angle_velocity": spine_angle_velocity,
        "_midhip_y_smooth": midhip_y_smooth,
        "_midhip_upward_speed": midhip_vertical_speed,
    })

    return features


def compute_lateral_step_stride_proxy(label, sub, coords, step_frames):
    features = {
        "A08_step_length_measured_norm_mean": np.nan,
        "step_length_measured_norm_std": np.nan,
        "step_length_measured_norm_values": "",
        "stride_length_measured_norm_mean": np.nan,
        "stride_length_measured_norm_std": np.nan,
        "stride_length_measured_norm_values": "",
        "n_stride_estimates": 0,
        "measured_length_method": "not_computed_for_non_lateral",
    }

    if label != "lateral" or len(step_frames) == 0:
        return features

    ankle_x_distance = coords["ankle_x_distance"]
    midhip_x = coords["midhip_x"]

    step_event_mask = sub["Frame"].isin(step_frames).to_numpy()
    step_length_values = ankle_x_distance[step_event_mask]

    features["A08_step_length_measured_norm_mean"] = safe_nanmean(
        step_length_values
    )
    features["step_length_measured_norm_std"] = safe_nanstd(
        step_length_values
    )
    features["step_length_measured_norm_values"] = ";".join(
        f"{v:.6f}" for v in step_length_values if np.isfinite(v)
    )

    events = (
        sub[sub["Frame"].isin(step_frames)]
        .copy()
        .reset_index(drop=True)
    )

    if len(events) < 2:
        features["measured_length_method"] = "ankle_x_lateral"
        return features

    event_left_ankle_x, _, _ = get_xyv(events, POINTS["left_ankle"])
    event_right_ankle_x, _, _ = get_xyv(events, POINTS["right_ankle"])

    n_ref = min(5, len(midhip_x))

    walking_direction = (
        safe_nanmean(midhip_x[-n_ref:])
        - safe_nanmean(midhip_x[:n_ref])
    )

    leading_foot = []
    leading_x = []

    for i in range(len(events)):
        lx = event_left_ankle_x[i]
        rx = event_right_ankle_x[i]

        if not np.isfinite(lx) or not np.isfinite(rx):
            leading_foot.append("")
            leading_x.append(np.nan)
            continue

        if walking_direction >= 0:
            if rx >= lx:
                leading_foot.append("right")
                leading_x.append(rx)
            else:
                leading_foot.append("left")
                leading_x.append(lx)
        else:
            if rx <= lx:
                leading_foot.append("right")
                leading_x.append(rx)
            else:
                leading_foot.append("left")
                leading_x.append(lx)

    leading_x = np.asarray(leading_x, dtype=float)

    stride_lengths = []

    for i in range(len(leading_foot)):
        if leading_foot[i] == "" or not np.isfinite(leading_x[i]):
            continue

        for j in range(i + 1, len(leading_foot)):
            if (
                leading_foot[j] == leading_foot[i]
                and np.isfinite(leading_x[j])
            ):
                stride_lengths.append(abs(leading_x[j] - leading_x[i]))
                break

    stride_lengths = np.asarray(stride_lengths, dtype=float)

    features.update({
        "stride_length_measured_norm_mean": safe_nanmean(stride_lengths),
        "stride_length_measured_norm_std": safe_nanstd(stride_lengths),
        "stride_length_measured_norm_values": ";".join(
            f"{v:.6f}" for v in stride_lengths if np.isfinite(v)
        ),
        "n_stride_estimates": int(np.sum(np.isfinite(stride_lengths))),
        "measured_length_method": "ankle_x_lateral",
    })

    return features

def build_all_step_signals_dataframe(df, label):
    """
    Save ankle, heel, and foot-index distance signals for plotting.
    This is independent from which signal was selected for step detection.
    """

    frames = df["Frame"].to_numpy(int)

    signal_names = [
        "ankle_distance",
        "heel_distance",
        "foot_index_distance",
    ]

    out = pd.DataFrame({
        "Frame": frames,
        "segment": label,
    })

    for signal_name in signal_names:
        raw_signal = get_step_signal_by_name(df, signal_name)
        smooth = smooth_signal(raw_signal, window=11, polyorder=2)

        out[f"{signal_name}_raw"] = raw_signal
        out[f"{signal_name}_smooth"] = smooth
        out[f"{signal_name}_variance"] = safe_nanvar(smooth)

    return out


# ============================================================
# Signal dataframe
# ============================================================

def build_signal_dataframe(
    sub,
    label,
    local_signal_raw,
    body_scale,
    signal_smooth,
    step_frames,
    step_sources_df,
    coords,
    distance_features,
    joint_features,
    spine_features,
    sts_features,
):
    frames = sub["Frame"].to_numpy(int)

    signal_df = pd.DataFrame({
        "Frame": frames,
        "segment": label,

        "step_signal_raw": local_signal_raw,
        "body_scale": body_scale,
        "step_signal_smooth": signal_smooth,
        "is_step_event": False,

        "ankle_x_distance": coords["ankle_x_distance"],
        "knee_x_distance": coords["knee_x_distance"],

        "ankle_distance_for_paper": distance_features["_ankle_distance_for_paper"],
        "ankle_distance_smooth": distance_features["_ankle_distance_smooth"],
        "ankle_distance_trajectory": distance_features["_ankle_distance_trajectory"],
        "ankle_distance_acceleration": distance_features["_ankle_distance_acceleration"],

        "knee_distance_for_paper": distance_features["_knee_distance_for_paper"],
        "knee_distance_smooth": distance_features["_knee_distance_smooth"],
        "knee_distance_trajectory": distance_features["_knee_distance_trajectory"],
        "knee_distance_acceleration": distance_features["_knee_distance_acceleration"],

        "left_ankle_angle_deg": joint_features["_left_ankle_angle"],
        "right_ankle_angle_deg": joint_features["_right_ankle_angle"],
        "left_knee_angle_deg": joint_features["_left_knee_angle"],
        "right_knee_angle_deg": joint_features["_right_knee_angle"],
        "left_hip_angle_deg": joint_features["_left_hip_angle"],
        "right_hip_angle_deg": joint_features["_right_hip_angle"],

        "midhip_x": coords["midhip_x"],
        "midhip_y": coords["midhip_y"],

        "spine_angle_deg": spine_features["_spine_angle"],

        "spine_angle_standup_smooth_deg": sts_features[
            "_spine_angle_standup_smooth"
        ],
        "spine_angle_velocity_deg_per_sec": sts_features[
            "_spine_angle_velocity"
        ],
        "midhip_y_smooth": sts_features["_midhip_y_smooth"],
        "midhip_upward_speed_norm_per_sec": sts_features[
            "_midhip_upward_speed"
        ],
    })

    signal_df = signal_df.merge(
        step_sources_df,
        on="Frame",
        how="left"
    )

    if "step_event_source" not in signal_df.columns:
        signal_df["step_event_source"] = ""

    signal_df["step_event_source"] = signal_df["step_event_source"].fillna("")

    if len(step_frames) > 0:
        signal_df.loc[
            signal_df["Frame"].isin(step_frames),
            "is_step_event"
        ] = True

    return signal_df


# ============================================================
# Main segment-level feature extractor
# ============================================================

def extract_segment_features(
    df,
    fps,
    start_frame,
    end_frame,
    walking_distance_m,
    label,
    detection_context_sec=0.6,
    sitting_window_sec=1.0,
):
    """
    Extract all features for one segment.

    Replaces the old segment_features() function.
    """

    sub = _select_segment(df, start_frame, end_frame)

    if len(sub) < 2:
        result = {
            "segment": label,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "n_frames": len(sub),
            "gait_time_sec": np.nan,
            "walking_distance_m": walking_distance_m,
        }

        result.update(default_feature_values())

        return result, _empty_signal_dataframe()

    frames = sub["Frame"].to_numpy(int)

    segment_start_frame = int(frames[0])
    segment_end_frame = int(frames[-1])
    segment_n_frames = len(sub)

    gait_time_sec = (frames[-1] - frames[0] + 1) / fps

    # ------------------------------------------------------------
    # Step detection
    # ------------------------------------------------------------

    step_frames = np.array([], dtype=int)
    step_signal_used = "none"
    step_signal_visibility_mean = np.nan

    local_signal_raw = np.full(len(sub), np.nan)
    body_scale = np.full(len(sub), np.nan)
    signal_smooth = np.full(len(sub), np.nan)

    step_sources_df = pd.DataFrame({
        "Frame": frames,
        "step_event_source": "",
    })

    if label != "sit_to_stand":
        context_frames = int(detection_context_sec * fps)

        detect_start = max(
            int(df["Frame"].min()),
            start_frame - context_frames
        )

        detect_end = min(
            int(df["Frame"].max()),
            end_frame + context_frames
        )

        detect_sub = _select_segment(df, detect_start, detect_end)

        (
            _,
            all_step_frames,
            _,
            signal_name,
            signal_visibility,
            step_sources_df,
        ) = count_steps(
            detect_sub,
            fps,
            label=label
        )

        step_frames = all_step_frames[
            (all_step_frames >= start_frame)
            & (all_step_frames <= end_frame)
        ]

        step_sources_df = step_sources_df[
            (step_sources_df["Frame"] >= start_frame)
            & (step_sources_df["Frame"] <= end_frame)
        ].copy()

        step_signal_used = signal_name
        step_signal_visibility_mean = signal_visibility

        if signal_name == "multi_signal_frontal":
            local_plot_signal_name = "heel_distance"
        else:
            local_plot_signal_name = signal_name

        local_signal_raw = get_step_signal_by_name(
            sub,
            local_plot_signal_name
        )

        if label == "frontal":
            shoulder_width = distance_2d(
                sub,
                POINTS["left_shoulder"],
                POINTS["right_shoulder"]
            )

            hip_width = distance_2d(
                sub,
                POINTS["left_hip"],
                POINTS["right_hip"]
            )

            body_scale = shoulder_width + hip_width
            body_scale = smooth_signal(
                body_scale,
                window=11,
                polyorder=2
            ).copy()

            body_scale[body_scale <= 1e-8] = np.nan
            reference_scale = np.nanmedian(body_scale)

            local_signal = (
                local_signal_raw
                * reference_scale
                / body_scale
            )

        else:
            body_scale = np.full(len(sub), np.nan)
            local_signal = local_signal_raw

        signal_smooth = smooth_signal(
            local_signal,
            window=11,
            polyorder=2
        )

    # ------------------------------------------------------------
    # Compute feature groups
    # ------------------------------------------------------------

    coords = compute_basic_coordinates(sub)

    result = {
        "segment": label,
        "start_frame": segment_start_frame,
        "end_frame": segment_end_frame,
        "n_frames": segment_n_frames,
        "gait_time_sec": gait_time_sec,
        "walking_distance_m": walking_distance_m,
    }

    result.update(default_feature_values())

    spatiotemporal_features = compute_spatiotemporal_features(
        label=label,
        gait_time_sec=gait_time_sec,
        walking_distance_m=walking_distance_m,
        step_frames=step_frames,
        fps=fps,
        step_signal_used=step_signal_used,
        step_signal_visibility_mean=step_signal_visibility_mean,
    )

    body_position_features = compute_body_position_features(coords)

    step_width_features = compute_step_width_features(
        label=label,
        coords=coords,
    )

    distance_features = compute_distance_features(
        label=label,
        coords=coords,
        fps=fps,
    )

    joint_features = compute_joint_angle_features(
        sub=sub,
        label=label,
    )

    spine_features = compute_spine_walking_features(
        sub=sub,
        label=label,
    )

    sts_features = compute_sit_to_stand_features(
        df=df,
        sub=sub,
        label=label,
        fps=fps,
        start_frame=start_frame,
        end_frame=end_frame,
        sitting_window_sec=sitting_window_sec,
    )

    step_stride_proxy_features = compute_lateral_step_stride_proxy(
        label=label,
        sub=sub,
        coords=coords,
        step_frames=step_frames,
    )

    result.update(spatiotemporal_features)
    result.update(body_position_features)
    result.update(step_width_features)
    result.update({
        key: value
        for key, value in distance_features.items()
        if not key.startswith("_")
    })
    result.update({
        key: value
        for key, value in joint_features.items()
        if not key.startswith("_")
    })
    result.update({
        key: value
        for key, value in spine_features.items()
        if not key.startswith("_")
    })
    result.update({
        key: value
        for key, value in sts_features.items()
        if not key.startswith("_")
    })
    result.update(step_stride_proxy_features)

    signal_df = build_signal_dataframe(
        sub=sub,
        label=label,
        local_signal_raw=local_signal_raw,
        body_scale=body_scale,
        signal_smooth=signal_smooth,
        step_frames=step_frames,
        step_sources_df=step_sources_df,
        coords=coords,
        distance_features=distance_features,
        joint_features=joint_features,
        spine_features=spine_features,
        sts_features=sts_features,
    )

    return result, signal_df


# ============================================================
# Total summary features
# ============================================================

def compute_total_features(
    df,
    lateral_result,
    turning_result,
    frontal_result,
    lateral_distance_m,
    frontal_distance_m,
    turn_start,
    turn_end,
    fps,
):
    first_frame = int(df["Frame"].min())
    last_frame = int(df["Frame"].max())

    cornering_time_sec = (turn_end - turn_start + 1) / fps

    total_distance = lateral_distance_m + frontal_distance_m

    total_time = (
        lateral_result["gait_time_sec"]
        + cornering_time_sec
        + frontal_result["gait_time_sec"]
    )

    total_steps = (
        lateral_result["A02_number_of_steps"]
        + turning_result["A02_number_of_steps"]
        + frontal_result["A02_number_of_steps"]
    )

    total_speed = total_distance / total_time if total_time > 0 else np.nan

    if total_steps > 0:
        total_step_length = total_distance / total_steps
        total_stride_length = 2 * total_step_length
    else:
        total_step_length = np.nan
        total_stride_length = np.nan

    total_result = {
        "segment": "total",
        "start_frame": first_frame,
        "end_frame": last_frame,
        "n_frames": len(df),
        "gait_time_sec": total_time,
        "cornering_time_sec": cornering_time_sec,
        "walking_distance_m": total_distance,
        "A01_gait_speed_m_s": total_speed,
        "A02_number_of_steps": total_steps,
        "turning_steps": turning_result["A02_number_of_steps"],
        "A03_step_length_m": total_step_length,
        "A04_stride_length_m": total_stride_length,
        "A07_step_signal_used": "lateral+frontal",
        "step_signal_visibility_mean": np.nan,
    }

    return total_result