"""
registry.py

Feature registry for the gait feature extraction pipeline.

This file defines:
    - the list of implemented features
    - their output column names
    - their clinical/technical group
    - the phase where each feature is computed
    - whether each feature comes from the paper or was added during the project
"""

import numpy as np
import pandas as pd


# ============================================================
# FEATURE REGISTRY
# ============================================================

FEATURE_REGISTRY = {
    # ------------------------------------------------------------
    # Paper features
    # ------------------------------------------------------------
    "P01": {
        "column": "P01_ankle_joint_rom_deg",
        "name": "Ankle joint ROM",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P02": {
        "column": "P02_ankle_joint_angle_max_deg",
        "name": "Ankle joint angle max",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P03": {
        "column": "P03_ankle_joint_angle_min_deg",
        "name": "Ankle joint angle min",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P04": {
        "column": "P04_step_width_norm",
        "name": "Step width",
        "group": "distance",
        "phase": "frontal",
        "source": "paper",
    },
    "P05": {
        "column": "P05_ankle_distance_acceleration_max_norm_per_sec2",
        "name": "Bilateral ankle distance acceleration max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P06": {
        "column": "P06_ankle_distance_mean_norm",
        "name": "Bilateral ankle distance mean",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P07": {
        "column": "P07_ankle_distance_max_norm",
        "name": "Bilateral ankle distance max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P08": {
        "column": "P08_ankle_distance_variance_norm",
        "name": "Bilateral ankle distance variance",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P09": {
        "column": "P09_ankle_distance_trajectory_max_norm_per_sec",
        "name": "Trajectory of ankle distance max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P10": {
        "column": "P10_ankle_distance_trajectory_mean_norm_per_sec",
        "name": "Trajectory of ankle distance mean",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P11": {
        "column": "P11_knee_distance_acceleration_max_norm_per_sec2",
        "name": "Knee distance acceleration max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P12": {
        "column": "P12_knee_distance_mean_norm",
        "name": "Knee distance mean",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P13": {
        "column": "P13_knee_distance_max_norm",
        "name": "Knee distance max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P14": {
        "column": "P14_knee_distance_variance_norm",
        "name": "Knee distance variance",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P15": {
        "column": "P15_knee_distance_trajectory_max_norm_per_sec",
        "name": "Trajectory of knee distance max",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P16": {
        "column": "P16_knee_distance_trajectory_mean_norm_per_sec",
        "name": "Trajectory of knee distance mean",
        "group": "distance",
        "phase": "lateral",
        "source": "paper",
    },
    "P17": {
        "column": "P17_hip_joint_angle_mean_deg",
        "name": "Hip joint angle mean",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P18": {
        "column": "P18_hip_joint_angle_variance_deg",
        "name": "Hip joint angle variance",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P19": {
        "column": "P19_knee_joint_rom_deg",
        "name": "Knee joint ROM",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P20": {
        "column": "P20_knee_joint_angle_max_deg",
        "name": "Knee joint angle max",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P21": {
        "column": "P21_knee_joint_angle_min_deg",
        "name": "Knee joint angle min",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "paper",
    },
    "P22": {
        "column": "P22_midhip_x_mean",
        "name": "Midhip x mean",
        "group": "body_position",
        "phase": "all",
        "source": "paper",
    },
    "P23": {
        "column": "P23_midhip_x_variance",
        "name": "Midhip x variance",
        "group": "body_position",
        "phase": "all",
        "source": "paper",
    },
    "P24": {
        "column": "P24_midhip_y_mean",
        "name": "Midhip y mean",
        "group": "body_position",
        "phase": "all",
        "source": "paper",
    },
    "P25": {
        "column": "P25_midhip_y_variance",
        "name": "Midhip y variance",
        "group": "body_position",
        "phase": "all",
        "source": "paper",
    },
    "P26": {
        "column": "P26_spine_angle_in_stand_up_max_deg",
        "name": "Spine angle in stand up max",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "paper",
    },
    "P27": {
        "column": "P27_spine_angle_in_stand_up_mean_deg",
        "name": "Spine angle in stand up mean",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "paper",
    },
    "P28": {
        "column": "P28_spine_angle_in_sitting_mean_deg",
        "name": "Spine angle in sitting",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "paper",
    },

    # ------------------------------------------------------------
    # Additional features
    # ------------------------------------------------------------
    "A01": {
        "column": "A01_gait_speed_m_s",
        "name": "Gait speed",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A02": {
        "column": "A02_number_of_steps",
        "name": "Number of steps",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A03": {
        "column": "A03_step_length_m",
        "name": "Step length",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A04": {
        "column": "A04_stride_length_m",
        "name": "Stride length",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A05": {
        "column": "A05_cadence_steps_per_min",
        "name": "Cadence",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A06": {
        "column": "A06_step_time_cv_percent",
        "name": "Step time coefficient of variation",
        "group": "spatiotemporal",
        "phase": "walking",
        "source": "additional",
    },
    "A07": {
        "column": "A07_step_signal_used",
        "name": "Step signal used",
        "group": "step_detection",
        "phase": "walking",
        "source": "additional",
    },
    "A08": {
        "column": "A08_step_length_measured_norm_mean",
        "name": "Measured step-length proxy",
        "group": "spatiotemporal",
        "phase": "lateral",
        "source": "additional",
    },
    "A09": {
        "column": "A09_spine_angle_lateral_mean_deg",
        "name": "Spine angle during walking",
        "group": "spine",
        "phase": "walking",
        "source": "additional",
    },
    "A10": {
        "column": "A10_mean_knee_rom_deg",
        "name": "Mean knee ROM",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A11": {
        "column": "A11_sit_to_stand_time_sec",
        "name": "Sit-to-stand time",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A12": {
        "column": "A12_spine_angle_standup_sd_deg",
        "name": "Spine angle stand-up standard deviation",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A13": {
        "column": "A13_spine_angle_standup_rom_deg",
        "name": "Spine angle stand-up ROM",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A14": {
        "column": "A14_spine_angle_velocity_max_deg_per_sec",
        "name": "Spine angular velocity max",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A15": {
        "column": "A15_midhip_y_displacement_norm",
        "name": "Midhip vertical displacement",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A16": {
        "column": "A16_midhip_vertical_speed_max_norm_per_sec",
        "name": "Midhip vertical speed max",
        "group": "sit_to_stand",
        "phase": "sit_to_stand",
        "source": "additional",
    },
    "A17": {
        "column": "A17_ankle_angle_mean_deg",
        "name": "Ankle angle mean",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A18": {
        "column": "A18_ankle_angle_variance_deg",
        "name": "Ankle angle variance",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A19": {
        "column": "A19_knee_angle_mean_deg",
        "name": "Knee angle mean",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A20": {
        "column": "A20_knee_angle_variance_deg",
        "name": "Knee angle variance",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A21": {
        "column": "A21_hip_angle_rom_deg",
        "name": "Hip angle ROM",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A22": {
        "column": "A22_hip_angle_max_deg",
        "name": "Hip angle max",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
    "A23": {
        "column": "A23_hip_angle_min_deg",
        "name": "Hip angle min",
        "group": "joint_angles",
        "phase": "lateral",
        "source": "additional",
    },
}


# ============================================================
# Registry utilities
# ============================================================

def feature_registry_dataframe():
    """
    Return the feature registry as a dataframe.

    Useful for checking:
        - feature ID
        - output column name
        - feature name
        - group
        - phase
        - source
    """

    rows = []

    for feature_id, info in FEATURE_REGISTRY.items():
        row = {"feature_id": feature_id}
        row.update(info)
        rows.append(row)

    return pd.DataFrame(rows)


def default_feature_values():
    """
    Return default values for all registered feature columns.

    This ensures that every extracted segment has the same output columns,
    even when a feature is not applicable to that segment.
    """

    defaults = {}

    for info in FEATURE_REGISTRY.values():
        defaults[info["column"]] = np.nan

    defaults["A02_number_of_steps"] = 0
    defaults["A07_step_signal_used"] = "none"

    return defaults


def feature_values_dataframe(result):
    """
    Combine the feature registry with the extracted values from one result row.

    This is useful in notebooks for inspecting which features were computed,
    which were not applicable, and which returned missing values.
    """

    registry_df = feature_registry_dataframe()
    registry_df["value"] = registry_df["column"].map(result)

    return registry_df