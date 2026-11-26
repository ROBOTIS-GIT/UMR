"""Export AI Sapiens K1 qpos in the numeric CSV format consumed by Cyclo."""
from pathlib import Path

import mujoco
import numpy as np


# Cyclo Lab and Cyclo MJLab use this order for both K1 revisions.
K1_CSV_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint",
)


def export_k1_csv(robot_name, model, qpos, npz_path, fps):
    """Write a sibling *_umr.csv for K1; leave other robot outputs alone.

    Columns: root xyz (metres), quaternion xyzw, 23 joint angles (radians).
    The original frame clock and poses are preserved; Cyclo handles resampling.
    """
    if robot_name != "robotis_ai_sapiens_k1":
        return None
    poses = np.asarray(qpos)
    if poses.ndim != 2 or poses.shape[1] != model.nq or len(poses) == 0:
        raise ValueError("K1 CSV export requires nonempty qpos with shape (frames, model.nq)")
    if not np.isfinite(poses).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("K1 CSV export requires finite poses and a positive FPS")
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(roots) != 1:
        raise ValueError("K1 CSV export requires exactly one floating root")
    root = int(model.jnt_qposadr[roots[0]])
    addresses = []
    for name in K1_CSV_JOINT_NAMES:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"K1 CSV export requires hinge joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint]))
    values = np.concatenate((
        poses[:, root:root + 3],
        poses[:, [root + 4, root + 5, root + 6, root + 3]],
        poses[:, addresses],
    ), axis=1)
    path = Path(npz_path)
    stem = path.stem.removesuffix("_umr_qpos").removesuffix("_umr")
    path = path.with_name(stem + "_umr.csv")
    np.savetxt(path, values, delimiter=",", fmt="%.9f")
    print(f"[HumanoidRetarget] saved {path} (Cyclo K1 CSV, xyzw, {fps:g} FPS). "
          f"Use --input_fps {fps:g} with csv_to_npz.py.")
    return path
