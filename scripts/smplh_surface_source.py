"""Pickle-free native SMPL-H surface interchange (metres, Y-up).

Users supply licensed model assets separately. No SMPL-H assets are distributed
with this adapter. Cached surfaces retain their original 16 shape coefficients.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np

FORMAT = "umr_smplh_surface_v1"


def is_sequence(sequence):
    return sequence is not None and sequence.get("source_format") == FORMAT


def load(path, *, template_only=False):
    """Return None for other NPZ formats; fail closed for malformed declared data."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if "source_format" not in archive or str(archive["source_format"].item()) != FORMAT:
            return None
        required = ("vertices", "joints", "template_vertices", "template_joints",
                    "faces", "vertex_part_ids", "betas", "fps")
        missing = set(required) - set(archive.files)
        if missing:
            raise ValueError(f"Missing SMPL-H surface fields: {sorted(missing)}")
        selected = tuple(key for key in required if not template_only or key not in {"vertices", "joints"})
        data = {key: np.array(archive[key], copy=True) for key in selected}
        for key in ("input_sha256", "model_sha256"):
            if key in archive:
                digest = str(archive[key].item())
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError(f"Invalid SHA-256 field: {key}")
                data[key] = digest
    vertices = data["template_vertices"][None] if template_only else data["vertices"]
    joints = data["template_joints"][None] if template_only else data["joints"]
    if vertices.ndim != 3 or vertices.shape[0] == 0 or vertices.shape[1] == 0 or vertices.shape[2] != 3:
        raise ValueError("vertices must have shape (frames, vertices, 3)")
    if joints.shape != (len(vertices), 52, 3):
        raise ValueError("joints must have shape (frames, 52, 3)")
    if data["template_vertices"].shape != vertices.shape[1:] or data["template_joints"].shape != (52, 3):
        raise ValueError("Template and motion topology must match")
    faces = data["faces"]
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must be nonempty integer triangles")
    if faces.min() < 0 or faces.max() >= vertices.shape[1]:
        raise ValueError("Triangle vertex index out of bounds")
    parts = data["vertex_part_ids"]
    if parts.shape != (vertices.shape[1],) or not np.issubdtype(parts.dtype, np.integer) or parts.min() < 0 or parts.max() > 19:
        raise ValueError("vertex_part_ids must use UMR body-part IDs (0..19)")
    if data["betas"].shape != (16,):
        raise ValueError("betas must retain all 16 native SMPL-H coefficients")
    for key in selected:
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"Non-finite SMPL-H field: {key}")
    if data["fps"].shape != () or float(data["fps"]) <= 0:
        raise ValueError("fps must be a positive scalar")
    data.update(source_format=FORMAT, fps=float(data["fps"]), gender="neutral", output_up="y")
    return data


def template_name(sequence):
    digest = hashlib.sha256()
    for key in ("template_vertices", "template_joints", "faces", "vertex_part_ids", "betas"):
        value = np.ascontiguousarray(sequence[key])
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(value.tobytes())
    return "smplh_surface_" + digest.hexdigest()[:16]


def build_sample(sequence, build, name, num_points, seed, oversample_ratio,
                 curvature_weight, curvature_power, center_mode, bbox_center_ratio):
    vertices = sequence["template_vertices"]
    center, _, saved_mode = build.smpl_center_offset(
        vertices, sequence["template_joints"], center_mode, bbox_center_ratio)
    vertices = vertices - center[None, :]
    points, face_ids = build.sample_surface(vertices, sequence["faces"], num_points,
        seed, oversample_ratio, curvature_weight, curvature_power)
    return {"name": name, "points": build.as_root_frame(points),
            "vertices": build.as_root_frame(vertices), "faces": sequence["faces"],
            "sample_face_ids": face_ids, "root_offset": center.astype(np.float32),
            "center_mode": saved_mode, "betas": sequence["betas"].copy()}


def template_config(sequence, options=None):
    options = options or {}
    if options.get("source", "motion") != "motion" or not options.get("use_betas", True):
        raise ValueError("Cached SMPL-H surfaces require source=motion and use_betas=true; re-export to change shape")
    return {"name": template_name(sequence), "type": "smplh",
            "gender": "neutral", "betas": None, "source": "motion"}
