"""SMPL-H parameter and cached-surface input adapter (metres, Y-up).

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


def load(path, *, template_only=False, model_path=None):
    """Read native parameters or cached surfaces; return None for other formats."""
    with np.load(Path(path), allow_pickle=False) as archive:
        if "source_format" not in archive:
            if "poses" in archive and archive["poses"].shape[1:] == (52, 3):
                return load_parameters(path, model_path=model_path, template_only=template_only)
            return None
        if str(archive["source_format"].item()) != FORMAT:
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


def vertex_parts(weights, parents):
    """Map native SMPL-H skinning weights onto UMR's anatomical segment IDs."""
    parts = np.array([15,14,2,7,5,11,7,6,8,7,6,8,4,16,17,4,3,9,12,13,10,1])
    if weights.ndim != 2 or weights.shape[1] != 52 or parents.shape != (52,):
        raise ValueError('Expected 52-joint native SMPL-H skinning data')
    if not np.isfinite(weights).all() or (weights < 0).any() or not np.allclose(weights.sum(1), 1, atol=2e-5):
        raise ValueError('Invalid SMPL-H skinning weights')
    groups = np.zeros((len(weights), 20), dtype=np.float32)
    for joint in range(52):
        ancestor, visited = joint, set()
        while ancestor >= 22:
            if ancestor in visited or ancestor >= 52:
                raise ValueError('Invalid or cyclic SMPL-H hierarchy')
            visited.add(ancestor)
            ancestor = int(parents[ancestor])
        if ancestor < 0:
            raise ValueError('Finger joint has no anatomical ancestor')
        groups[:, parts[ancestor]] += weights[:, joint]
    return groups.argmax(1).astype(np.int32)


def load_parameters(path, *, model_path=None, fps=None, template_only=False, batch_size=32):
    """Evaluate FlowHMR's absolute SMPL-H parameters without an intermediate file.

    No hand means, pose correctives or coordinate conversion are added. Both the
    direct loader and optional surface exporter use this evaluation path.
    """
    import torch
    from smplh_surface_model import SMPLHSurfaceModel

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        poses, trans, betas = (np.array(archive[key], dtype=np.float32) for key in ('poses', 'trans', 'betas'))
        if fps is None:
            if 'mocap_framerate' not in archive:
                raise ValueError('SMPL-H parameters require mocap_framerate; or use the surface exporter with --fps')
            fps = float(archive['mocap_framerate'].item())
    if (poses.ndim != 3 or poses.shape[1:] != (52, 3) or len(poses) == 0
            or trans.shape != (len(poses), 3) or betas.shape not in {(16,), (1, 16)}):
        raise ValueError('Expected poses (T,52,3), trans (T,3), and betas (16,) or (1,16)')
    if not all(np.isfinite(value).all() for value in (poses, trans, betas)):
        raise ValueError('Non-finite SMPL-H input')
    if not np.isfinite(fps) or fps <= 0 or batch_size < 1:
        raise ValueError('fps and batch-size must be positive')
    model_path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[1] / 'smplh/neutral/model.npz'
    model = SMPLHSurfaceModel(model_path)
    shape = torch.from_numpy(betas.reshape(1, 16))
    data = {}
    with torch.no_grad():
        if not template_only:
            vertices, joints = [], []
            for start in range(0, len(poses), batch_size):
                batch_vertices, batch_joints = model(
                    torch.from_numpy(poses[start:start + batch_size]), shape,
                    torch.from_numpy(trans[start:start + batch_size]),
                )
                vertices.append(batch_vertices.numpy())
                joints.append(batch_joints.numpy())
            data.update(vertices=np.concatenate(vertices), joints=np.concatenate(joints))
        template_vertices, template_joints = model(torch.zeros(1, 52, 3), shape, torch.zeros(1, 3))
    data.update(
        source_format=FORMAT, gender='neutral', output_up='y', fps=float(fps),
        template_vertices=template_vertices[0].numpy(), template_joints=template_joints[0].numpy(),
        faces=model.faces, vertex_part_ids=vertex_parts(model.weights.numpy(), model.parents.numpy()),
        betas=betas.reshape(16),
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )
    return data
