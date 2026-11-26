"""Evaluate native SMPL-H surfaces using UMR's torch/smplx dependencies.

FlowHMR exports absolute axis-angle hand poses and uses shape blending without
pose corrective blend shapes. Keep those semantics rather than using SMPLH's
PCA/hand-mean/pose-corrective defaults. No FlowHMR code is imported here.
"""
from pathlib import Path

import numpy as np
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints


class SMPLHSurfaceModel:
    """CPU float32 evaluator for the neutral, 52-joint, 16-beta NPZ model."""

    def __init__(self, model_path):
        with np.load(Path(model_path), allow_pickle=False) as data:
            self.template = torch.tensor(data['v_template'], dtype=torch.float32)
            self.shape_dirs = torch.tensor(data['shapedirs'], dtype=torch.float32)
            self.regressor = torch.tensor(data['J_regressor'], dtype=torch.float32)
            self.weights = torch.tensor(data['weights'], dtype=torch.float32)
            self.parents = torch.tensor(data['kintree_table'][0], dtype=torch.long)
            self.faces = np.array(data['f'], dtype=np.int32)
        count = len(self.template)
        if (self.template.shape != (count, 3)
                or self.shape_dirs.shape != (count, 3, 16)
                or self.regressor.shape != (52, count)
                or self.weights.shape != (count, 52)
                or self.parents.shape != (52,)):
            raise ValueError('Expected a 52-joint SMPL-H model with 16 shape coefficients')
        self.parents[0] = -1
        self.joint_template = self.regressor @ self.template
        self.joint_shape_dirs = torch.einsum('jv,vdb->jdb', self.regressor, self.shape_dirs)

    def __call__(self, poses, betas, trans):
        """Return vertices (T,V,3) and skeletal joints (T,52,3), in metres/Y-up."""
        batch = len(poses)
        rotations = batch_rodrigues(poses.reshape(-1, 3)).reshape(batch, 52, 3, 3)
        shapes = betas.repeat(batch, 1)
        shaped_vertices = self.template + blend_shapes(shapes, self.shape_dirs)
        mesh_joints = vertices2joints(self.regressor, shaped_vertices)
        _, transforms = batch_rigid_transform(rotations, mesh_joints, self.parents)
        skinning = torch.matmul(
            self.weights.unsqueeze(0).expand(batch, -1, -1),
            transforms.reshape(batch, 52, 16),
        ).reshape(batch, -1, 4, 4)
        homogeneous = torch.cat((shaped_vertices, torch.ones(batch, len(self.template), 1)), dim=2)
        vertices = (skinning @ homogeneous.unsqueeze(-1))[:, :, :3, 0]

        # Match the separate skeleton evaluator's operation order as well as
        # its joint definitions; regressing the skinned mesh is not equivalent.
        shaped_joints = self.joint_template[None] + blend_shapes(betas, self.joint_shape_dirs)
        joints, _ = batch_rigid_transform(rotations, shaped_joints.repeat(batch, 1, 1), self.parents)
        return vertices + trans[:, None, :], joints + trans[:, None, :]
