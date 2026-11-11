#!/usr/bin/env python3
"""Export FlowHMR SMPL-H parameters in the user's licensed FlowHMR environment."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import smplh_surface_source as surface


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='FlowHMR NPZ: poses, trans, betas')
    parser.add_argument('output', type=Path, help='New output NPZ (never overwritten)')
    parser.add_argument('--model', type=Path, required=True, help='Licensed neutral SMPL-H model.npz used by FlowHMR')
    parser.add_argument('--fps', type=float, required=True, help='Actual input frame rate; no resampling')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    if args.batch_size < 1 or not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('fps and batch-size must be positive')
    import torch
    from flowhmr.core.bodymodels.smpl_skeleton import SMPLMesh, SMPLSkeleton
    with np.load(args.input, allow_pickle=False) as data:
        poses, trans, betas = (np.array(data[key], dtype=np.float32) for key in ('poses', 'trans', 'betas'))
    if poses.ndim != 3 or poses.shape[1:] != (52, 3) or len(poses) == 0 or trans.shape != (len(poses), 3) or betas.shape not in {(16,), (1,16)}:
        raise ValueError('Expected poses (T,52,3), trans (T,3), and betas (16,) or (1,16)')
    if not all(np.isfinite(value).all() for value in (poses, trans, betas)):
        raise ValueError('Non-finite FlowHMR input')
    # Use one explicit asset for both forward kinematics and mesh anatomy.
    mesh = SMPLMesh(model_path=str(args.model.resolve()))
    skeleton = SMPLSkeleton(model_path=str(args.model.resolve()))
    shape = torch.from_numpy(betas.reshape(1,16))
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, len(poses), args.batch_size):
            parameters = {'poses': torch.from_numpy(poses[start:start+args.batch_size]),
                          'shapes': shape, 'trans': torch.from_numpy(trans[start:start+args.batch_size])}
            vertices.append(mesh(parameters)['vertices'].detach().cpu().numpy())
            joints.append(skeleton(parameters)['keypoints3d'].detach().cpu().numpy())
        parameters = {'poses': torch.zeros(1,52,3), 'shapes': shape, 'trans': torch.zeros(1,3)}
        template_vertices = mesh(parameters)['vertices'][0].detach().cpu().numpy()
        template_joints = skeleton(parameters)['keypoints3d'][0].detach().cpu().numpy()
    with np.load(args.model, allow_pickle=False) as model:
        faces = np.array(model['f'], dtype=np.int32)
        weights = np.array(model['weights'], dtype=np.float32)
        parents = np.array(model['kintree_table'][0], dtype=np.int64)
    if len(weights) != len(template_vertices):
        raise ValueError('Model topology does not match FlowHMR mesh')
    # Exclusive creation prevents accidental replacement of user data.
    with args.output.open('xb') as output:
        np.savez_compressed(output, source_format=np.array(surface.FORMAT),
            vertices=np.concatenate(vertices), joints=np.concatenate(joints),
            template_vertices=template_vertices, template_joints=template_joints,
            faces=faces, vertex_part_ids=vertex_parts(weights, parents),
            betas=betas.reshape(16), fps=np.array(args.fps),
            input_sha256=np.array(hashlib.sha256(args.input.read_bytes()).hexdigest()),
            model_sha256=np.array(hashlib.sha256(args.model.read_bytes()).hexdigest()))
    surface.load(args.output)
    print(f'Validated surface archive: {args.output}')

if __name__ == '__main__':
    main()
