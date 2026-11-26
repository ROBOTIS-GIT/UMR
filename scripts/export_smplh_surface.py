#!/usr/bin/env python3
"""Export 52-joint, 16-beta SMPL-H parameters to surface motion in the UMR environment."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import smplh_surface_source as surface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='SMPL-H NPZ: poses (T,52,3), trans (T,3), betas (16,) or (1,16)')
    parser.add_argument('output', type=Path, help='New output NPZ (never overwritten)')
    parser.add_argument('--model', type=Path, required=True, help='Matching licensed neutral SMPL-H model.npz')
    parser.add_argument('--fps', type=float, required=True, help='Actual input frame rate; no resampling')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    if args.batch_size < 1 or not np.isfinite(args.fps) or args.fps <= 0:
        parser.error('fps and batch-size must be positive')
    data = surface.load_parameters(args.input, model_path=args.model, fps=args.fps, batch_size=args.batch_size)
    # Exclusive creation prevents accidental replacement of user data.
    with args.output.open('xb') as output:
        np.savez_compressed(output, **{key: value for key, value in data.items()
                                      if key not in {'gender', 'output_up'}})
    surface.load(args.output)
    print(f'Validated surface archive: {args.output}')

if __name__ == '__main__':
    main()
