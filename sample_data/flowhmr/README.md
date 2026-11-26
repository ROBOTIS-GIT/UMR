# FlowHMR motion and AI Sapiens K1

This adapter retains native SMPL-H geometry and all 16 shape coefficients rather
than interpreting SMPL-H parameters using an SMPL-X body model. It changes input
handling and robot configuration, not UMR's optimization objective or solver.

## Prerequisites and licenses

Run retargeting in the UMR environment; FlowHMR code and its Python
environment are not required. Obtain the matching neutral SMPL-H model through
its authorized distribution. FlowHMR code,
weights, SMPL-H assets and input motion each have separate terms: UMR's MIT license
does not grant permission to redistribute them or commercially use generated
outputs. This repository does not include any of those assets. Review current
terms for your intended application; free distribution is not automatically
non-commercial use.

## Retarget directly

1. Run FlowHMR to obtain an NPZ with `poses (T,52,3)`, `trans (T,3)`,
   `betas (16,)` or `(1,16)`, and scalar `mocap_framerate`.
   Poses are absolute axis-angle rotations, including fingers, in radians;
   translations are metres in Y-up coordinates.
2. Place the matching licensed neutral SMPL-H model at
   `UMR/smplh/neutral/model.npz`.
3. Copy `robot_configs/humanoid_retarget_robotis_ai_sapiens_k1_example.json`
   to your robot configuration and set `robot.xml` to your K1 MJCF.
   Copy `humanoid_retarget_defaults_flowhmr.json` to your source configuration.
   Set `motion.data` to the **FlowHMR reconstruction NPZ** and, if needed,
   `smplh_model_path` to the licensed model. Use absolute paths or explicit
   `./`/`../` paths relative to the config. This is AI Sapiens K1, not Booster K1.
4. Run in the UMR environment:

   ```bash
   conda activate umr
   python scripts/humanoid_retarget_pipeline.py \
     --config /path/to/k1.json --defaults /path/to/flowhmr.json \
     --stage all --skip-view
   ```

UMR computes the surface and joint coordinates internally. No FlowHMR imports
or intermediate surface archive are needed. Shape blending retains all 16
coefficients; hand means and pose corrective blend shapes are not added,
matching the separate converter. This adapter supports the stated FlowHMR
schema, not arbitrary SMPL-family NPZ files.

Keep `use_betas=true` and `zero_source_finger_pose=false`. The correspondence
template is content-addressed using geometry, topology, semantic labels and
shape coefficients; matching direct and cached inputs reuse the same template.

### Optional surface export

Previously generated surface NPZ files remain supported as `motion.data`,
without requiring the licensed model again. To precompute one, or to specify
the actual frame rate when input metadata is absent:

```bash
python scripts/export_smplh_surface.py \
  /path/to/motion.npz output/surface.npz \
  --model smplh/neutral/model.npz --fps 30
```

Replace 30 with the actual motion sample rate. The exporter does not resample
and refuses to overwrite an existing file. Both paths use the same evaluator.

## K1 CSV for Cyclo

With `robot.name` set to `robotis_ai_sapiens_k1`, retargeting keeps the original
NPZ and also writes a CSV. Set `retarget.out` to a name such as
`output/walk_01_umr_qpos.npz` to distinguish the UMR archive from Cyclo training
NPZ files; the source-rate CSV is saved as `output/walk_01_umr.csv`. The K1 pipeline's generated
default name also ends in `_umr_qpos.npz`. Explicit output paths are used as given,
and CSV export uses `_umr.csv` instead of `_umr_qpos.npz` (without duplicating
an existing `_umr` suffix). This distinguishes it from the resampled Cyclo CSV.
Other robot profiles are unchanged.
The CSV has no header: root position in metres, root quaternion in **xyzw** order,
then the 23 K1 joint angles in Cyclo CSV order (radians).

The CSV preserves the NPZ frame rate, including any motion stride; it is not
resampled. The save message prints the actual FPS. For FlowHMR at 30 FPS with
stride 1, run this inside Cyclo MJLab to produce 50 Hz training/deployment files:

```bash
python scripts/tools/motion/csv_to_npz.py \
  -f /path/to/walk_01_umr.csv \
  --input_fps 30 --output_fps 50 \
  --model third_party/ai_sapiens/ai_sapiens_description/mujoco/k1/k1.xml \
  --output_name /path/to/converted/walk_01.npz
```

Use the model matching the retargeted robot (the example K1 profile uses Rev.1).
For Cyclo Lab, use its `csv_to_npz.py` with `--robot k1_rev1` instead of `--model`.
Use the FPS printed by UMR rather than assuming 30 for other inputs or strides.
The current Cyclo CLI accepts integer FPS; fractional rates require converter
support before using this command. No FPS is embedded in the headerless CSV;
the original NPZ retains it.

CSV export happens when retargeting runs. If the pipeline reuses an existing NPZ,
use `--force-retarget` to regenerate the result and its CSV.

## Validation status and limits

Run `python -m unittest discover -s tests -v` in the UMR environment. The suite
covers malformed archives, all-16-beta identity, axis conversion, native labels,
degenerate triangle projection and merged-default/template consistency. The
full-import regression is explicitly skipped when UMR dependencies are missing.

A bounded integration smoke test has also exercised real SMPL-H forward models,
K1 MJCF sampling, 64-point correspondence training for one epoch, and two-frame
retargeting, producing finite (2,30) qpos output. Synthetic poses and deliberately
short training establish execution only, not motion quality. No claim of parity
with a production motion dataset or hardware safety follows from this test.

Review output in simulation and validate joint ordering/limits and downstream
motion formats separately. A UMR qpos archive is not automatically a robot
policy-training CSV. Longer representative motion validation and independent
review remain necessary before calling this integration production-ready.

### Standalone conversion parity (2026-11-26)

The standalone exporter was compared with the previous exporter using FlowHMR
`df15cadf56e738bf80c42c93d5e73687ff1002d2` and the same neutral SMPL-H model.
For an existing 694-frame reconstruction, all output arrays and metadata were
exactly equal: vertices, 52 skeletal joints, template vertices/joints, faces,
body-part labels, all 16 betas, FPS, and input/model hashes. Maximum absolute
vertex and joint differences were **0 metres** in this CPU float32 comparison.
The identical template geometry also preserves the correspondence template ID.

The comparison ran in an existing UMR Python 3.12 environment with PyTorch
2.11.0+cu128 and UMR's `smplx` dependency. Synthetic regression tests cover root
rotation/translation, finger articulation, the 16th shape coefficient, batch
boundaries, invalid input, and refusal to overwrite an output. No FlowHMR code
is needed by the exporter or its regression tests. Licensed models and motion
fixtures are not committed. This establishes conversion parity for the tested
inputs, not robot motion quality or hardware deployment readiness.

### Direct-input parity (2026-11-26)

The direct parameter loader was compared against the previously generated
FlowHMR-based surface archive for the same 694-frame reconstruction. All
vertices, joints, template geometry, labels and betas were exactly equal;
maximum coordinate difference was **0 metres**, with the same template ID.

Using the same Rev.2 model and a shared 64-point, one-epoch correspondence
checkpoint, two-frame retargeting produced identical qpos, frame IDs and FPS
for direct and cached input in both the standard and chunked retargeters.
The standard retargeter's CSV files were byte-for-byte identical. This is a
bounded equivalence check, not an assessment of motion quality. All 34
regression tests passed in the UMR environment; licensed fixtures remain local.
