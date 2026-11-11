# FlowHMR surfaces and AI Sapiens K1

This adapter retains native SMPL-H geometry and all 16 shape coefficients rather
than interpreting SMPL-H parameters using an SMPL-X body model. It changes input
handling and robot configuration, not UMR's optimization objective or solver.

## Prerequisites and licenses

Use separately installed UMR and FlowHMR environments. Obtain the exact SMPL-H
assets required by FlowHMR through their authorized distribution. FlowHMR code,
weights, SMPL-H assets and input motion each have separate terms: UMR's MIT license
does not grant permission to redistribute them or commercially use generated
outputs. This repository does not include any of those assets. Review current
terms for your intended application; free distribution is not automatically
non-commercial use.

## Export and retarget

1. Run FlowHMR normally to obtain its NPZ containing poses (T,52,3), trans (T,3)
   and betas (1,16). Record the actual frame rate. No frame-rate conversion,
   grounding, camera adjustment or pose correction is performed by this exporter.
2. Using the FlowHMR Python environment, run the exporter with its checkout
   explicitly selected (no editable package installation or PYTHONPATH required):

   `python /path/to/UMR/scripts/export_flowhmr_surface.py motion.npz surface.npz --flowhmr-root /path/to/FlowHMR --model /path/to/FlowHMR/assets/body_models/smplh/neutral/model.npz --fps 30`

   Replace 30 with the actual frame rate. The model must be the same neutral
   SMPL-H model resolved by FlowHMR. The exporter checks dimensions and finite
   values, but does not establish asset provenance or licensing.
3. Copy `robot_configs/humanoid_retarget_robotis_ai_sapiens_k1_example.json` to
   your own configuration. Set `robot.xml` to your separately obtained AI Sapiens
   K1 MJCF. Copy `humanoid_retarget_defaults_flowhmr.json` for source settings
   and set its `motion.data` to the surface NPZ. Use absolute paths or explicit `./`/`../` paths relative to the config;
   the upstream resolver may interpret other relative paths from the repository root.
   Check the body/joint names against your model revision. This is not Booster K1.
4. In the UMR environment run:

   `python scripts/humanoid_retarget_pipeline.py --config /path/to/k1.json --defaults /path/to/flowhmr.json --stage all --skip-view`

The correspondence template is content-addressed using geometry, topology,
semantic labels and every shape coefficient. Existing SMPL-X correspondence
assets must not be reused as if they were SMPL-H assets. Cached finger poses
cannot be changed during retargeting; re-export if a different pose is needed.

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

### Latest-code validation (2026-11-11 UTC)

This adapter targets FlowHMR `df15cadf56e738bf80c42c93d5e73687ff1002d2`,
verified as upstream main on the date above. Its repository code LICENSE is
Apache-2.0. Separate pretrained weights, SMPL-H model assets and input/output
data terms still require review for the intended use.

Validation used a hash-verified snapshot of the actual latest source, not only
a comparison of individual FK files. All 223 available source/document blobs
matched upstream; one documentation PNG was unavailable and not needed to run.
The latest `save_fbx` serializer was executed with network-shaped synthetic
output and produced the expected NPZ schema. The exporter was then executed
from a different working directory with `--flowhmr-root`, without PYTHONPATH or
an editable FlowHMR installation.

A 207-frame existing reconstruction was processed through the latest FlowHMR
SMPL-H forward models and the UMR adapter using its existing trained
correspondence slots. Compared with the historical retarget result, root-position
RMSE was 1.41e-8 metres, joint RMSE 7.92e-8 radians, and maximum joint difference
8.35e-7 radians. All output frames were finite. UMR base was
`d1cdaedaf67a827198da71c5fa831f48b2e51f91`, also verified as upstream main.

This validates the latest-code serialization/FK-to-retarget interface. It does
not represent a fresh video-to-motion model inference run, validate optional
RGBMocap postprocessing, retrain correspondence for all motions, establish
commercial clearance, or certify hardware safety. Test assets remain external.

### Current FlowHMR input workflow

Follow the current upstream installation instructions and run, from FlowHMR:

`python run_v2m_demo_generation.py video.mp4`

Current outputs are `output/<video stem>/<checkpoint stem>.npz`; nonzero seeds
append `_seed<N>`. Use the actual generated NPZ, not an assumed filename. The
current serializer writes poses `(T,52,3)`, betas `(1,16)`, trans `(T,3)` and
`mocap_framerate=30`. Its output already includes FlowHMR's world-orientation
conversion; do not apply that conversion again before the surface exporter.
Verify the selected file's frame rate rather than relying on video FPS.

`--postprocess` is optional and currently uses RGBMocap contact planning and
anatomical IK. It changes generated motion and has separate dependencies; the
adapter does not enable or reproduce it. Record whether it was enabled when
comparing motion results.
