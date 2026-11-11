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
2. From the FlowHMR checkout, using its Python environment, run:

   `python /path/to/UMR/scripts/export_flowhmr_surface.py motion.npz surface.npz --model assets/body_models/smplh/neutral/model.npz --fps 30`

   Replace 30 with the actual frame rate. The model must be the same neutral
   SMPL-H model resolved by FlowHMR. The exporter checks dimensions and finite
   values, but does not establish asset provenance or licensing.
3. Copy `robot_configs/humanoid_retarget_robotis_ai_sapiens_k1_example.json` to
   your own configuration. Set `robot.xml` to your separately obtained AI Sapiens
   K1 MJCF and `motion.data` to the surface NPZ. Use absolute paths or explicit `./`/`../` paths relative to the config;
   the upstream resolver may interpret other relative paths from the repository root.
   Check the body/joint names against your model revision. This is not Booster K1.
4. In the UMR environment run:

   `python scripts/humanoid_retarget_pipeline.py --config /path/to/k1.json --stage all --skip-view`

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

### Recorded compatibility checks

The implementation was exercised against UMR base
`d1cdaedaf67a827198da71c5fa831f48b2e51f91` and local FlowHMR
`b0d00759fff918ed8b1ae85847a6dc46c9e54e4d`. These are tested revisions,
not a claim that either repository's latest remote revision was checked.
The tested FlowHMR checkout carries research/non-profit terms; this test is not
commercial clearance or a statement about the license of another revision.

A 207-frame historical reconstruction was re-exported and retargeted with its
existing trained correspondence slots. Template geometry, joints, faces and
anatomical labels matched; motion vertex differences were at most 2.39e-7 metres.
Compared with the prior retarget result, root-position RMSE was 1.41e-8 metres,
joint RMSE 7.92e-8 radians, and maximum joint difference 8.35e-7 radians. All
207 output frames were finite. This checks adapter compatibility for one case;
it does not establish newly trained correspondence quality, all-motion parity,
commercial licensing, or safe hardware execution. Test assets remain external.
