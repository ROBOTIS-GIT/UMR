"""Standalone conversion checks with a synthetic, unlicensed SMPL-H fixture."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from smplh_surface_model import SMPLHSurfaceModel
import smplh_surface_source as surface


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model_path = self.root / 'model.npz'
        self.template = np.zeros((53, 3), dtype=np.float32)
        self.template[:52, 0] = np.arange(52) * .01
        self.template[52] = self.template[51] + [0, .1, 0]
        shape_dirs = np.zeros((53, 3, 16), dtype=np.float32)
        shape_dirs[:, 2, 15] = .04
        weights = np.zeros((53, 52), dtype=np.float32)
        weights[:52] = np.eye(52)
        weights[52, 51] = 1
        self.model_data = dict(
            v_template=self.template, shapedirs=shape_dirs,
            J_regressor=np.eye(52, 53, dtype=np.float32), weights=weights,
            kintree_table=np.array([[-1] + list(range(51)), list(range(52))]),
            f=np.array([[0, 51, 52]], dtype=np.int32),
            # Neither hand means nor pose correctives belong in this path.
            hands_meanl=np.ones(45, dtype=np.float32),
            hands_meanr=np.ones(45, dtype=np.float32),
            posedirs=np.ones((53, 3, 459), dtype=np.float32),
        )
        np.savez(self.model_path, **self.model_data)
        self.model = SMPLHSurfaceModel(self.model_path)
        self.betas = torch.zeros(1, 16)
        self.betas[0, 15] = 2

    def test_zero_pose_keeps_all_betas_and_absolute_hand_pose(self):
        with torch.no_grad():
            vertices, joints = self.model(torch.zeros(1, 52, 3), self.betas, torch.zeros(1, 3))
        expected = self.template + [0, 0, .08]
        np.testing.assert_allclose(vertices[0], expected, atol=1e-6)
        np.testing.assert_allclose(joints[0], expected[:52], atol=1e-6)

    def test_root_rotation_translation_and_hand_articulation(self):
        poses = torch.zeros(3, 52, 3)
        poses[1, 0, 2] = np.pi / 2
        poses[2, 51, 0] = np.pi / 2
        translation = torch.tensor([[.1, .2, .3]]).repeat(3, 1)
        with torch.no_grad():
            vertices, joints = self.model(poses, self.betas, translation)
        expected = np.broadcast_to(self.template + [0, 0, .08], (3, 53, 3)).copy()
        expected[1, :, 0] = -self.template[:, 1]
        expected[1, :, 1] = self.template[:, 0]
        expected[2, 52] = expected[2, 51] + [0, 0, .1]
        expected += translation.numpy()[:, None]
        np.testing.assert_allclose(vertices, expected, atol=1e-6)
        np.testing.assert_allclose(joints, expected[:, :52], atol=1e-6)

    def test_rejects_model_with_truncated_shape_space(self):
        self.model_data['shapedirs'] = self.model_data['shapedirs'][:, :, :10]
        np.savez(self.model_path, **self.model_data)
        with self.assertRaisesRegex(ValueError, '16 shape coefficients'):
            SMPLHSurfaceModel(self.model_path)

    def run_export(self, input_path, output_path, batch_size):
        return subprocess.run([
            sys.executable, str(SCRIPTS / 'export_smplh_surface.py'),
            str(input_path), str(output_path), '--model', str(self.model_path),
            '--fps', '29.97', '--batch-size', str(batch_size),
        ], cwd=self.root, env=dict(os.environ, PYTHONPATH='', OMP_NUM_THREADS='1'),
            capture_output=True, text=True)

    def test_cli_without_flowhmr_checkout_and_batch_boundaries(self):
        motion = self.root / 'motion.npz'
        poses = np.zeros((3, 52, 3), dtype=np.float32)
        poses[1, 0, 2] = np.pi / 2
        poses[2, 51, 0] = np.pi / 2
        np.savez(motion, poses=poses, betas=self.betas.numpy(), trans=np.zeros((3, 3), dtype=np.float32))
        outputs = []
        for batch in (1, 2, 32):
            path = self.root / f'surface_{batch}.npz'
            result = self.run_export(motion, path, batch)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(surface.load(path))
        for key in ('vertices', 'joints', 'template_vertices', 'template_joints', 'faces', 'vertex_part_ids', 'betas'):
            for output in outputs[1:]:
                np.testing.assert_allclose(output[key], outputs[0][key], atol=1e-6)
        self.assertEqual(outputs[0]['fps'], 29.97)
        self.assertEqual(outputs[0]['input_sha256'], hashlib.sha256(motion.read_bytes()).hexdigest())
        self.assertEqual(outputs[0]['model_sha256'], hashlib.sha256(self.model_path.read_bytes()).hexdigest())
        original = (self.root / 'surface_1.npz').read_bytes()
        result = self.run_export(motion, self.root / 'surface_1.npz', 1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Output already exists', result.stderr)
        self.assertEqual((self.root / 'surface_1.npz').read_bytes(), original)

    def test_direct_input_matches_export_and_pipeline_template(self):
        import json
        import humanoid_retarget_pipeline as pipeline
        import smpl_surface_retarget_common as common
        motion = self.root / 'motion.npz'
        poses = np.zeros((33, 52, 3), dtype=np.float32)
        poses[:, 0, 2] = np.linspace(0, .5, 33)
        poses[:, 51, 0] = .3
        np.savez(motion, poses=poses, trans=np.ones((33, 3), dtype=np.float32),
                 betas=self.betas.numpy(), mocap_framerate=np.array(29.97))
        output = self.root / 'surface.npz'
        result = self.run_export(motion, output, 32)
        self.assertEqual(result.returncode, 0, result.stderr)
        cached = surface.load(output)
        direct = surface.load(motion, model_path=self.model_path)
        for key in cached:
            np.testing.assert_array_equal(direct[key], cached[key], err_msg=key)
        only = surface.load(motion, model_path=self.model_path, template_only=True)
        self.assertNotIn('vertices', only)
        self.assertEqual(surface.template_name(only), surface.template_name(cached))
        motions, _ = common.load_motion_collection(motion, smplh_model_path=self.model_path)
        ids = np.array([32, 1, 0])
        for a, b in zip(common.source_motion_vertices_joints(motions['motion'], ids, None),
                        common.source_motion_vertices_joints(cached, ids, None)):
            np.testing.assert_array_equal(a, b)
        config_path = self.root / 'config.json'
        config_path.write_text(json.dumps({'motion': {'data': './motion.npz'},
                                           'smplh_model_path': './model.npz'}))
        config = pipeline.load_pipeline_config(
            config_path, SCRIPTS.parent / 'humanoid_retarget_defaults_flowhmr.json')
        self.assertEqual(config['smplh_model_path'], str(self.model_path))
        self.assertEqual(pipeline.smpl_template_config(config)['name'], surface.template_name(cached))

    def test_direct_input_requires_frame_rate_and_ignores_other_schemas(self):
        motion = self.root / 'motion.npz'
        np.savez(motion, poses=np.zeros((1, 52, 3)), trans=np.zeros((1, 3)), betas=np.zeros(16))
        with self.assertRaisesRegex(ValueError, 'mocap_framerate'):
            surface.load(motion, model_path=self.model_path)
        np.savez(motion, poses=np.zeros((1, 55, 3)), trans=np.zeros((1, 3)), betas=np.zeros(10))
        self.assertIsNone(surface.load(motion, model_path=self.model_path))

    def test_nonfinite_motion_is_rejected_before_output(self):
        motion = self.root / 'invalid.npz'
        poses = np.zeros((1, 52, 3), dtype=np.float32)
        poses[0, 0, 0] = np.nan
        np.savez(motion, poses=poses, trans=np.zeros((1, 3)), betas=np.zeros(16))
        output = self.root / 'surface.npz'
        result = self.run_export(motion, output, 1)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Non-finite', result.stderr)
        self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
