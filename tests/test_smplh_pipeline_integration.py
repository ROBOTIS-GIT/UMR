"""Merged-config regression; run with the full UMR dependency environment."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
DEPENDENCIES = ('torch','mujoco','clarabel','smplx','trimesh','scipy')

@unittest.skipUnless(all(importlib.util.find_spec(x) for x in DEPENDENCIES), 'Full UMR runtime required')
class PipelineTests(unittest.TestCase):
    def test_example_defaults_and_template_identity(self):
        import humanoid_retarget_pipeline as pipeline
        import retarget_smpl_to_humanoid_surface_vector as retarget
        import smplh_surface_source as source
        config = pipeline.load_pipeline_config(ROOT / 'robot_configs/humanoid_retarget_robotis_ai_sapiens_k1_example.json')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'surface.npz'
            np.savez(path, source_format=np.array(source.FORMAT),
                vertices=np.zeros((2,3,3)), joints=np.zeros((2,52,3)),
                template_vertices=np.zeros((3,3)), template_joints=np.zeros((52,3)),
                faces=np.array([[0,1,2]]), vertex_part_ids=np.array([1,2,3]),
                betas=np.arange(16.), fps=np.array(30.))
            config['motion']['data'] = str(path)
            key, sequence = pipeline._motion_sequence_for_template(config)
            self.assertEqual(key, 'surface')
            self.assertEqual(config['motion']['seq_key'], '')
            self.assertIs(config['retarget']['zero_source_finger_pose'], False)
            self.assertEqual(pipeline.smpl_template_config(config), retarget.smpl_template_config(config, sequence))
            self.assertEqual(pipeline.retarget_source_type(config, sequence), 'smplh')
