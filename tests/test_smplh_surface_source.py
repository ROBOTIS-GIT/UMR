import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import smplh_surface_source as source

class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'motion.npz'
        self.data = dict(source_format=np.array(source.FORMAT),
            vertices=np.zeros((2, 3, 3)), joints=np.zeros((2, 52, 3)),
            template_vertices=np.zeros((3, 3)), template_joints=np.zeros((52, 3)),
            faces=np.array([[0, 1, 2]]), vertex_part_ids=np.array([1, 2, 3]),
            betas=np.arange(16, dtype=float), fps=np.array(30.))
    def tearDown(self):
        self.tmp.cleanup()
    def load(self):
        np.savez(self.path, **self.data)
        return source.load(self.path)
    def test_roundtrip_and_axes(self):
        value = self.load()
        self.assertEqual(value['output_up'], 'y')
        np.testing.assert_array_equal(value['betas'], np.arange(16))
    def test_last_shape_coefficient_changes_identity(self):
        first = source.template_name(self.load())
        self.data['betas'][-1] += 1
        self.assertNotEqual(first, source.template_name(self.load()))
    def test_bad_faces(self):
        self.data['faces'][0, 2] = 3
        with self.assertRaises(ValueError): self.load()
    def test_missing(self):
        del self.data['joints']
        with self.assertRaises(ValueError): self.load()
    def test_nonfinite(self):
        self.data['vertices'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError): self.load()
    def test_truncated_betas(self):
        self.data['betas'] = np.zeros(10)
        with self.assertRaises(ValueError): self.load()
    def test_template_only_avoids_motion_arrays_and_preserves_provenance(self):
        self.data['input_sha256'] = np.array('a' * 64)
        self.data['model_sha256'] = np.array('b' * 64)
        self.load()
        value = source.load(self.path, template_only=True)
        self.assertNotIn('vertices', value)
        self.assertNotIn('joints', value)
        self.assertEqual(value['input_sha256'], 'a' * 64)
        self.assertEqual(value['model_sha256'], 'b' * 64)
        self.assertEqual(source.template_name(value), source.template_name(source.load(self.path)))
    def test_other_format(self):
        self.data['source_format'] = np.array('other')
        self.assertIsNone(self.load())
    def test_zero_fps(self):
        self.data['fps'] = np.array(0.)
        with self.assertRaises(ValueError): self.load()
    def test_object_arrays_rejected(self):
        self.data['betas'] = np.array([object()], dtype=object)
        with self.assertRaises(ValueError): self.load()


class IntegrationContracts(unittest.TestCase):
    """Exercise adapter dispatch without importing unrelated solver dependencies."""
    def function(self, filename, name, namespace):
        import ast
        path = Path(__file__).resolve().parents[1] / 'scripts' / filename
        tree = ast.parse(path.read_text())
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace[name]
    def test_motion_dispatch_preserves_faces_and_order(self):
        fn = self.function('smpl_surface_retarget_common.py', 'source_motion_vertices_joints',
                           {'smplh_surface_source': source})
        sequence = {'source_format': source.FORMAT, 'vertices': np.zeros((3,4,3)),
                    'joints': np.zeros((3,52,3)), 'faces': np.array([[0,1,2]])}
        vertices, joints, faces = fn(sequence, np.array([2,0]), None)
        self.assertEqual(vertices.shape, (2,4,3))
        self.assertEqual(joints.shape, (2,52,3))
        np.testing.assert_array_equal(faces, sequence['faces'])
        with self.assertRaises(ValueError): fn(sequence, [0], None, zero_source_finger_pose=True)
    def test_smplh_y_up_transform(self):
        matrix = np.array([[1,0,0],[0,0,-1],[0,1,0]])
        fn = self.function('smpl_surface_retarget_common.py', 'source_points_to_retarget_frame',
                          {'np': np, 'Y_UP_TO_Z_UP_MATRIX': matrix})
        np.testing.assert_array_equal(fn([[0,1,0]], 'smplh', 'y'), [[0,0,1]])
    def test_skinning_labels_and_cycle_rejection(self):
        from smplh_surface_source import vertex_parts
        weights = np.zeros((2,52)); weights[0,20]=1; weights[1,22]=1
        parents = np.zeros(52, dtype=int); parents[22] = 20
        np.testing.assert_array_equal(vertex_parts(weights, parents), [10,10])
        parents[22] = 22
        with self.assertRaises(ValueError): vertex_parts(weights, parents)

if __name__ == '__main__':
    unittest.main()
