import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from k1_csv_export import K1_CSV_JOINT_NAMES, export_k1_csv


class K1CsvTests(unittest.TestCase):
    def model(self, names):
        joints = "".join(
            f'<body><joint name="{name}" type="hinge"/><geom type="sphere" size=".01"/></body>'
            for name in names
        )
        return mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/>'
            '<geom type="sphere" size=".02"/>' + joints + '</body></worldbody></mujoco>'
        )

    def test_reorders_joints_and_quaternion_without_mutating_qpos(self):
        model = self.model(reversed(K1_CSV_JOINT_NAMES))
        poses = np.tile(model.qpos0, (3, 1))
        poses[:, :3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        poses[:, 3:7] = [0.5, -0.5, 0.5, -0.5]
        expected = np.arange(23) / 100
        for index, name in enumerate(K1_CSV_JOINT_NAMES):
            joint = model.joint(name).id
            poses[:, model.jnt_qposadr[joint]] = expected[index]
        original = poses.copy()
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "result_umr_qpos.npz"
            np.savez_compressed(npz, qpos=poses, fps=[30])
            saved_bytes = npz.read_bytes()
            path = export_k1_csv("robotis_ai_sapiens_k1", model, poses, npz, 30)
            self.assertEqual(path, Path(directory) / "result_umr.csv")
            csv = np.loadtxt(path, delimiter=",")
            self.assertEqual(csv.shape, (3, 30))
            np.testing.assert_allclose(csv[:, :3], poses[:, :3])
            np.testing.assert_allclose(csv[:, 3:7], poses[:, [4, 5, 6, 3]])
            np.testing.assert_allclose(csv[:, 7:], np.tile(expected, (3, 1)))
            np.testing.assert_array_equal(poses, original)
            self.assertEqual(npz.read_bytes(), saved_bytes)
            plain_path = export_k1_csv("robotis_ai_sapiens_k1", model, poses,
                                       Path(directory) / "plain.npz", 30)
            self.assertEqual(plain_path.name, "plain_umr.csv")
            self.assertEqual(plain_path.read_bytes(), path.read_bytes())

    def test_other_robot_does_not_export(self):
        self.assertIsNone(export_k1_csv("unitree_g1", None, None, "unused.npz", 30))

    def test_missing_joint_is_reported_before_writing(self):
        model = self.model(K1_CSV_JOINT_NAMES[:-1])
        with tempfile.TemporaryDirectory() as directory:
            npz = Path(directory) / "result.npz"
            with self.assertRaisesRegex(ValueError, "right_wrist_roll_joint"):
                export_k1_csv("robotis_ai_sapiens_k1", model, model.qpos0[None], npz, 30)
            self.assertEqual(list(Path(directory).glob('*.csv')), [])

    def test_nonfinite_pose_is_rejected(self):
        model = self.model(K1_CSV_JOINT_NAMES)
        poses = model.qpos0[None].copy()
        poses[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            export_k1_csv("robotis_ai_sapiens_k1", model, poses, "unused.npz", 30)


if __name__ == "__main__":
    unittest.main()
