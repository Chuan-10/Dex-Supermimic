"""Run: MUJOCO_GL=disable python -m unittest discover -s tests/dex_supermimic -v."""

import unittest
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply

from dex_supermimic.tasks.pen_spin.config.wuji_hand2.env_cfgs import wuji_hand2_pen_spin_env_cfg


class PenPlaybackResetTests(unittest.TestCase):
    def setUp(self):
        self.play = wuji_hand2_pen_spin_env_cfg(play=True)
        self.train = wuji_hand2_pen_spin_env_cfg(play=False)

    def test_play_retains_loss_detection_without_orientation_cutoff(self):
        loss = self.play.terminations["object_position_error"]
        self.assertEqual(loss.params["threshold"], 0.20)
        self.assertFalse(loss.time_out)
        self.assertNotIn("object_orientation_error", self.play.terminations)
        self.assertTrue(self.play.terminations["reference_end"].time_out)
        self.assertIn("unstable", self.play.terminations)

    def test_training_limits_are_unchanged(self):
        self.assertEqual(self.train.terminations["object_position_error"].params["threshold"], 0.05)
        self.assertEqual(self.train.terminations["object_orientation_error"].params["threshold_deg"], 60.0)

    def test_loss_detection_uses_reference_error_in_the_palm_frame(self):
        # Translate and rotate the palm so a world-origin distance check cannot
        # accidentally pass. The three pens deviate by 4, 10 and 21 cm.
        palm_pos = torch.tensor([[1.0, 2.0, 0.5]]).repeat(3, 1)
        palm_quat = torch.tensor([[0.5, 0.5, 0.5, 0.5]]).repeat(3, 1)
        reference_pos = torch.tensor([[0.0, 0.0, 0.05]]).repeat(3, 1)
        actual_pos = reference_pos.clone()
        actual_pos[:, 0] += torch.tensor([0.04, 0.10, 0.21])
        reference = torch.cat((reference_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(3, 1)), dim=-1)
        motion = SimpleNamespace(object_pose=reference)
        env = SimpleNamespace(
            scene={
                "robot": SimpleNamespace(
                    data=SimpleNamespace(
                        body_link_pos_w=palm_pos[:, None],
                        body_link_quat_w=palm_quat[:, None],
                    )
                ),
                "object": SimpleNamespace(
                    data=SimpleNamespace(
                        root_link_pos_w=palm_pos + quat_apply(palm_quat, actual_pos),
                        root_link_quat_w=palm_quat,
                    )
                ),
            },
            command_manager=SimpleNamespace(get_term=lambda name: motion),
        )
        play_loss = self.play.terminations["object_position_error"]
        train_loss = self.train.terminations["object_position_error"]
        self.assertEqual(play_loss.func(env, **play_loss.params).tolist(), [False, False, True])
        self.assertEqual(train_loss.func(env, **train_loss.params).tolist(), [False, True, True])


if __name__ == "__main__":
    unittest.main()
