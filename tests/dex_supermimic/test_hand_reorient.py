"""Run with: uv run python -m unittest discover -s tests/dex_supermimic -v."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
from dex_supermimic.tasks.dex_reorient.config.task import checkpoint_task, make_env_cfg


class TaskSelectionTests(unittest.TestCase):
    def test_loading_a_different_hand_is_rejected(self):
        from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner

        runner = SimpleNamespace(device="cpu", cfg=DexSuperMimicCfg(task="hand1-reorient"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old_hand2.pt"
            torch.save({"config": {"control_hz": 50}}, path)
            for load in (DexSuperMimicRunner.load, DexSuperMimicRunner.load_model_only):
                with self.assertRaisesRegex(ValueError, "task mismatch"):
                    load(runner, path)

    def test_checkpoint_compatibility(self):
        self.assertEqual(checkpoint_task({}), "hand2-reorient")
        self.assertEqual(checkpoint_task({"task": "hand1-reorient"}), "hand1-reorient")
        with self.assertRaises(ValueError):
            checkpoint_task({"task": "unknown"})
        with self.assertRaises(ValueError):
            DexSuperMimicCfg(task="unknown")

    def test_recipes_and_packaged_assets(self):
        for task, axis in (("hand1-reorient", 0), ("hand2-reorient", 1)):
            for hz in (20, 50):
                with self.subTest(task=task, hz=hz):
                    cfg = make_env_cfg(2, hz, task=task)
                    self.assertEqual(cfg.scene.num_envs, 2)
                    self.assertEqual(cfg.rewards["cage_escape"].params["up_axis"], axis)
                    self.assertAlmostEqual(cfg.sim.mujoco.timestep * cfg.decimation, 1 / hz)
                    self.assertTrue(all(t.history_length <= 1 for t in cfg.observations["critic"].terms.values()))
                    model = cfg.scene.entities["robot"].spec_fn().compile()
                    self.assertEqual((model.nq, model.nu), (20, 20))
                    self.assertEqual(make_env_cfg(2, hz, play=True, task=task).scene.num_envs, 2)


@unittest.skipUnless(torch.cuda.is_available(), "MuJoCo Warp integration requires CUDA")
class SimulatorParityTests(unittest.TestCase):
    def test_reward_observation_and_kinematics_parity(self):
        from mjlab.envs import ManagerBasedRlEnv
        from dex_supermimic.rl.dex_supermimic.sim import SimReader
        from dex_supermimic.rl.dex_supermimic.observations import TaskObservations
        from dex_supermimic.rl.dex_supermimic.rotations import mv
        from dex_supermimic.rl.dex_supermimic.state import unpack

        for task in ("hand1-reorient", "hand2-reorient"):
            for hz in (20, 50):
                with self.subTest(task=task, hz=hz):
                    cfg = make_env_cfg(2, hz, play=True, task=task)
                    cfg.auto_reset = False
                    env = ManagerBasedRlEnv(cfg, device="cuda:0")
                    reader = None
                    try:
                        env.reset()
                        reader = SimReader(env)
                        reader.attach_capture()
                        observations = TaskObservations(reader)
                        for _ in range(24):
                            reader.refresh_frame()
                            memory = reader.memory()
                            action = torch.rand(2, 20, device="cuda:0") * 0.4 - 0.2
                            _, _, terminated, truncated, _ = env.step(action)
                            cap = reader.reward_snapshot
                            target = env.action_manager.get_term("joint_pos").processed_action
                            reward, _, terms = reader.mdp.rewards(
                                cap["physics"],
                                memory,
                                action,
                                target,
                                cap["contact"],
                                cap["torque"],
                            )
                            # mjlab rewards see body/site kinematics from before the
                            # final substep; the WM predicts integrated qpos/qvel.
                            # Compare terms that do not depend on those stale fields.
                            for name in terms.keys() - {"orientation_alignment", "tip_slide", "cage_escape"}:
                                torch.testing.assert_close(
                                    terms[name], cap["terms"][name], atol=3e-4, rtol=1e-4, msg=name
                                )
                            self.assertTrue(torch.isfinite(reward).all())
                            reader.refresh_frame()
                            state = reader.state()
                            frame = observations.frame(state, reader.memory(), observations.auxiliary())
                            expected = []
                            for name in cfg.observations["critic"].terms:
                                term = env.observation_manager.get_term_cfg("critic", name)
                                value = term.func(env, **term.params)
                                if name == "dr_params":
                                    # Detailed friction/PD context already supplies
                                    # the other three privileged DR features.
                                    value = value[:, [1, 4, 5]]
                                if term.scale is not None:
                                    value = value * term.scale
                                expected.append(value)
                            torch.testing.assert_close(frame, torch.cat(expected, -1), atol=3e-4, rtol=1e-4)
                            q, qd, *_ = unpack(state)
                            tips = reader.kin.tips(reader.kin.forward(q, qd))[0]
                            actual = mv(
                                reader.palm_rot.transpose(-1, -2)[:, None],
                                reader.robot.data.site_pos_w[:, reader.tip_ids] - reader.palm_pos[:, None],
                            )
                            torch.testing.assert_close(tips, actual, atol=2e-5, rtol=1e-4)
                            from dex_supermimic.tasks.dex_reorient.mdp.cage import _compute_cage_bounds_in_palm

                            points = reader.kin.forward(q, qd)[0][:, reader.mdp.cage_ids]
                            axis = cfg.rewards["cage_escape"].params["up_axis"]
                            lo, hi = _compute_cage_bounds_in_palm(points, 0.01, up_axis=axis)
                            # Probe just inside/outside each face to catch axis swaps.
                            center = (lo + hi) / 2
                            for dim in range(3):
                                for epsilon in (-0.001, 0.001):
                                    pos = center.clone()
                                    pos[:, dim] = hi[:, dim] + epsilon
                                    probe = state.clone()
                                    probe[:, 40:43] = pos
                                    _, counter = reader.mdp.cage(probe, torch.zeros(2, 1, device=state.device))
                                    self.assertTrue((counter == (1 if epsilon > 0 else 0)).all())
                            done = terminated | truncated
                            if done.any():
                                env.reset(env_ids=done.nonzero().flatten())
                    finally:
                        if reader is not None:
                            reader.detach_capture()
                        env.close()


if __name__ == "__main__":
    unittest.main()
