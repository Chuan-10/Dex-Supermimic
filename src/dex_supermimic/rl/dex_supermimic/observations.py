"""Differentiable clean task observations for real and imagined states.

Both critic and dynamics use only the current frame, including simulator context.
"""

import torch

from .rotations import mv

from .state import unpack

BASE_TERMS = {
    "joint_angles": 20,
    "qpos_error": 20,
    "cube_pos_in_tag": 3,
    "cube_ori_error": 6,
    "action_history": 20,
    "true_joint_pos": 20,
    "true_joint_vel": 20,
    "fingertip_positions": 15,
}
CURRENT_TERMS = {
    "cube_pos_in_tag_clean": 3,
    "cube_ori_error_clean": 6,
    "cube_linvel": 3,
    "cube_angvel": 3,
    "pert_dir": 3,
    "pert_velocity": 6,
    "state_progress": 4,
    "cage_counter": 1,
    # Friction/Kp/Kd ratios are redundant with the detailed simulator context.
    "dr_params": 3,
    "palm_rot_6d": 6,
}
BASE_DIM = sum(BASE_TERMS.values())
CURRENT_DIM = sum(CURRENT_TERMS.values())
FRAME_DIM = BASE_DIM + CURRENT_DIM


class TaskObservations:
    def __init__(self, reader):
        self.reader = reader
        self.mdp = reader.mdp
        env = reader.env
        tip_cfg = env.observation_manager.get_term_cfg("critic", "fingertip_positions")
        self.tip_ids = tip_cfg.params["asset_cfg"].body_ids
        self.tip_scale = tip_cfg.scale
        self.tip_offset = torch.tensor(tip_cfg.params["offset"], device=env.device)
        self.default_velocity = reader.robot.data.default_joint_vel[0].clone()
        self.episode_steps = env.max_episode_length
        self.context_dim = reader.xi.shape[-1] + FRAME_DIM

    def auxiliary(self):
        """Exogenous metadata, stored per transition, never inferred from env IDs."""
        reader = self.reader
        env = reader.env
        from dex_supermimic.tasks.dex_reorient.mdp.runtime_state import (
            get_reorient_runtime_state,
        )

        runtime = get_reorient_runtime_state(env)
        cfg = env.observation_manager.get_term_cfg("critic", "dr_params")
        dr = cfg.func(env, **cfg.params)[:, [1, 4, 5]]
        return torch.cat(
            (
                reader.palm_pos - env.scene.env_origins,
                reader.palm_rot.flatten(-2),
                runtime.pert_force_dir,
                runtime.pert_velocity_cache,
                dr,
            ),
            -1,
        ).clone()

    def frame(self, x, memory, auxiliary):
        """Match the original critic's term frames, including frames and scaling."""
        position, rotation, direction, perturbation, dr = auxiliary.split(
            (3, 9, 3, 6, 3), -1
        )
        rotation = rotation.reshape(-1, 3, 3)
        q, qd, _, _, v, w = unpack(x)
        policy = self.mdp.observation(x, memory)
        bodies = self.mdp.kin.forward(q, qd)[0][:, self.tip_ids]
        tips = (
            mv(rotation[:, None], bodies) + position[:, None] - self.tip_offset
        ).flatten(-2)
        base = torch.cat(
            (
                policy,
                q - self.mdp.default,
                qd - self.default_velocity,
                tips * self.tip_scale,
            ),
            -1,
        )
        progress = torch.cat(
            (
                memory["hold"] / self.mdp.hold_steps,
                memory["reward_window"] / self.mdp.switch_steps,
                memory["age"] / self.episode_steps,
                memory["count"],
            ),
            -1,
        )
        current = torch.cat(
            (
                policy[:, 40:49],
                mv(rotation, v),
                mv(rotation, w),
                direction,
                perturbation,
                progress,
                (memory["cage"] / self.mdp.max_outside).clamp(max=1),
                dr,
                rotation[:, 1:, :].flatten(-2),
            ),
            -1,
        )
        return torch.cat((base, current), -1)

    def context(self, x, memory, auxiliary, simulator_context):
        return torch.cat((simulator_context, self.frame(x, memory, auxiliary)), -1)

    def dynamics(self, x, simulator_context, disturbance):
        """Dynamics-only features: physics state plus startup and exogenous context."""
        q, qd, position, rotation, velocity, angular_velocity = unpack(x)
        return torch.cat(
            (
                q,
                qd,
                position,
                rotation.flatten(-2),
                velocity,
                angular_velocity,
                simulator_context,
                disturbance,
            ),
            -1,
        )

    @staticmethod
    def critic(context):
        # Current task frame followed by simulator parameters, with no time axis.
        return torch.cat((context[..., -FRAME_DIM:], context[..., :-FRAME_DIM]), -1)
