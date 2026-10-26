"""Pen tracking adapter: exact reference playback and differentiable task terms.

The simulator retains upstream RSI, adaptive sampling, camera corruption and DR.
Imagination replays the deterministic reference and maintains EMA/load/camera state.
External wrenches are held at the sampled replay value during imagination.
"""

from typing import ClassVar

import torch

from .kinematics import HandKinematics
from .mdp import DexMDP
from .rotations import axis_angle_to_mat, mm, quat_to_mat
from .sim import SimReader
from .state import MEMORY_DIMS, unpack

PEN_MEMORY_DIMS = dict(
    MEMORY_DIMS,
    cage=20,
    time=1,
    start=1,
    end=1,
    limits=20,
    camera_raw=6,
    camera_obs=6,
    camera_holding=1,
    camera_ticks=1,
    camera_bias_pos=3,
    camera_bias_rot=9,
    encoder_bias=20,
    frozen_target=20,
    mask=20,
    mask_remaining=1,
    mask_active=1,
)


def flatten_pen_memory(memory):
    return torch.cat([memory[key] for key in PEN_MEMORY_DIMS], -1)


def unflatten_pen_memory(value):
    return dict(zip(PEN_MEMORY_DIMS, value.split(tuple(PEN_MEMORY_DIMS.values()), -1)))


class PenMDP(DexMDP):
    def __init__(self, reader):
        robot, env = reader.robot, reader.env
        data = robot.data
        super().__init__(
            reader.kin,
            data.default_joint_pos[0].clone(),
            data.joint_pos_limits[0, :, 0].clone(),
            data.joint_pos_limits[0, :, 1].clone(),
            torch.zeros(3, device=env.device),
            torch.eye(3, device=env.device),
            50,
        )
        self.motion = env.command_manager.get_term("motion")
        self.alpha = env.cfg.actions["joint_pos"].alpha
        self.link_ids = [reader.kin.names.index("robot/" + name) for name in self.motion.cfg.link_names]
        self.nominal_limits = torch.as_tensor(
            env.sim.mj_model.actuator_forcerange[robot.indexing.ctrl_ids.cpu().numpy(), 1].copy(),
            device=env.device,
            dtype=torch.float32,
        )
        self.radius = self.motion.cfg.axis_point_radius_m
        self.bound = self.motion.cfg.object_position_tanh_scale_m
        self.camera_enabled = env.cfg.observations["policy"].enable_corruption
        self.mask_cfg = env.cfg.actions["joint_pos"]
        self.env = env
        import hashlib

        digest = hashlib.sha256()
        for field in ("joint_pos", "object_pose", "link_pos"):
            digest.update(self.motion._frames[field].cpu().numpy().tobytes())
        digest.update(self.motion.group.trajectory_end.cpu().numpy().tobytes())
        self.reference_signature = digest.hexdigest()

    def control_target(self, action, memory, bias):
        residual = self.alpha * (0.5 * action).clamp(-0.6, 0.6) + (1 - self.alpha) * memory["c1"]
        reference = self.reference(memory)[0]
        target = reference + residual - bias
        cfg = self.mask_cfg
        if cfg.mask_prob > 0:
            from dex_supermimic.tasks.pen_spin.mdp.curriculum_utils import (
                get_curriculum_value,
            )

            scale = get_curriculum_value(self.env, cfg.mask_curriculum_term, 0.0) if cfg.mask_curriculum_term else 1.0
            active_at_start = memory["mask_active"] > 0
            remaining = memory["mask_remaining"] - active_at_start.float()
            active = active_at_start & (remaining > 0)
            fire = (~active_at_start) & (torch.rand_like(remaining) < cfg.mask_prob * scale)
            chosen = torch.rand_like(action).topk(cfg.mask_num_dofs, dim=-1).indices
            selected = torch.zeros_like(action).scatter(-1, chosen, 1)
            mask = torch.where(fire, selected, memory["mask"] * active.float())
            maximum = max(
                round(cfg.mask_duration_min + (cfg.mask_duration_max - cfg.mask_duration_min) * scale),
                cfg.mask_duration_min,
            )
            duration = torch.randint(cfg.mask_duration_min, maximum + 1, remaining.shape, device=action.device).float()
            remaining = torch.where(fire, duration, remaining)
            active = active | fire
            target = torch.where((mask > 0) & (memory["age"] > 0), memory["frozen_target"], target)
            memory.update(mask=mask, mask_remaining=remaining, mask_active=active.float())
        memory["frozen_target"] = target
        return residual, target

    def reference(self, memory):
        frame = (memory["start"] + memory["time"].clamp_min(0)).minimum(memory["end"] - 1).long().squeeze(-1)
        frames = self.motion._frames
        return frames["joint_pos"][frame], frames["object_pose"][frame], frames["link_pos"][frame]

    def points(self, position, rotation, bounded=False):
        offset = rotation[..., :, 0] * self.radius
        points = torch.stack((position + offset, position - offset), -2)
        return self.bound * torch.tanh(points / self.bound) if bounded else points

    def reference_points(self, pose, bounded=False):
        return self.points(pose[..., :3], quat_to_mat(pose[..., 3:]), bounded)

    def command_step(self, x, memory, next_goal, soft=False):
        return dict(memory, time=memory["time"] + 1)

    def pd_torque(self, x, u, context=None):
        q, qd, *_ = unpack(x)
        kp = self.kp_default * context[..., -40:-20]
        kd = self.kd_default * context[..., -20:]
        limits = context[..., self.limit_slice]
        return (kp * (u - q) + kd * qd).clamp(-limits, limits)

    def rewards(self, x1, memory, action, residual, contact, torque, soft=False):
        q, qd, pos, rot, *_ = unpack(x1)
        ref_q, ref_pose, ref_links = self.reference(memory)
        links = self.kin.forward(q, qd)[0][..., self.link_ids, :]
        point_error = (self.points(pos, rot) - self.reference_points(ref_pose)).norm(dim=-1).mean(-1)
        load = (torque.abs() / memory["limits"].clamp_min(1e-8)).clamp(max=1)
        charge = (memory["cage"] + self.dt * (load - memory["cage"] / 2)).clamp(0, 1)
        terms = {
            "object_pos": 2 * torch.exp(-((pos - ref_pose[..., :3]).norm(dim=-1) / 0.05).square()),
            "object_rot": 8 * torch.exp(-(point_error / 0.04).square()),
            "link_pos": torch.exp(-((links - ref_links).norm(dim=-1) / 0.025).square()).mean(-1),
            "joint_pos": torch.exp(
                -((q - ref_q).abs() / (self.upper - self.lower).clamp_min(1e-6) / 0.1).square()
            ).mean(-1),
            "action_rate": -0.1
            * ((action - memory["a1"]).square() + (action - 2 * memory["a1"] + memory["a2"]).square()).sum(-1),
            "torque": -(torque / self.nominal_limits).square().sum(-1),
            "overcurrent": -4 * (charge - 0.8).clamp_min(0).sum(-1) / 0.2,
            "finger_collision": -contact.sum(-1),
        }
        terms = {name: value * self.dt for name, value in terms.items()}
        return sum(terms.values()), charge, terms

    def termination(self, x, memory):
        _, _, pos, rot, *_ = unpack(x)
        _, ref, _ = self.reference(memory)
        position = torch.sigmoid(((pos - ref[..., :3]).norm(dim=-1) - 0.05) / 0.002)
        cosine = (rot[..., :, 0] * quat_to_mat(ref[..., 3:])[..., :, 0]).sum(-1)
        orientation = torch.sigmoid((0.5 - cosine) / 0.02)
        return 1 - (1 - position) * (1 - orientation)

    def observation(self, x, memory):
        q, qd, pos, rot, *_ = unpack(x)
        ref_q, ref_pose, ref_links = self.reference(memory)
        command = torch.cat((ref_q, self.reference_points(ref_pose, True).flatten(-2), ref_links.flatten(-2)), -1)
        return torch.cat(
            (command, q - self.default, qd * 0.25, self.points(pos, rot, True).flatten(-2), memory["a1"]), -1
        )

    def policy_observation(self, x, memory, auxiliary):
        obs = self.observation(x, memory)
        if not self.camera_enabled:
            return obs
        q, qd, pos, rot, *_ = unpack(x)
        bias_rot = memory["camera_bias_rot"].reshape(-1, 3, 3)
        axis = torch.nn.functional.normalize(torch.randn_like(pos), dim=-1)
        angle = pos.new_empty(pos.shape[0]).uniform_(-0.015, 0.015)
        rotation = mm(axis_angle_to_mat(axis, angle), mm(bias_rot, rot))
        position = pos + memory["camera_bias_pos"] + torch.empty_like(pos).uniform_(-0.002, 0.002)
        raw = self.points(position, rotation, True).flatten(-2)
        roll = torch.rand_like(memory["camera_holding"])
        holding = torch.where(memory["camera_holding"] > 0, (roll < 0.78) & (memory["camera_ticks"] < 40), roll < 0.03)
        camera = torch.where(holding, memory["camera_obs"], memory["camera_raw"])
        memory.update(
            camera_raw=raw,
            camera_obs=camera,
            camera_holding=holding.float(),
            camera_ticks=torch.where(holding, memory["camera_ticks"] + 1, 0),
        )
        return torch.cat(
            (
                obs[..., :89],
                q - self.default + memory["encoder_bias"] + torch.empty_like(q).uniform_(-0.015, 0.015),
                qd * 0.25 + torch.empty_like(qd).uniform_(-0.05, 0.05),
                camera,
                memory["a1"],
            ),
            -1,
        )

    @staticmethod
    def flatten_history(history):
        return history[:, -1]


class PenSimReader(SimReader):
    def __init__(self, env):
        self.env, self.device = env, env.device
        self.robot, self.obj = env.scene["robot"], env.scene["object"]
        self.tip_ids = self.robot.find_sites(".*_finger[1-5]_tip")[0]
        self.kin = HandKinematics(
            env.sim.mj_model,
            self.robot.indexing.body_ids.cpu().tolist(),
            self.robot.indexing.joint_ids.cpu().tolist(),
            self.robot.indexing.site_ids[self.tip_ids].cpu().tolist(),
            env.device,
        )
        self.refresh_frame()
        self.mdp = PenMDP(self)
        self.actuator_defaults = self._read_actuator_defaults()
        for key, attr in (("kp", "kp_default"), ("kd", "kd_default"), ("force_limit", "force_limit"), ("gear", "gear")):
            setattr(self.mdp, attr, self.actuator_defaults[key].to(self.device))
        sensor = env.scene["finger_collision"]
        names = sensor.primary_names
        self.middle = [names.index(f"right_finger{f}_link{l}_col") for f in range(1, 6) for l in (2, 3)]
        self.distal = [names.index(f"right_finger{f}_link4_col") for f in range(1, 6)]
        self.tips = [names.index(f"right_finger{f}_tip_sensor_col") for f in range(1, 6)]
        self.xi = self.context()
        self.reward_snapshot = None

    def context(self):
        base = super().context()
        base_names = self.context_names
        robot, obj = self.robot.indexing, self.obj.indexing
        values, names = [], []
        for name, ids in (
            ("body_mass", torch.cat((robot.body_ids, obj.body_ids)).long()),
            ("body_inertia", torch.cat((robot.body_ids, obj.body_ids)).long()),
            ("body_ipos", torch.cat((robot.body_ids, obj.body_ids)).long()),
            ("body_iquat", torch.cat((robot.body_ids, obj.body_ids)).long()),
            ("geom_size", self.collision_geom_ids()),
            ("geom_solref", self.collision_geom_ids()),
            ("geom_solimp", self.collision_geom_ids()),
            ("dof_armature", robot.joint_v_adr.long()),
            ("dof_damping", robot.joint_v_adr.long()),
            ("dof_frictionloss", robot.joint_v_adr.long()),
            ("actuator_forcerange", robot.ctrl_ids.long()),
        ):
            current = getattr(self.env.sim.model, name)
            default = self.env.sim.get_default_field(name)
            value = current[:, ids] if current.ndim == default.ndim + 1 else current[ids][None]
            value = value.expand(self.env.num_envs, *value.shape[1:]).flatten(1).clone()
            if name == "actuator_forcerange":
                # Store upper symmetric torque caps only.
                value = value[:, 1::2]
                start = sum(v.shape[-1] for v in values)
                self.mdp.limit_slice = slice(start, start + 20)
            values.append(value)
            names.append((name, value.shape[-1]))
        self.context_names = names + base_names
        return torch.cat((*values, base), -1)

    def memory(self):
        env, motion = self.env, self.mdp.motion
        act = env.action_manager.get_term("joint_pos")
        scalar = lambda x: x.float().unsqueeze(-1).clone()
        z = torch.zeros(env.num_envs, 1, device=self.device)
        out = {k: z.expand(-1, d).clone() for k, d in PEN_MEMORY_DIMS.items()}
        out.update(
            c1=act.processed_action.clone(),
            c2=act.processed_action.clone(),
            a1=env.action_manager.action.clone(),
            a2=env.action_manager.prev_action.clone(),
            goal=torch.eye(3, device=self.device).flatten()[None].expand(env.num_envs, -1),
            cage=env.reward_manager.get_term_cfg("overcurrent").func._charge.clone(),
            age=scalar(env.episode_length_buf),
            time=scalar(motion.time_steps),
            start=scalar(motion.trajectory_starts),
            end=scalar(motion.trajectory_ends),
            limits=self.xi[:, self.mdp.limit_slice].clone(),
            encoder_bias=self.robot.data.encoder_bias.clone(),
            frozen_target=act._targets.clone(),
            mask=act._mask_dofs.float().clone(),
            mask_remaining=scalar(act._mask_remaining),
            mask_active=scalar(act._mask_active),
        )
        camera = env.observation_manager.get_term_cfg("policy", "object_pose").func
        if hasattr(camera, "_history"):
            from dex_supermimic.tasks.pen_spin.mdp.observations import (
                bounded_object_pose_to_points_wrist_local,
            )

            def points(age):
                return (
                    bounded_object_pose_to_points_wrist_local(
                        camera._history[age], motion.axis_points, position_tanh_scale_m=self.mdp.bound
                    )
                    .flatten(1)
                    .clone()
                )

            out.update(
                camera_raw=points(torch.zeros_like(camera._age)),
                camera_obs=points(camera._age),
                camera_holding=scalar(camera._holding),
                camera_ticks=scalar(camera._hold_ticks),
                camera_bias_pos=camera._bias_pos.clone(),
                camera_bias_rot=quat_to_mat(camera._bias_quat).flatten(1),
            )
        return out

    def executed_target(self):
        return self.env.action_manager.get_term("joint_pos")._targets.clone()

    def contact(self):
        found = self.env.scene["finger_collision"].data.found > 0
        return torch.cat((found[:, self.middle], found[:, self.distal] | found[:, self.tips]), -1).float()

    def attach_capture(self):
        env = self.env
        self._original_reward = env.reward_manager.compute

        def reward(dt):
            physics, contact = self.state(), self.contact().clone()
            torque = self.robot.data.actuator_force.clone()
            nan = env.termination_manager.get_term("unstable").clone()
            drop = env.termination_manager.terminated.clone()
            result = self._original_reward(dt)
            self.reward_snapshot = {
                "physics": physics,
                "contact": contact,
                "torque": torque,
                "nan": nan,
                "drop": drop,
                "reward": result.clone(),
                "terms": {
                    name: env.reward_manager._step_reward[:, j].clone() * (dt if env.reward_manager._scale_by_dt else 1)
                    for j, name in enumerate(env.reward_manager._term_names)
                },
            }
            return result

        env.reward_manager.compute = reward

    def detach_capture(self):
        self.env.reward_manager.compute = self._original_reward


class PenObservations:
    base_terms: ClassVar[dict[str, int]] = {
        "motion": 89,
        "joint_pos": 20,
        "joint_vel": 20,
        "object_pose": 6,
        "last_action": 20,
        "link_pos": 63,
        "joint_position_error": 20,
        "link_position_error": 63,
    }

    def __init__(self, reader):
        self.reader, self.mdp = reader, reader.mdp
        self.frame_dim = sum(self.base_terms.values())
        # Charge/time/physical caps are exact task/controller metadata.
        self.context_dim = reader.xi.shape[-1] + self.frame_dim + 21

    def auxiliary(self):
        reader = self.reader
        n = reader.env.num_envs
        prefix = torch.cat((reader.palm_pos, reader.palm_rot.flatten(1), torch.zeros(n, 9, device=reader.device)), -1)
        return torch.cat(
            (
                prefix,
                reader.robot.data.body_external_wrench.flatten(1),
                reader.obj.data.body_external_wrench.flatten(1),
            ),
            -1,
        ).clone()

    @staticmethod
    def disturbance(auxiliary):
        return auxiliary[:, 21:]

    def frame(self, x, memory, auxiliary):
        q, qd, *_ = unpack(x)
        ref_q, _, ref_links = self.mdp.reference(memory)
        links = self.mdp.kin.forward(q, qd)[0][:, self.mdp.link_ids].flatten(1)
        return torch.cat((self.mdp.observation(x, memory), links, q - ref_q, links - ref_links.flatten(1)), -1)

    def context(self, x, memory, auxiliary, simulator_context):
        return torch.cat(
            (self.frame(x, memory, auxiliary), memory["cage"], memory["time"] * self.mdp.dt, simulator_context), -1
        )

    @staticmethod
    def critic(context):
        return context

    @staticmethod
    def dynamics(x, simulator_context, disturbance):
        return torch.cat((x, simulator_context, disturbance), -1)
