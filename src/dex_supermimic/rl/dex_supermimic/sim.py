"""State/context readers and exact reward-phase capture for mjlab 1.6.0."""

import torch
from .rotations import quat_to_mat, mm, mv
from .state import pack
from .kinematics import HandKinematics
from .mdp import DexMDP
from .observations import DR_PARAM_NAMES


class SimReader:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.obj = env.scene["object"]
        self.device = env.device
        self.kin = HandKinematics(
            env.sim.mj_model,
            self.robot.indexing.body_ids.cpu().tolist(),
            self.robot.indexing.joint_ids.cpu().tolist(),
            self.robot.indexing.site_ids[self.robot.find_sites(".*_finger[1-5]_tip")[0]].cpu().tolist(),
            env.device,
        )
        self.tip_ids = self.robot.find_sites(".*_finger[1-5]_tip")[0]
        tag = self.robot.find_sites(".*_wrist_tag")[0][0]
        root = self.robot.data.root_link_pose_w
        self.palm_pos = root[:, :3].clone()
        self.palm_rot = quat_to_mat(root[:, 3:])
        tag_pose = self.robot.data.site_pose_w[:, tag]
        tag_p = mv(self.palm_rot[0].T, tag_pose[0, :3] - root[0, :3])
        tag_r = mm(self.palm_rot[0].T, quat_to_mat(tag_pose[0, 3:]))
        data = self.robot.data
        self.mdp = DexMDP(
            self.kin,
            data.default_joint_pos[0].clone(),
            data.soft_joint_pos_limits[0, :, 0].clone(),
            data.soft_joint_pos_limits[0, :, 1].clone(),
            tag_p,
            tag_r,
            round(1 / env.step_dt),
            getattr(data, "actuator_effort_limit", None),
            cage_up_axis=env.reward_manager.get_term_cfg("cage_escape").params["up_axis"],
        )
        if self.mdp.effort_limit is not None:
            self.mdp.effort_limit = self.mdp.effort_limit[0].clone()
        self.actuator_defaults = self._read_actuator_defaults()
        self.mdp.kp_default = self.actuator_defaults["kp"].to(self.device)
        self.mdp.kd_default = self.actuator_defaults["kd"].to(self.device)
        self.mdp.force_limit = self.actuator_defaults["force_limit"].to(self.device)
        self.mdp.gear = self.actuator_defaults["gear"].to(self.device)
        # Reuse the original geometry->five-finger contact mapping.
        from dex_supermimic.tasks.dex_reorient.mdp.runtime_state import (
            get_fingertip_contact_binding,
        )
        from mjlab.managers import SceneEntityCfg

        tip_cfg = SceneEntityCfg("robot", site_names=(".*_finger[1-5]_tip",))
        tip_cfg.site_ids = self.tip_ids
        self.tip_binding = get_fingertip_contact_binding(env, tip_cfg, SceneEntityCfg("tip_object_contact"))
        self.context_names = []
        self.xi = self.context()
        self.reward_snapshot = None

    def _read_actuator_defaults(self):
        actuator_ids = self.robot.indexing.ctrl_ids.long()
        model = self.env.sim.model
        gainprm = self.env.sim.get_default_field("actuator_gainprm")
        biasprm = self.env.sim.get_default_field("actuator_biasprm")
        kp = gainprm[actuator_ids, 0] if gainprm.ndim == 2 else gainprm[0, actuator_ids, 0]
        kd = biasprm[actuator_ids, 2] if biasprm.ndim == 2 else biasprm[0, actuator_ids, 2]
        # Joint effort limits are shared across environments in the current task.
        force = model.actuator_forcerange
        force = force[:, actuator_ids, :] if force.ndim == 3 else force[actuator_ids, :]
        force_limit = force.abs().amax(-1)
        if force_limit.ndim == 2:
            force_limit = force_limit[0]
        if hasattr(model, "actuator_gear"):
            gear_values = model.actuator_gear
            gear_values = gear_values[:, actuator_ids, :] if gear_values.ndim == 3 else gear_values[actuator_ids, :]
            gear = gear_values[..., 0]
            if gear.ndim == 2:
                gear = gear[0]
        else:
            gear = torch.ones_like(kp)
        return dict(kp=kp.float(), kd=kd.float(), force_limit=force_limit.float(), gear=gear.float())

    def refresh_frame(self):
        root = self.robot.data.root_link_pose_w
        self.palm_pos = root[:, :3].clone()
        self.palm_rot = quat_to_mat(root[:, 3:])

    def gravity(self):
        return mv(self.palm_rot.transpose(-1, -2), self.palm_pos.new_tensor([0, 0, -9.81]))

    def state(self):
        d = self.env.sim.data
        idx = self.obj.indexing
        fq = idx.free_joint_q_adr.long()
        fv = idx.free_joint_v_adr.long()
        obj_q = d.qpos[:, fq]
        obj_v = d.qvel[:, fv]
        r = quat_to_mat(obj_q[:, 3:7])
        inv = self.palm_rot.transpose(-1, -2)
        return pack(
            d.qpos[:, self.robot.indexing.joint_q_adr.long()].clone(),
            d.qvel[:, self.robot.indexing.joint_v_adr.long()].clone(),
            mv(inv, obj_q[:, :3] - self.palm_pos),
            mm(inv, r),
            mv(inv, obj_v[:, :3]),
            mv(inv, mv(r, obj_v[:, 3:6])),
        )

    def memory(self):
        env = self.env
        cmd = env.command_manager.get_term("reorient_command")
        act = env.action_manager.get_term("joint_pos")
        # After process_action, prev_action is a_{t-1}; at the next step boundary
        # action is a_t and prev_action is a_{t-1}.
        scalar = lambda x: x.float().unsqueeze(-1).clone()
        cage = getattr(env, "_cage_penalty_counter", torch.zeros(env.num_envs, device=env.device))
        hf = env.reward_manager.get_term_cfg("action_hf").func
        c1 = act.processed_action.clone()
        c2 = hf._c2.clone() if getattr(hf, "_c2", None) is not None else c1.clone()
        return dict(
            c1=c1,
            c2=c2,
            a1=env.action_manager.action.clone(),
            a2=env.action_manager.prev_action.clone(),
            goal=mm(self.palm_rot.transpose(-1, -2), quat_to_mat(cmd.goal_quat)).flatten(-2),
            hold=scalar(cmd.hold_counter),
            goal_timer=scalar(cmd.goal_timer),
            window=scalar(cmd.in_success_window),
            window_timer=scalar(cmd.window_timer),
            reward_window=scalar(cmd.reward_window_timer),
            reward_hold=scalar(cmd.reward_hold_counter),
            reward_snapshot=scalar(cmd.reward_hold_counter_snapshot),
            within=scalar(cmd.within_threshold),
            cage=scalar(cage),
            age=scalar(env.episode_length_buf),
            count=scalar(cmd.goal_reach_count),
        )

    def contact(self):
        scene = self.env.scene
        tips = self.tip_binding.contact_mask(scene["tip_object_contact"].data.found > 0).float()
        palm = (scene["palm_object_found"].data.found > 0).any(-1, keepdim=True).float()
        distal = (scene["distal_finger_object_found"].data.found > 0).any(-1, keepdim=True).float()
        collision = (scene["finger_collision"].data.found > 0).float().flatten(1)
        return torch.cat((tips, palm, distal, collision), -1)

    def collision_geom_ids(self):
        """Return active robot/object collision geoms used by the friction context.

        Visual-only geoms, collision meshes disabled by the Hand 2 overlay, the
        calibration-rig table, and the non-colliding mount are filtered out.
        """
        ids = torch.cat(
            (self.robot.indexing.geom_ids, self.obj.indexing.geom_ids)
        ).long()
        contype = torch.as_tensor(self.env.sim.model.geom_contype).long()
        conaffinity = torch.as_tensor(self.env.sim.model.geom_conaffinity).long()
        active = (contype[ids] != 0) | (conaffinity[ids] != 0)
        return ids[active]

    def context(self):
        """Read the reference DR summary and actuator context used by all modules."""
        model = self.env.sim.model
        N = self.env.num_envs
        values = []
        self.context_names = []
        # Include the complete reference-task DR summary. Its friction mean
        # replaces the former vector of per-geom friction ratios.
        dr_cfg = self.env.observation_manager.get_term_cfg("critic", "dr_params")
        dr_params = dr_cfg.func(self.env, **dr_cfg.params)
        if dr_params.shape[-1] != len(DR_PARAM_NAMES):
            raise ValueError(
                f"Expected {len(DR_PARAM_NAMES)} reference DR ratios, got "
                f"{dr_params.shape[-1]}"
            )
        for name, value in zip(
            DR_PARAM_NAMES,
            dr_params.unbind(-1),
            strict=True,
        ):
            values.append(value.reshape(N, 1))
            self.context_names.append((name, 1))

        # Exact per-joint gains remain at the end for analytic torque reconstruction.
        actuator_ids = self.robot.indexing.ctrl_ids.long()
        default_gainprm = self.env.sim.get_default_field("actuator_gainprm")
        default_biasprm = self.env.sim.get_default_field("actuator_biasprm")
        if (
            actuator_ids.numel() == 0
            or default_gainprm.shape[-1] < 1
            or default_biasprm.shape[-1] < 3
        ):
            raise ValueError(
                "Dex SuperMimic actuator context requires at least one "
                "actuator with gainprm[0] and biasprm[2] slots"
            )
        env_a, act_a = torch.meshgrid(
            torch.arange(N, device=self.device), actuator_ids, indexing="ij"
        )
        current_kp = model.actuator_gainprm[env_a, act_a, 0]
        default_kp = default_gainprm[actuator_ids, 0].unsqueeze(0)
        kp_ratio = current_kp / default_kp.clamp_min(1e-8)

        current_kd = model.actuator_biasprm[env_a, act_a, 2]
        default_kd = default_biasprm[actuator_ids, 2].unsqueeze(0)
        # Position actuators store -kd here. Keep the divisor negative so a
        # near-zero default cannot flip the ratio's sign.
        kd_ratio = current_kd / default_kd.clamp(max=-1e-8)
        values.append(kp_ratio)
        self.context_names.append(("actuator_kp_ratios", kp_ratio.shape[-1]))
        values.append(kd_ratio)
        self.context_names.append(("actuator_kd_ratios", kd_ratio.shape[-1]))
        return torch.cat(values, -1)

    def attach_capture(self):
        """Capture postphysics state and reward-phase labels without extra forward passes.

        No extra MuJoCo forward pass, and no changes to the original task's timing.
        """
        env = self.env
        original_step = env.sim.step
        original_reward = env.reward_manager.compute
        self._original_step = original_step
        self._original_reward = original_reward
        self._substep = 0

        def step():
            original_step()
            self._substep += 1

        def reward(dt):
            physics = self.state()
            contact = self.contact().clone()
            torque = self.robot.data.actuator_force.clone()
            cage_drop = env.termination_manager.get_term("cage_drop").clone()
            nan = env.termination_manager.get_term("nan_guard").clone()
            result = original_reward(dt)
            self.reward_snapshot = dict(
                terms={
                    name: (
                        env.reward_manager._step_reward[:, j].clone() * (dt if env.reward_manager._scale_by_dt else 1.0)
                    )
                    for j, name in enumerate(env.reward_manager._term_names)
                },
                physics=physics,
                contact=contact,
                torque=torque,
                drop=cage_drop,
                nan=nan,
                reward=result.clone(),
            )
            return result

        env.sim.step = step
        env.reward_manager.compute = reward

    def detach_capture(self):
        self.env.sim.step = self._original_step
        self.env.reward_manager.compute = self._original_reward
