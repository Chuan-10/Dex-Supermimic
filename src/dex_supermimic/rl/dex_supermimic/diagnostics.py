"""Fixed-data, physical-unit diagnostics and fail-closed action-direction evidence.

No normalization is updated here. Dataset tensors contain real simulator labels
and command/controller memory at every step.
"""

from pathlib import Path
import torch
from torch.nn import functional as F
from .state import difference, unpack, unflatten_memory
from .mdp import angle_error

GROUPS = ("q_rad", "qd_rad_s", "p_m", "R_rad", "v_m_s", "omega_rad_s")
SIZES = (20, 20, 3, 3, 3, 3)


def grouped_loss(prediction, target, scales, weights):
    errors = difference(prediction, target).split(SIZES, -1)
    groups = torch.stack([(e / scale).square().mean(-1) for e, scale in zip(errors, scales)], -1)
    weight = groups.new_tensor(weights)
    return (groups * weight).sum(-1) / weight.sum(), groups


def transition_strata(x, x1, contact, previous_contact, drop, cage, max_outside):
    """Overlapping stationary/moving/contact-switch/failure strata, in SI units."""
    _, qd, _, _, v, omega = unpack(x1)
    moving = (qd.square().mean(-1).sqrt() > 0.05) | (v.norm(dim=-1) > 0.01) | (omega.norm(dim=-1) > 0.1)
    switch = (contact != previous_contact).any(-1)
    failure = drop | (cage.squeeze(-1) >= max_outside * 0.5)
    return torch.stack((~moving, moving, switch, failure), -1)


def summarize(out, key, values, mask):
    values = values[mask.bool()]
    if not values.numel():
        return
    out[key + "/mean"] = values.mean().item()
    out[key + "/p95"] = values.quantile(0.95).item()


@torch.no_grad()
def evaluate_world_model(runner, dataset, horizons=(1, 4, 8, 16, 32)):
    """Open-loop action replay, with measured external pushes and recursive task state.

    The world model is treated as the simulator: only the physical start, startup
    context and measured exogenous disturbances are replayed. Command/controller
    memory advances from each predicted transition.
    """
    w = {k: v.to(runner.device) for k, v in dataset.items()}
    if not {"aux", "sim_context", "m"}.issubset(w):
        raise ValueError("Validation observation layout mismatch; regenerate the dataset")
    if "control_hz" in w and int(w["control_hz"]) != runner.cfg.control_hz:
        raise ValueError("Validation/checkpoint control frequency mismatch")
    result = {}
    for horizon in horizons:
        if horizon > w["x"].shape[1]:
            continue
        x = w["x"][:, 0]
        memory = unflatten_memory(w["m"][:, 0])
        auxiliary = w["aux"][:, 0]
        simulator = w["sim_context"] if "sim_context" in w else runner.reader.xi[w["env"]]
        for j in range(horizon):
            dynamics = runner.observations.dynamics(x, simulator, auxiliary[:, 12:21])
            x1, contact = runner.wm(x, w["u"][:, j], dynamics)
            torque = runner.mdp.pd_torque(x, w["u"][:, j], simulator)
            _, cage, _ = runner.mdp.rewards(
                x1, memory, w["action"][:, j], w["c"][:, j], contact.sigmoid(), torque, soft=True
            )
            next_goal = memory["goal"].reshape(-1, 3, 3)
            memory = runner.advance_memory(
                x1, memory, w["action"][:, j], w["c"][:, j], cage, next_goal
            )
            push = w["push"][:, j]
            disturbed = push[..., -6:-3].abs().sum(-1, keepdim=True) > 0
            velocity = push[..., -6:-3]
            direction = torch.where(
                disturbed,
                velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-8),
                velocity,
            )
            auxiliary = runner.advance_disturbance(
                auxiliary,
                disturbed,
                direction,
                velocity,
            )
            x = x1 + w["push"][:, j]
        mask = w["mask"][:, horizon - 1].bool()
        j = horizon - 1
        m = memory
        _, _, predicted_terms = runner.mdp.rewards(
            x1, m, w["action"][:, j], w["c"][:, j], contact.sigmoid(), torque
        )
        _, _, true_terms = runner.mdp.rewards(
            w["x1"][:, j],
            m,
            w["action"][:, j],
            w["c"][:, j],
            w["contact"][:, j],
            w["torque"][:, j],
        )
        metrics = {}
        for key, error in zip(GROUPS, difference(x1, w["x1"][:, j]).split(SIZES, -1)):
            metrics[f"final/{key}"] = error.norm(dim=-1) if key == "R_rad" else error.square().mean(-1).sqrt()
        _, predicted_qd, _, _, predicted_v, predicted_omega = unpack(x1)
        _, true_qd, _, _, true_v, true_omega = unpack(w["x1"][:, j])
        metrics["predicted_joint_speed_rad_s"] = predicted_qd.square().mean(-1).sqrt()
        metrics["true_joint_speed_rad_s"] = true_qd.square().mean(-1).sqrt()
        metrics["predicted_object_speed_m_s"] = predicted_v.norm(dim=-1)
        metrics["true_object_speed_m_s"] = true_v.norm(dim=-1)
        metrics["predicted_object_angular_speed_rad_s"] = predicted_omega.norm(dim=-1)
        metrics["true_object_angular_speed_rad_s"] = true_omega.norm(dim=-1)
        prob = contact.sigmoid()
        truth = w["contact"][:, j]
        metrics["contact/brier"] = (prob - truth).square().mean(-1)
        metrics["contact/error_rate"] = ((prob >= 0.5) != truth.bool()).float().mean(-1)
        metrics["contact/bce"] = F.binary_cross_entropy_with_logits(contact, truth, reduction="none").mean(-1)
        metrics["torque_Nm"] = (torque - w["torque"][:, j]).square().mean(-1).sqrt()
        for name, value in predicted_terms.items():
            metrics["reward_abs_bias/" + name] = (value - true_terms[name]).abs()
            metrics["reward_signed_bias/" + name] = value - true_terms[name]
        goal = unflatten_memory(w["m"][:, 0])["goal"].reshape(-1, 3, 3)
        initial = angle_error(unpack(w["x"][:, 0])[3], goal)
        real_progress = initial - angle_error(unpack(w["x1"][:, j])[3], goal)
        model_progress = initial - angle_error(unpack(x1)[3], goal)
        metrics["true_goal_progress_rad"] = real_progress
        metrics["model_goal_progress_rad"] = model_progress
        metrics["goal_progress_abs_bias_rad"] = (model_progress - real_progress).abs()
        categories = {"all": mask}
        if "stratum" in w:
            for k, name in enumerate(("stationary", "moving", "contact_switch", "near_failure")):
                categories[name] = mask & w["stratum"][:, :horizon, k].any(1)
        for category, selected in categories.items():
            result[f"h{horizon}/{category}/count"] = int(selected.sum())
            for key, values in metrics.items():
                summarize(result, f"h{horizon}/{category}/{key}", values, selected)
    return result


def load_dataset(path, device):
    return torch.load(Path(path), map_location=device, weights_only=True)


class HorizonGate:
    """Only explicit measured real-simulator evidence enables actor imagination.

    Evidence is tied to the current model revision. A new WM update invalidates
    it; checking a stale checkpoint cannot unlock a newly fitted model.
    """

    def __init__(self):
        self.approved = {}

    def record(self, horizon, revision, evidence, min_pairs=32, min_agreement=0.8):
        required = (
            "pairs",
            "reward_effective_pairs",
            "angle_effective_pairs",
            "reward_sign_agreement",
            "angle_sign_agreement",
            "suggested_real_reward_gain",
            "suggested_real_angle_gain",
            "restore_error",
        )
        if any(k not in evidence for k in required):
            raise ValueError("Incomplete action-direction evidence")
        passed = (
            evidence["pairs"] >= min_pairs
            and min_pairs <= evidence["reward_effective_pairs"] <= evidence["pairs"]
            and min_pairs <= evidence["angle_effective_pairs"] <= evidence["pairs"]
            and evidence["reward_sign_agreement"] >= min_agreement
            and evidence["angle_sign_agreement"] >= min_agreement
            and evidence["suggested_real_reward_gain"] > 0
            and evidence["suggested_real_angle_gain"] > 0
            and evidence["restore_error"] <= 1e-6
        )
        if passed:
            self.approved[horizon] = revision
        else:
            self.approved.pop(horizon, None)
        return passed

    def horizon(self, revision, maximum):
        return max((h for h, r in self.approved.items() if r == revision and h <= maximum), default=0)


def action_direction_check(
    snapshot, restore, real_rollout, model_rollout, actions, epsilon=0.03, response_threshold=1e-6
):
    """Paired smooth raw-action probes via a full simulator checkpoint adapter.

    snapshot/restore must include physics, solver warm start, RNG, DR, controller,
    command, observation history and task memory. real_rollout/model_rollout return
    per-environment (original_reward_sum, fixed_goal_angle_progress). The adapter
    is checked by replaying the baseline twice; the caller must provide an actual
    Warp adapter, never a state-vector-only reset.
    """
    original = snapshot()
    try:
        restore(original)
        baseline = real_rollout(actions).detach()
        restore(original)
        repeated = real_rollout(actions).detach()
        restore_error = (baseline - repeated).abs().max().item()
        raw = actions.detach().requires_grad_(True)
        predicted = model_rollout(raw)
        (gradient,) = torch.autograd.grad(predicted[..., 0].sum(), raw)

        # A temporal low-pass filter avoids injecting single-step discontinuities.
        def smooth(direction):
            filtered = direction.clone()
            for t in range(1, len(filtered)):
                filtered[t] = 0.7 * filtered[t - 1] + 0.3 * direction[t]
            return filtered / filtered.abs().amax(dim=(0, 2), keepdim=True).clamp_min(1e-8)

        direction = smooth(torch.randn_like(actions))
        suggested = smooth(gradient)
        real, model = [], []
        for a in (
            (actions + epsilon * direction).clamp(-1, 1),
            (actions - epsilon * direction).clamp(-1, 1),
            (actions + epsilon * suggested).clamp(-1, 1),
        ):
            restore(original)
            real.append(real_rollout(a).detach())
            restore(original)
            model.append(model_rollout(a).detach())
        actual_delta, predicted_delta = real[0] - real[1], model[0] - model[1]
        # Only measurable real responses belong in the denominator. A model's
        # zero/tiny/nonfinite response must not hide a measurable prediction error.
        measurable = torch.isfinite(actual_delta) & (actual_delta.abs() > response_threshold)
        correct = (
            (actual_delta.sign() == predicted_delta.sign())
            & torch.isfinite(predicted_delta)
            & (predicted_delta.abs() > response_threshold)
        )
        effective_pairs = measurable.sum(0)
        agreement = (correct & measurable).sum(0) / effective_pairs.clamp_min(1)
        gain = real[2] - baseline
        return dict(
            pairs=actions.shape[1],
            reward_effective_pairs=int(effective_pairs[0]),
            angle_effective_pairs=int(effective_pairs[1]),
            reward_sign_agreement=agreement[0].item(),
            angle_sign_agreement=agreement[1].item(),
            suggested_real_reward_gain=gain[..., 0].mean().item(),
            suggested_real_angle_gain=gain[..., 1].mean().item(),
            restore_error=restore_error,
        )
    finally:
        restore(original)


class WarpTaskSnapshot:
    """In-process transactional snapshot for mjlab 1.6 runtime objects.

    Saves every Warp model/data array (including solver warm start), reachable
    task/manager tensors and Python containers, and all random number generators.
    Restores object identities as well as values, so closures retain their state.
    This is intentionally separate from the reduced learned state representation.
    """

    def __init__(self, runner):
        import random
        import numpy as np
        import warp as wp

        self.entries, self.arrays, self.tensors = [], [], []
        self.rng = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seen = set()

        def walk(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                self.tensors.append((value, value.clone()))
            elif isinstance(value, wp.array):
                self.arrays.append((value, wp.clone(value)))
            elif isinstance(value, dict):
                self.entries.append((value, dict(value), "dict"))
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                self.entries.append((value, list(value), "list"))
                for v in value:
                    walk(v)
            elif isinstance(value, tuple):
                for v in value:
                    walk(v)
            elif hasattr(value, "__dict__") and type(value).__module__.startswith(
                ("mjlab.", "dex_supermimic.", "mujoco_warp.", "warp._src.types")
            ):
                attributes = dict(vars(value))
                # Do not descend into foreign native MuJoCo/CUDA runtime objects.
                self.entries.append((value, attributes, "object"))
                for v in attributes.values():
                    walk(v)

        walk(runner.env)
        walk(runner.reader)
        wp.synchronize()

    @torch.no_grad()
    def restore(self):
        import random
        import numpy as np
        import warp as wp

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for obj, contents, kind in self.entries:
            if kind == "dict":
                obj.clear()
                obj.update(contents)
            elif kind == "list":
                obj[:] = contents
            else:
                vars(obj).clear()
                vars(obj).update(contents)
        for target, saved in self.arrays:
            wp.copy(target, saved)
        for target, saved in self.tensors:
            # Expanded views have overlapping storage; their base is also saved.
            if not any(n > 1 and stride == 0 for n, stride in zip(target.shape, target.stride())):
                target.copy_(saved)
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        if self.rng[3] is not None:
            torch.cuda.set_rng_state_all(self.rng[3])
        wp.synchronize()


def validate_action_horizon(runner, horizon, epsilon=0.03):
    """Compare smooth raw-action probes from the identical full Warp task state."""
    from .state import ema_target

    outer = WarpTaskSnapshot(runner)
    try:
        x0 = runner.reader.state().clone()
        memory = {k: v.clone() for k, v in runner.reader.memory().items()}
        auxiliary = runner.observations.auxiliary()
        simulator = runner.reader.xi
        goal = memory["goal"].reshape(-1, 3, 3)
        initial_angle = angle_error(unpack(x0)[3], goal)
        with torch.no_grad():
            actions = runner.actor(runner.obs)[None].expand(horizon, -1, -1).clone()

        def real_rollout(sequence):
            total = x0.new_zeros(x0.shape[0])
            alive = torch.ones_like(total, dtype=torch.bool)
            final_angle = initial_angle.clone()
            with torch.no_grad():
                for a in sequence:
                    _, reward, terminated, truncated, _ = runner.env.step(a)
                    total += torch.where(alive, reward, 0.0)
                    angle = angle_error(unpack(runner.reader.reward_snapshot["physics"])[3], goal)
                    final_angle = torch.where(alive, angle, final_angle)
                    done = terminated | truncated
                    alive &= ~done
                    ids = done.nonzero().flatten()
                    if ids.numel():
                        runner.env.reset(env_ids=ids)
            return torch.stack((total, initial_angle - final_angle), -1)

        def model_rollout(sequence):
            x = x0
            m = dict(memory)
            model_auxiliary = auxiliary
            total = x0.new_zeros(x0.shape[0])
            for a in sequence:
                c, u = ema_target(
                    a,
                    m["c1"],
                    runner.mdp.default,
                    runner.mdp.lower,
                    runner.mdp.upper,
                    runner.reader.robot.data.encoder_bias,
                    runner.mdp.alpha,
                    m["age"].squeeze(-1),
                    runner.mdp.dt,
                )
                x1, contact = runner.wm(
                    x, u,
                    runner.observations.dynamics(x, simulator, model_auxiliary[:, 12:21]),
                )
                torque = runner.mdp.pd_torque(x, u, simulator)
                reward, cage, _ = runner.mdp.rewards(x1, m, a, c, contact.sigmoid(), torque, soft=True)
                total = total + reward
                updated = runner.advance_memory(x1, m, a, c, cage, goal)
                # Direction diagnostics are deterministic and do not sample interval pushes.
                m, x = updated, x1
            return torch.stack((total, initial_angle - angle_error(unpack(x)[3], goal)), -1)

        return action_direction_check(
            lambda: outer, lambda state: state.restore(), real_rollout, model_rollout, actions, epsilon
        )
    finally:
        outer.restore()


def main():
    """Re-evaluate any old/new checkpoint on exactly the same saved dataset."""
    import json
    from dataclasses import dataclass
    import tyro

    @dataclass
    class Config:
        checkpoint: str
        dataset: str
        output: str = "logs/dex_diagnostics"
        device: str = "cpu"

    cfg = tyro.cli(Config)
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.tasks.dex_reorient.config.wuji_hand2.task import make_env_cfg
    from .config import DexSuperMimicCfg
    from .runner import DexSuperMimicRunner

    checkpoint = torch.load(cfg.checkpoint, map_location=cfg.device, weights_only=False)
    dataset = load_dataset(cfg.dataset, cfg.device)
    values = {k: v for k, v in checkpoint["config"].items() if k in DexSuperMimicCfg.__dataclass_fields__}
    # Context is carried in the dataset; original replay env indices are not read.
    values.update(
        device=cfg.device,
        num_envs=2,
        checkpoint=None,
        model_checkpoint=cfg.checkpoint,
        reset_model_statistics=False,
        diagnostic_dataset=cfg.dataset,
        compile=False,
        amp=False,
        logger="none",
        wandb_mode=None,
    )
    runner_cfg = DexSuperMimicCfg(**values)
    env_cfg = make_env_cfg(runner_cfg.num_envs, runner_cfg.control_hz)
    env_cfg.seed = runner_cfg.seed
    env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
    runner = None
    try:
        runner = DexSuperMimicRunner(env, runner_cfg, cfg.output)
        result = runner.run_diagnostics()
        print(json.dumps(result, indent=2))
    finally:
        if runner:
            runner.close()
        env.close()


if __name__ == "__main__":
    main()
