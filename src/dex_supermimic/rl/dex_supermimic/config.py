"""Configuration shared by the runner and its CLI."""

import math
from dataclasses import dataclass, asdict

from dex_supermimic.tasks.dex_reorient.config.task import DEFAULT_TASK, TaskName


@dataclass(kw_only=True)
class DexSuperMimicCfg:
    task: TaskName = DEFAULT_TASK
    """Robot/task to train. Existing checkpoints default to Hand 2."""
    seed: int = 42
    logger: str = "tensorboard"
    """Metric backend: wandb, tensorboard, or none (default tensorboard)."""
    wandb_project: str = "dex-supermimic"
    """W&B project for Dex SuperMimic runs."""
    wandb_entity: str = "cumulusalpha"
    """W&B entity/team for Dex SuperMimic runs."""
    wandb_mode: str | None = None
    """Optional W&B mode passed as WANDB_MODE, e.g. offline or disabled."""
    device: str = "cuda:0"
    num_envs: int = 1024
    control_hz: int = 50
    max_iterations: int = 8_000
    buffer_steps: int = 1000
    warmup_steps: int = 64
    utd: float = 0.4
    """Training iterations per vectorized simulator step."""
    wm_only_iters: int = 500
    wm_window: int = 8
    """Fixed multi-step world-model supervision window, matching SuperMimic."""
    wm_failure_frac: float = 0.25
    wm_failure_horizon: int = 16
    final_state_weight: float = 1.0
    contact_weight: float = 0.1
    model_checkpoint: str | None = None
    reset_model_statistics: bool = False
    variant: str = "fixed"
    wm_batches: int = 10
    wm_batch: int = 2048
    wm_hidden: tuple[int, ...] = (2048, 2048, 2048)
    pi_batch: int = 1024
    horizon: int = 32
    max_horizon: int = 32
    horizon_ramp_iters: int = 1000
    critic_epochs: int = 5
    critic_minibatches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 2e-4
    lr_decay_iters: int = 250_000
    lr_final_ratio: float = 0.03
    action_noise: float = 0.1
    failure_warmup_iters: int = 300
    failure_head: bool = True
    failure_clip: float = 0.1
    amp: bool = True
    compile: bool = True
    compile_policy: bool = True
    """Compile imagination-side task functions. Set false if startup compilation is too slow."""
    compile_policy_dynamic: bool = False
    """Use dynamic shapes for imagination compilation (lower startup cost, less specialization)."""
    profile: bool = False
    """Log cumulative collect/world-model/policy wall time and phase fractions."""
    save_interval: int = 2000
    log_interval: int = 50
    holdout_fraction: float = 0.1
    log_dir: str = "logs/dex_supermimic"
    checkpoint: str | None = None

    def __post_init__(self):
        if self.task not in ("hand1-reorient", "hand2-reorient"):
            raise ValueError(f"Unknown task: {self.task!r}")
        if self.logger not in ("wandb", "tensorboard", "none"):
            raise ValueError("logger must be wandb, tensorboard, or none")
        if self.logger != "wandb" and self.wandb_mode is not None:
            raise ValueError("wandb_mode requires logger=wandb")
        if any(
            v < 0 for v in (self.final_state_weight, self.contact_weight)
        ):
            raise ValueError("Loss weights must be nonnegative")
        if self.variant not in ("fixed", "legacy"):
            raise ValueError("variant must be fixed or legacy")
        if self.checkpoint and self.model_checkpoint:
            raise ValueError("Choose resume or model-only initialization")
        if not 0 <= self.wm_failure_frac < 1:
            raise ValueError("Invalid failure sampling mixture")
        if not math.isfinite(self.utd) or not 0 < self.utd <= 1:
            raise ValueError("utd must be in (0, 1]")
        if self.lr <= 0 or not 0 < self.lr_final_ratio <= 1:
            raise ValueError("Invalid learning-rate schedule")
        if self.control_hz not in (20, 50):
            raise ValueError("control_hz must be 20 or 50")
        if not (0 <= self.holdout_fraction < 1):
            raise ValueError("holdout_fraction must be in [0, 1)")
        if self.num_envs < 2 or self.buffer_steps < self.wm_window or self.warmup_steps < max(self.horizon, self.wm_window):
            raise ValueError("Need >=2 envs, buffer >= wm_window and warmup >= max(horizon, wm_window)")
        for value in (
            self.wm_window,
            self.wm_failure_horizon,
            self.lr_decay_iters,
            self.wm_batches,
            self.wm_batch,
            self.pi_batch,
            self.horizon,
            self.max_horizon,
            self.critic_epochs,
            self.critic_minibatches,
            self.save_interval,
            self.log_interval,
        ):
            if value < 1:
                raise ValueError("Batch sizes, horizons and intervals must be positive")
        if self.max_horizon < self.horizon:
            raise ValueError("max_horizon must be >= horizon")
        if self.critic_minibatches > self.horizon * self.pi_batch:
            raise ValueError("critic_minibatches exceeds the imagined transition count")
        if self.failure_clip <= 0 or self.failure_clip > 1:
            raise ValueError("failure_clip must be in (0, 1]")

    def to_dict(self):
        return asdict(self)
