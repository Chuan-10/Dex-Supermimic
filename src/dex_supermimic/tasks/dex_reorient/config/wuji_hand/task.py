"""SuperMimic recipe for wuji_hand cube reorientation."""

from ..recipe import configure_supermimic
from .env_cfgs import wuji_hand_reorient_env_cfg


def make_env_cfg(num_envs=1024, control_hz=50, play=False):
    return configure_supermimic(
        wuji_hand_reorient_env_cfg(num_envs=num_envs, play=play),
        num_envs,
        control_hz,
    )
