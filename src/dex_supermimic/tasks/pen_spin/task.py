"""Standalone Hand 2 pen-spin recipe for Dex SuperMimic."""

from pathlib import Path

from .config.wuji_hand2.env_cfgs import wuji_hand2_pen_spin_env_cfg


def make_env_cfg(num_envs=1024, control_hz=50, play=False, recipe_path=None, clips_root=None):
    if control_hz != 50:
        raise ValueError("Pen spin requires 50 Hz reference playback")
    cfg = wuji_hand2_pen_spin_env_cfg(play=play)
    cfg.scene.num_envs = num_envs
    motion = cfg.commands["motion"]
    motion.recipe_path = (
        str(Path(recipe_path).expanduser().resolve())
        if recipe_path
        else str(Path(__file__).parent / "data/recipe/03_continuous_multi_axis.json")
    )
    motion.clips_root = str(Path(clips_root).expanduser().resolve()) if clips_root else None
    # Action freezing and external wrenches are observed exogenous inputs. The
    # world model is conditioned on their realized targets/wrenches, not RNG.
    return cfg
