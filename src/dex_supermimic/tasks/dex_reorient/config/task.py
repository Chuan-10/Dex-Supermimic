"""Select a robot-specific task without importing the reference checkout."""

from typing import Literal

TaskName = Literal["hand2-reorient", "hand1-reorient"]
DEFAULT_TASK: TaskName = "hand2-reorient"


def make_env_cfg(num_envs=1024, control_hz=50, play=False, task: TaskName = DEFAULT_TASK):
    if task == "hand1-reorient":
        from .wuji_hand.task import make_env_cfg as factory
    elif task == "hand2-reorient":
        from .wuji_hand2.task import make_env_cfg as factory
    else:
        raise ValueError(f"Unknown Dex SuperMimic task: {task!r}")
    return factory(num_envs, control_hz, play)


def checkpoint_task(config) -> TaskName:
    """Checkpoints saved before task selection belong to Hand 2."""
    task = config.get("task", DEFAULT_TASK)
    if task not in ("hand1-reorient", "hand2-reorient"):
        raise ValueError(f"Unknown checkpoint task: {task!r}")
    return task
