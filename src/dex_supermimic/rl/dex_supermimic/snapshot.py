"""Full simulator snapshots for paired real-task evaluation."""

import torch


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
