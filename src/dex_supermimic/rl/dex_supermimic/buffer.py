"""GPU time/environment replay with correct episode and numerical-validity masks."""

import torch


class ReplayBuffer:
    def __init__(self, capacity, num_envs, device, holdout_fraction=0.1):
        self.capacity = capacity
        self.num_envs = num_envs
        self.device = device
        self.train_end = max(1, num_envs - max(1, int(num_envs * holdout_fraction))) if holdout_fraction else num_envs
        self.data = {}
        self.ptr = 0
        self.size = 0

    def add(self, transition):
        if not self.data:
            self.data = {
                k: torch.empty((self.capacity, *v.shape), device=self.device, dtype=v.dtype)
                for k, v in transition.items()
            }
        for k, v in transition.items():
            self.data[k][self.ptr].copy_(torch.nan_to_num(v) if v.is_floating_point() else v)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch, window=1, holdout=False, stratified_fraction=0.0, generator=None,
               failure_frac=0.0, failure_horizon=16):
        if self.size < window:
            raise ValueError("Not enough replay steps")
        lo, hi = (self.train_end, self.num_envs) if holdout else (0, self.train_end)
        if lo == hi:
            raise ValueError("No held-out environments")
        count = self.size - window + 1
        nenv = hi - lo
        probability = torch.full((count * nenv,), 1 / (count * nenv), device=self.device)
        if stratified_fraction and "stratum" in self.data:
            chronological = (self.ptr - self.size + torch.arange(self.size, device=self.device)) % self.capacity
            labels = self.data["stratum"][chronological, lo:hi]
            # A window belongs to every class encountered anywhere in the window.
            membership = labels.unfold(0, window, 1).any(-1).reshape(-1, 4)
            present = membership.any(0)
            balanced = torch.zeros_like(probability)
            for category in range(4):
                pool = membership[:, category].float()
                balanced += pool / pool.sum().clamp_min(1)
            balanced /= present.sum().clamp_min(1)
            if bool(present.any()):
                probability = (1 - stratified_fraction) * probability + stratified_fraction * balanced
        ids = torch.multinomial(probability, batch, replacement=True, generator=generator)
        if failure_frac:
            if stratified_fraction:
                raise ValueError("Choose failure-window or stratified sampling")
            chronological = (self.ptr - self.size + torch.arange(self.size, device=self.device)) % self.capacity
            term = self.data["terminated"][chronological, lo:hi].int()
            cumulative = torch.nn.functional.pad(term.cumsum(0), (0, 0, 1, 0))
            starts = torch.arange(count, device=self.device)
            ends = (starts + failure_horizon).clamp(max=self.size)
            failure = ((cumulative[ends] - cumulative[starts]) > 0).flatten()
            n_failure = failure.sum()
            n_draw = int(batch * failure_frac)
            probs = torch.where(n_failure > 0, failure.float(), torch.ones_like(probability))
            if n_draw:
                ids[:n_draw] = torch.multinomial(probs, n_draw, replacement=True, generator=generator)
            ids = ids[torch.randperm(batch, device=self.device, generator=generator)]
            q = (n_draw / batch) * (n_failure > 0).float()
            probability = (1 - q) * probability + q * failure.float() / n_failure.clamp_min(1)
        t, env = ids // nenv, ids % nenv + lo
        rows = (self.ptr - self.size + t[:, None] + torch.arange(window, device=self.device)) % self.capacity
        out = {k: v[rows, env[:, None]] for k, v in self.data.items()}
        # Keep a valid terminal transition, never cross into the next episode.
        alive = torch.cat((torch.ones_like(out["done"][:, :1]), ~out["done"][:, :-1]), 1).float().cumprod(1)
        # An invalid state also poisons all later open-loop predictions.
        out["mask"] = alive * out["valid"].float().cumprod(1)
        out["importance"] = (1 / (count * nenv * probability[ids]))[:, None].expand(-1, window)
        out["env"] = env
        return out
