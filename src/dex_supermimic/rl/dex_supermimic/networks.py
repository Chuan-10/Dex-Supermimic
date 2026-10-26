"""Compact dynamics and an independent real-termination/failure head."""

import torch
from torch import nn

from .common import RunningNorm, mlp
from .state import DELTA_DIM, STATE_DIM, integrate


class WorldModel(nn.Module):
    def __init__(
        self, dynamics_dim, contact_dim, hidden=(2048, 2048, 2048), amp=True
    ):
        super().__init__()
        self.contact_dim = contact_dim
        self.amp = amp
        # Dynamics context contains physical state, startup simulator parameters
        # and the measured/sampled transition disturbance. Task command state is
        # deliberately absent: it is recursively derived after each WM step.
        inp = dynamics_dim + 20
        self.norm = RunningNorm(inp)
        self.net = mlp(
            inp,
            list(hidden),
            DELTA_DIM + contact_dim,
            act=nn.SiLU,
            layer_norm=True,
        )
        self.delta_norm = RunningNorm(DELTA_DIM, eps=1e-4)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[DELTA_DIM:] = -3.0
        self.register_buffer(
            "error_floor",
            torch.tensor(
                [0.003] * 20
                + [0.05] * 20
                + [0.001] * 3
                + [0.005] * 3
                + [0.02] * 3
                + [0.05] * 3
            ),
        )

    def inputs(self, dynamics, u):
        return torch.cat((dynamics, u), -1)

    def forward(self, x, u, dynamics):
        features = self.norm(self.inputs(dynamics, u))
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda
        ):
            y = self.net(features)
        y = y.float()
        d1, c = y.split((DELTA_DIM, self.contact_dim), -1)
        scale = self.delta_norm.std()
        return (
            integrate(x, d1 * scale + self.delta_norm.mean),
            c,
        )


class FailureHead(nn.Module):
    """Dreamer-style continue head: P(real termination entering this imagined state).

    It consumes the same privileged task/critic features built from either true
    simulator state or the world model's predicted ``x1``; no simulator state is
    queried inside imagination.
    """

    def __init__(self, context_dim):
        super().__init__()
        dim = STATE_DIM + context_dim
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, [256, 128], 1)

    def features(self, x1, context):
        return torch.cat((x1, context), -1)

    def forward(self, features):
        return self.net(self.norm(features)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim=207):
        super().__init__()
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, [512, 256, 128], 20)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs):
        return torch.tanh(self.net(self.norm(obs)))


class Critic(nn.Module):
    def __init__(self, dim, amp=True):
        super().__init__()
        self.amp = amp
        self.norm = RunningNorm(dim)
        self.net = mlp(dim, [512, 256, 128], 1)

    def forward(self, x):
        features = self.norm(x)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp and x.is_cuda):
            return self.net(features).float().squeeze(-1)
