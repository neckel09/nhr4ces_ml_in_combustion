from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class FNNConfig:
    input_dim: int
    output_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"          # "relu" | "tanh" | "silu" | "elu" | "gelu"

def _get_activation(name: str) -> nn.Module:
    activations: dict[str, nn.Module] = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(activations)}.")
    return activations[name]

class FNN(nn.Module):
    """
    Feedforward Neural Network for non-linear regression.
    """

    def __init__(self, cfg: FNNConfig) -> None:
        super().__init__()

        dims: list[int] = [cfg.input_dim] + cfg.hidden_dims + [cfg.output_dim]
        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(_get_activation(cfg.activation))

        self.net: nn.Sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)