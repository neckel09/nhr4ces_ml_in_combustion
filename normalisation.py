
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset


class NormalisationInterface(ABC):
    """Abstract interface for input / output normalisation."""

    @abstractmethod
    def fit(self, data: Tensor) -> "NormalisationInterface":
        """Compute normalisation statistics from data."""
        ...

    @abstractmethod
    def transform(self, data: Tensor) -> Tensor:
        """Apply normalisation."""
        ...

    @abstractmethod
    def inverse_transform(self, data: Tensor) -> Tensor:
        """Undo normalisation."""
        ...

    def fit_transform(self, data: Tensor) -> Tensor:
        return self.fit(data).transform(data)


class StandardScaler(NormalisationInterface):
    """Zero-mean / unit-variance scaling  (z = (x - μ) / σ)."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.mean_: Optional[Tensor] = None
        self.std_: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "StandardScaler":
        self.mean_ = data.mean(dim=0)
        self.std_ = data.std(dim=0).clamp(min=self.eps)
        return self

    def transform(self, data: Tensor) -> Tensor:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first."
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data: Tensor) -> Tensor:
        assert self.mean_ is not None and self.std_ is not None, "Call fit() first."
        return data * self.std_ + self.mean_


class MinMaxScaler(NormalisationInterface):
    """Min-max scaling  (z = (x - min) / (max - min))."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self.min_: Optional[Tensor] = None
        self.range_: Optional[Tensor] = None

    def fit(self, data: Tensor) -> "MinMaxScaler":
        self.min_ = data.min(dim=0).values
        #self.range_ = (data.max(dim=0).values - self.min_).clamp(min=self.eps)
        self.range_ = (data.max(dim=0).values - self.min_)
        return self

    def transform(self, data: Tensor) -> Tensor:
        assert self.min_ is not None and self.range_ is not None, "Call fit() first."
        return (data - self.min_) / self.range_

    def inverse_transform(self, data: Tensor) -> Tensor:
        assert self.min_ is not None and self.range_ is not None, "Call fit() first."
        return data * self.range_ + self.min_


class RootScaler(NormalisationInterface):
    """Min-max scaling  (z = (x - min) / (max - min))."""

    def __init__(self, eps: float = 1e-15) -> None:
        self.eps = eps
        self.absmax_: Optional[Tensor] = None
        self.N = 5

    def fit(self, data: Tensor) -> "MinMaxScaler":
        data_ = data.detach().clone()
        self.absmax_ = data_.abs().max(dim=0).values
        data_ = data_ / self.absmax_ + self.eps
        data_ = torch.sign(data_) * torch.pow(torch.abs(data_), 1/self.N)
        self.logabsmax_ = data_.abs().max(dim=0).values
        return self

    def transform(self, data: Tensor) -> Tensor:
        assert self.absmax_ is not None, "Call fit() first."
        data = data / self.absmax_ + self.eps
        return torch.sign(data) * torch.pow(torch.abs(data), 1/self.N) / self.logabsmax_

    def inverse_transform(self, data: Tensor) -> Tensor:
        assert self.absmax_ is not None, "Call fit() first."
        return torch.sign(data) * torch.pow(torch.abs(data * self.logabsmax_), self.N) * self.absmax_


