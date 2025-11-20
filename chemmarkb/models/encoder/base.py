import abc
from typing import TYPE_CHECKING
import torch
from torch import nn

from ...data.common import ProjectionBatch


class BaseEncoder(nn.Module, abc.ABC):

    @abc.abstractmethod
    def forward(
        self, 
        batch: ProjectionBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:

        ...

    @property
    @abc.abstractmethod
    def dim(self) -> int:

        ...

    if TYPE_CHECKING:
        def __call__(
            self, 
            batch: ProjectionBatch
        ) -> tuple[torch.Tensor, torch.Tensor]: ...