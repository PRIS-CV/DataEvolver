from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from ..schemas import DatasetRequest, DatasetSample


class GeneratorAdapter(ABC):
    name = "base"

    @abstractmethod
    def generate(self, request: DatasetRequest, output_root: Path) -> List[DatasetSample]:
        raise NotImplementedError
