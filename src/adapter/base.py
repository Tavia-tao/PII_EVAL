from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.schema import Block, Prediction, Sample


@dataclass(frozen=True)
class Overrides:
    """仅归因实验可注入上游金标；常规推理不可见 gold。"""

    ocr_text: str | None = None
    layout_blocks: list[Block] | None = None


class SUTAdapter(ABC):
    @abstractmethod
    async def predict(self, sample: Sample, overrides: Overrides | None = None) -> Prediction:
        raise NotImplementedError
