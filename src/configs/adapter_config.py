from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class AdapterConfig:
    method: str = "none"          # "none" | "lora" | "dora" | (future: "ia3", etc.)
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None   # None = let apply_adapter() pick sane defaults