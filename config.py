from dataclasses import dataclass


@dataclass
class Config:
    target_sr: int = 16000

    compute_intrusive_metrics: bool = False
    
    compute_non_intrusive_metrics: bool = False

    target_lufs: float = -23.0