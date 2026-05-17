from dataclasses import dataclass


@dataclass
class Config:
    target_sr: int = 16000

    compute_intrusive_metrics: bool = False
    
    compute_non_intrusive_metrics: bool = True
    
    output_csv: str | None = None

    target_lufs: float = -23.0