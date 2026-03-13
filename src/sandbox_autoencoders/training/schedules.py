from __future__ import annotations


def beta_for_epoch(epoch: int, total_epochs: int, beta: float, schedule: str, warmup_epochs: int) -> float:
    if schedule == "constant":
        return beta
    if schedule == "warmup":
        if warmup_epochs <= 0:
            return beta
        return beta * min(1.0, float(epoch + 1) / float(warmup_epochs))
    if schedule == "cyclical":
        cycle = max(1, total_epochs // 3)
        progress = ((epoch % cycle) + 1) / cycle
        return beta * progress
    raise ValueError(f"Unknown schedule: {schedule}")
