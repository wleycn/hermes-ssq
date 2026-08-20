"""不确定性量化套件（研究简报 2026-08-17 / 2026-08-18）。
- conformal_predict: C1 集合覆盖率（接 select_numbers 上游）
- edl_probe        : C2 EDL 区分度先验实验
"""
from ml.conformal.conformal_predict import (
    ConformalSet,
    build_from_history,
    summarize_coverage,
)
from ml.conformal.edl_probe import (
    run_edl_experiment,
    summarize_edl,
)

__all__ = [
    "ConformalSet",
    "build_from_history",
    "summarize_coverage",
    "run_edl_experiment",
    "summarize_edl",
]
