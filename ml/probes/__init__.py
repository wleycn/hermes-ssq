"""随机性验证探针套件（研究简报 2026-08-18 ~ 2026-08-22）。

均为检验器（非预测器），纯 numpy/scipy 实现，可独立单测。
- surrogate_probe : P0 元验证框架（RS/AAFT/IAAFT）
- nist_probe      : P0 NIST SP 800-22 适用子集（带样本量功效 caveat）
- ordinal_probe   : P1 排列熵 / Amigó χ² i.i.d. 检验
- transfer_entropy: P1 跨球位有向依赖检验
- mfdfa_probe     : P1 多重分形去趋势波动分析（长程记忆/幂律尺度, 08-20）
- rmt_probe       : P1 随机矩阵理论特征值谱（高维谱, 08-20, 33号码×窗口规格）
- calibration_probe: P1 概率校准诊断（Brier/ECE/isotonic 监控, 08-22）
"""
from ml.probes.surrogate_probe import (
    make_surrogates,
    surrogate_zscore,
    surrogate_pvalue,
    run_surrogate_probe,
)

__all__ = [
    "make_surrogates",
    "surrogate_zscore",
    "surrogate_pvalue",
    "run_surrogate_probe",
]
