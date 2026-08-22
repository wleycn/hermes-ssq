"""随机性验证探针套件（研究简报 2026-08-18 ~ 2026-08-22）。

均为检验器（非预测器），纯 numpy/scipy 实现，可独立单测。
- surrogate_probe : P0 元验证框架（RS/AAFT/IAAFT）
- nist_probe      : P0 NIST SP 800-22 适用子集（带样本量功效 caveat）
- ordinal_probe   : P1 排列熵 / Amigó χ² i.i.d. 检验
- transfer_entropy: P1 跨球位有向依赖检验
- mfdfa_probe     : P1 多重分形去趋势波动分析（长程记忆/幂律尺度, 08-20）
- rmt_probe       : P1 随机矩阵理论特征值谱（高维谱, 08-20, 33号码×窗口规格）
- calibration_probe: P1 概率校准诊断（Brier/ECE/isotonic 监控, 08-22）
- visibility_probe: P1 可见图度分布检验（图论家族, 08-18 简报 08-22 补落地）
- lz_probe        : P1 Lempel-Ziv 可压缩性（算法信息论, 08-19 简报 08-22 补落地）
- rqa_probe       : P1 递归量化分析（递归域, 08-19 简报 08-22 补落地）
- mse_probe       : P1 多尺度样本熵（熵家族延伸, 08-20 简报 08-22 补落地）
- renyi_probe     : P1 Rényi 广义熵谱（熵家族延伸, 08-22 简报 08-22 补落地）
- dcca_probe      : P1 去趋势交叉相关分析（跨序列依赖, 08-20 简报 08-22 补落地）
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
