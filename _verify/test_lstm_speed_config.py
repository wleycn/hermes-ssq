#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 LSTM 加速参数已生效(不触发真实训练, 秒级)。

运行: .venv/bin/python -m pytest _verify/test_lstm_speed_config.py -q
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path("/home/hermes/workspace/python/SSQ")


def _load(name, path):
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_lstm_config_epochs_reduced():
    cfg = _load("config", ROOT / "ml" / "config.py")
    assert cfg.LSTM_CONFIG["epochs"] == 80, "epochs 应已降到 80"
    assert cfg.LSTM_CONFIG["window_size"] == 128


def test_lstm_all_default_window_fixed():
    """lstm_all 的 prepare_data 默认 window 应为 128 (不再是 330)。"""
    L = _load("lstm_model", ROOT / "ml" / "models" / "lstm_model.py")
    import inspect
    src = inspect.getsource(L.LSTMAllModel.prepare_data)
    assert "window_size or self.config.get(\"window_size\", 128)" in src, \
        "lstm_all 默认 window 应改为 128"


def test_train_loop_has_progress_print():
    """训练循环应有进度打印, 避免静默误判卡死。"""
    L = _load("lstm_model", ROOT / "ml" / "models" / "lstm_model.py")
    import inspect
    src = inspect.getsource(L._train_loop)
    assert "epoch" in src and "print(" in src, "训练循环应打印 epoch 进度"
