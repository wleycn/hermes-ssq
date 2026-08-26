#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CRON 绑定: 多脚本共用  cron job=246a519bce0b / 98164fe6c0d6 / ced57f0994d8
# 本文件是逻辑真身; 改这里即生效, 勿改各脚本里的副本
"""SSQ 统一 PG 连接工厂: 从 ~/.hermes/.env 的 DATABASE_URL 读凭证, 不硬编码密码。

所有需要连 PG 的脚本统一 import 本模块, 消除 batch_predict_pg/retrain_pipeline/
select_numbers/ssq_send_picks/db_draw/reconcile_picks/pg_schema 各处的 hardcode。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# 自动把项目根加入 sys.path, 使任意调用方式下 `import ml.pg_conn` 都能解析(幂等)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# SSQ 项目固定 schema
SCHEMA = "ssq"

# .env 位置(与项目其他脚本约定一致)
_ENV_PATH = Path.home() / ".hermes" / ".env"


def _load_env() -> dict[str, str]:
    """读 ~/.hermes/.env 到 dict(轻量, 不依赖 python-dotenv)。"""
    env: dict[str, str] = {}
    if _ENV_PATH.exists():
        for line in _ENV_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # 进程环境变量优先(若已 export)
    for k, v in os.environ.items():
        if k not in env:
            env[k] = v
    return env


def get_database_url() -> str:
    """返回 DATABASE_URL; 缺失时抛清晰错误。"""
    env = _load_env()
    url = env.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "未找到 DATABASE_URL(应在 ~/.hermes/.env 或环境变量), 无法连接 PG")
    return url


def connect() -> Any:
    """建立 PG 连接(类型明确的工厂, 避免 **dict 混合类型)。"""
    import psycopg
    return psycopg.connect(get_database_url())


# 兼容旧代码: 保留 PG dict 形态(database_url → dict), 供仍用 **PG 的代码平滑迁移
def pg_dict() -> dict[str, Any]:
    """从 DATABASE_URL 解析出 dict 形态(供 psycopg.connect(**PG) 兼容旧调用)。"""
    from urllib.parse import urlparse
    url = get_database_url()
    p = urlparse(url)
    return {
        "host": p.hostname or "localhost",
        "port": p.port or 5432,
        "user": p.username or "hermes",
        "password": p.password or "",
        "dbname": p.path.lstrip("/") or "hermes",
    }


def purge_stale_predictions(conn: Any, days: int = 30) -> int:
    """清理 model_predictions 中超期的行(data_date 早于今天 - days)。

    设计(M1, Rocky 2026-08-26 拍板): 清理逻辑放在生成/入库之后执行,
    每次 batch_predict_pg 写库即触发, 防止表无限增长。
    draw_history 全量保留, 不在此清理范围内。
    返回删除行数。
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {SCHEMA}.model_predictions WHERE data_date < %s;",
            (cutoff,),
        )
        n = cur.rowcount
    conn.commit()
    return n
