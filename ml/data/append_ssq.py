#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# CRON 绑定: 壳=~/.hermes/scripts/append_ssq.py  cron job=246a519bce0b (经 update_ssq 间接触发)
# 本文件是逻辑真身; 改这里即生效, 勿改壳里的副本
"""
安全地向 1.csv 追加双色球开奖记录（纯标准库，保留 CRLF 格式 + 尾部查重）。

设计要点：
- 不依赖 pandas，避免 to_csv 破坏原始 CRLF 行尾。
- 尾部查重：只追加「文件尾部 incremental_check_rows 行内不存在的新期号」。
- 单期幂等：重复追加同一期不会重复写入。
- 支持两种用法：
  1) 命令行追加一行：  python3 append_ssq.py <dNum> <yNum> <mNum> <dDate> <r1>..<r6> <blue>
  2) 作为模块被 cron 调用：  append_records([ {...}, ... ]) -> (added, skipped)

文件格式（每行，逗号分隔，CRLF 结尾）：
  dNum,yNum,mNum,dDate,Red1,Red2,Red3,Red4,Red5,Red6,Blue1
示例：
  2026092,2026,08,2026-08-11,09,11,12,25,30,33,11
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

# 与仓库 ml/config.py 保持一致
INCREMENTAL_CHECK_ROWS = 100

# CSV 路径：优先用环境变量 SSQ_CSV，否则回退到仓库默认数据文件
import os as _os
_DEFAULT_CSV = Path("/home/hermes/workspace/python/SSQ/ml/data/1.csv")
CSV_PATH = Path(_os.environ.get("SSQ_CSV", str(_DEFAULT_CSV))).resolve()
HEADERS = ["dNum", "yNum", "mNum", "dDate",
           "Red1", "Red2", "Red3", "Red4", "Red5", "Red6", "Blue1"]


def _read_rows(path: Path) -> list[list[str]]:
    """读取全部数据行（不含表头），保留为字符串列表。"""
    if not path.exists():
        return []
    rows: list[list[str]] = []
    # 以 utf-8-sig 兼容可能的 BOM；CSV 解析自动处理引号/逗号
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return []
        # 兼容：若首行是表头则跳过
        if first[:1] and first[0].strip().lower() in ("dnum", "deliver number"):
            pass
        else:
            rows.append(first)
        for r in reader:
            if r and any(c.strip() for c in r):
                rows.append(r)
    return rows


def _tail_dnums(rows: list[list[str]], n: int = INCREMENTAL_CHECK_ROWS) -> set[str]:
    """取最后 n 行的 dNum（第 0 列）作为查重集合。"""
    return {r[0].strip() for r in rows[-n:] if r and r[0].strip()}


def append_records(records: list[dict]) -> tuple[list[str], list[str]]:
    """追加若干期。records 每项形如
    {'dNum':2026092,'yNum':2026,'mNum':8,'dDate':'2026-08-11',
     'Red1':9,...,'Red6':33,'Blue1':11}
    返回 (added_dnums, skipped_dnums)。
    """
    rows = _read_rows(CSV_PATH)
    existing = _tail_dnums(rows)

    added, skipped = [], []
    buf = io.StringIO(newline="")  # 用 CRLF 写出
    writer = csv.writer(buf, lineterminator="\r\n")

    # 确保文件存在且有表头
    need_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0

    # 先规范化传入记录为行列表，按 dNum 升序
    norm: list[list[str]] = []
    for rec in records:
        dnum = str(rec["dNum"]).strip()
        if dnum in existing:
            skipped.append(dnum)
            continue
        row = [
            str(rec["dNum"]).strip(),
            str(rec["yNum"]).strip(),
            f"{int(rec['mNum']):02d}",
            str(rec["dDate"]).strip(),
            *(f"{int(rec[f'Red{i}']):02d}" for i in range(1, 7)),
            f"{int(rec['Blue1']):02d}",
        ]
        norm.append(row)
        existing.add(dnum)

    norm.sort(key=lambda r: int(r[0]))
    for row in norm:
        writer.writerow(row)
        added.append(row[0])

    if norm:
        mode = "w" if need_header else "a"
        with CSV_PATH.open(mode, encoding="utf-8", newline="") as f:
            if need_header:
                hdr = csv.writer(f, lineterminator="\r\n")
                hdr.writerow(HEADERS)
            f.write(buf.getvalue())

    return added, skipped


def main(argv: list[str]) -> int:
    if len(argv) == 1:
        print("用法: python3 append_ssq.py <dNum> <yNum> <mNum> <dDate> <r1>..<r6> <blue>")
        print("示例: python3 append_ssq.py 2026092 2026 08 2026-08-11 09 11 12 25 30 33 11")
        return 2
    # argv[1:] = 11 字段
    fields = argv[1:]
    if len(fields) != 11:
        print(f"错误: 需要 11 个字段，收到 {len(fields)} 个")
        return 2
    rec = {
        "dNum": fields[0], "yNum": fields[1], "mNum": fields[2], "dDate": fields[3],
        "Red1": fields[4], "Red2": fields[5], "Red3": fields[6], "Red4": fields[7],
        "Red5": fields[8], "Red6": fields[9], "Blue1": fields[10],
    }
    added, skipped = append_records([rec])
    if added:
        print(f"已追加 {len(added)} 期: {added}")
    if skipped:
        print(f"已跳过(重复) {len(skipped)} 期: {skipped}")
    if not added and not skipped:
        print("未写入任何数据")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
