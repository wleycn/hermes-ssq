#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成双色球 5 组候选号码并通过 smtplib 直发邮件(绕过 hermes 坏掉的 email 通道)。

流程: select_numbers.generate -> 组装中文邮件正文 -> 发往 wleycn@163.com。

用法:
  .venv/bin/python send_ssq_picks.py                 # 生成并发送
  .venv/bin/python send_ssq_picks.py --dry-run      # 只打印, 不发送
  .venv/bin/python send_ssq_picks.py --seed 7       # 换随机种子重生成
"""
from __future__ import annotations
import argparse
import os
import smtplib
import sys
from datetime import datetime
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import select_numbers as SN


def read_env() -> dict:
    f = Path(os.path.expanduser("~/.hermes/.env"))
    v = {}
    if f.exists():
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, val = line.split("=", 1)
                v[k.strip()] = val.strip()
    return v


def build_body(red_mean, blue_mean, groups, run_at, models):
    """用 select_numbers 已算好的概率与选号结果组装中文邮件正文。"""
    picks = SN.generate(red_mean, blue_mean, groups=groups, seed=42)
    top_red = [int(x) + 1 for x in SN.np.argsort(red_mean)[::-1][:8]]
    top_blue = [int(x) + 1 for x in SN.np.argsort(blue_mean)[::-1][:5]]

    lines = []
    lines.append(f"双色球 5 组候选号码（ML 模型集成预测）")
    lines.append(f"生成时间：{datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append(f"模型来源：{', '.join(models)}（取最新一次预测，各模型概率均值集成）")
    lines.append("")
    lines.append(f"红球高概率 Top8：{top_red}")
    lines.append(f"蓝球高概率 Top5：{top_blue}")
    lines.append("=" * 48)
    for g in picks:
        lines.append(f"第{g['group']}组：红球 {g['red']}  +  蓝球 {g['blue']:02d}")
        lines.append(f"      热号(集成概率前8)：红球 {g['hot_reds']} ｜ 蓝球综合排名 #{g['blue_rank']}")
    lines.append("=" * 48)
    lines.append("")
    lines.append("【选取逻辑】")
    lines.append("1. 数据：基于 1.csv 全部历史开奖（2003–今，含最新 2026092 期）。")
    lines.append("2. 模型：参与集成的模型各对 33 个红球、16 个蓝球输出概率，")
    lines.append("   存入 PostgreSQL 后取均值集成，降低单模型偏差（本次参与模型见上方『模型来源』）。")
    lines.append("3. 选号：红球按集成概率做受控随机加权抽样（温度 0.6），每注 6 个不重复，")
    lines.append("   并约束奇偶比∈{2:4,3:3,4:2}、大小比(1-16小/17-33大)∈{2:4,3:3,4:2}；")
    lines.append("   蓝球按集成概率温度 0.7 加权抽样。这样既尊重模型高概率号，又保留合理分散。")
    lines.append(f"4. 标注『热号』= 该号码在 {len(models)} 个模型集成概率中排前 8，属模型重点看好号。")
    lines.append("")
    lines.append("⚠ 免责声明：以上为 ML 模型基于历史数据的统计派生结果，仅供娱乐参考，")
    lines.append("   双色球为独立随机事件，不保证任何中奖概率。请理性购彩。")
    return "\n".join(lines)


def send_email(subject: str, body: str, to_addr: str, dry_run: bool = False) -> bool:
    env = read_env()
    host = env.get("EMAIL_SMTP_HOST", "smtp.163.com")
    port = int(env.get("EMAIL_SMTP_PORT", "465"))
    user = env.get("EMAIL_ADDRESS") or env.get("EMAIL_USER", "")
    pwd = env.get("EMAIL_PASSWORD", "")
    if not pwd or not user:
        print("[warn] 邮件凭据缺失")
        return False
    if dry_run:
        print(f"[DRY-RUN] 将发往 {to_addr} via {host}:{port}\n--- 正文 ---\n{body}")
        return True
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = Header(subject, "utf-8")
    try:
        with smtplib.SMTP_SSL(host, port, timeout=30) as s:
            s.login(user, pwd)
            s.send_message(msg)
        print(f"✓ 邮件已发送至 {to_addr}")
        return True
    except Exception as e:
        print(f"✗ 邮件发送失败: {e!r}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--to", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import psycopg
    conn = psycopg.connect(**SN.PG)
    try:
        red_mean, blue_mean, run_at, models = SN.load_latest_probs(conn)
    finally:
        conn.close()

    body = build_body(red_mean, blue_mean, args.groups, run_at, models)
    subject = f"🎯 双色球 5 组候选号码（模型集成 · {datetime.now():%m-%d}）"
    to = args.to or read_env().get("EMAIL_HOME_ADDRESS") or "wleycn@163.com"
    ok = send_email(subject, body, to, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
