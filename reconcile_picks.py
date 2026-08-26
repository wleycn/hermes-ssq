#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开奖推荐命中核对: 开奖号 vs ssq.predicted_picks 逐组比对, 生成邮件 HTML 块。

设计(2026-08-20 Rocky 拍板):
- fetch_picks: 只读 PG ssq.predicted_picks, 按 period 取当期推荐(上期开奖后 22:15 落库的,
  不是"今天 22:15 发的下一期推荐")。
- analyze_hits: 纯函数, 输入开奖号 + 推荐行 → 每组命中明细(可单测, 不碰 DB)。
- render_hits_html: 与 ssq_send_picks.render_html 同风格表格(border=1/cellpadding=4),
  A. top5 5 组 与 B. wheel 30 注统一逐号标注 ✅/—, 蓝球 ✅/❌。
- 当期无推荐记录: 返回 None, 由调用方写占位说明, 不抛错。

用法: 由 ml/data/update_ssq.py 在"新增一期开奖"时调用, 命中块拼进开奖信邮件。
"""
from __future__ import annotations

from typing import Any, Optional

from ml.pg_conn import pg_dict

# 与 select_numbers.py 同款 PG 连接(SSQ 项目本地 PG)
PG = pg_dict()  # 凭证从 ~/.hermes/.env 的 DATABASE_URL 读, 不硬编码


def connect() -> Any:
    """建立 PG 连接(类型明确的工厂, 避免 **dict 混合类型)。"""
    import psycopg
    return psycopg.connect(host=PG["host"], port=PG["port"], user=PG["user"],
                           password=PG["password"], dbname=PG["dbname"])


def fetch_picks(conn: Any, period: str) -> list[dict]:
    """读当期推荐行, 按 mode, group_idx 排序。无记录返回空列表。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT mode, group_idx, reds, blue, popularity, seed "
            "FROM ssq.predicted_picks WHERE period=%s ORDER BY mode, group_idx",
            (period,))
        rows = cur.fetchall()
    out = []
    for mode, gidx, reds, blue, popularity, seed in rows:
        out.append({
            "mode": mode,
            "group_idx": int(gidx),
            "reds": [int(x) for x in reds.split(",")],
            "blue": int(blue),
            "popularity": float(popularity) if popularity is not None else None,
            "seed": int(seed),
        })
    return out


def prize_tier(red_hits: int, blue_hit: bool) -> str:
    """双色球奖级判定(固定奖口径, 浮动奖只标级)。"""
    if red_hits == 6 and blue_hit:
        return "一等奖"
    if red_hits == 6:
        return "二等奖"
    if red_hits == 5 and blue_hit:
        return "三等奖"
    if red_hits == 5 or (red_hits == 4 and blue_hit):
        return "四等奖"
    if red_hits == 4 or (red_hits == 3 and blue_hit):
        return "五等奖"
    if blue_hit:
        return "六等奖"
    return "未中奖"


def analyze_hits(drawn_reds: list[int], drawn_blue: int, picks: list[dict]) -> list[dict]:
    """逐组比对。返回每行: {mode, group_idx, reds, blue, hit_reds, red_count, blue_hit, prize}。"""
    drawn_set = set(int(x) for x in drawn_reds)
    out = []
    for p in picks:
        hit_reds = [r for r in p["reds"] if r in drawn_set]
        blue_hit = int(p["blue"]) == int(drawn_blue)
        out.append({
            "mode": p["mode"],
            "group_idx": p["group_idx"],
            "reds": p["reds"],
            "blue": p["blue"],
            "hit_reds": hit_reds,
            "red_count": len(hit_reds),
            "blue_hit": blue_hit,
            "prize": prize_tier(len(hit_reds), blue_hit),
        })
    return out


def _ball_html(num: int, hit: bool, blue: bool = False) -> str:
    """单个号码的 HTML: 命中=✅+红色加粗(蓝球命中=✅+蓝色加粗), 未中=灰色。"""
    if hit:
        color = "#1e5aff" if blue else "#c00"
        mark = "✅"
        return f'<b style="color:{color}">{num:02d}{mark}</b>'
    return f'<span style="color:#999">{num:02d}</span>'


def _rows_html(analyzed: list[dict], mode: str) -> str:
    """统一风格的命中表行: 红球逐号 ✅/— + 蓝球 + 结果(中X红 / 中X红+蓝 / 奖级)。"""
    rows = []
    for a in analyzed:
        if a["mode"] != mode:
            continue
        reds_html = " ".join(_ball_html(r, r in a["hit_reds"]) for r in a["reds"])
        blue_html = _ball_html(a["blue"], a["blue_hit"], blue=True)
        if a["blue_hit"] and a["red_count"] > 0:
            result = f"中{a['red_count']}红+蓝 · {a['prize']}"
        elif a["blue_hit"]:
            result = f"蓝球中 · {a['prize']}"
        elif a["red_count"] > 0:
            result = f"中{a['red_count']}红 · {a['prize']}"
        else:
            result = "未中奖"
        rows.append(
            f"<tr><td>{a['group_idx']}</td><td>{reds_html}</td>"
            f"<td>{blue_html}</td><td>{result}</td></tr>")
    return "".join(rows)


def render_hits_html(period: str, drawn_reds: list[int], drawn_blue: int,
                     analyzed: list[dict]) -> str:
    """生成命中核对 HTML 块(与推荐信同风格), 追加到开奖信。"""
    n_top5 = sum(1 for a in analyzed if a["mode"] == "top5")
    n_wheel = sum(1 for a in analyzed if a["mode"] == "wheel")
    hit_top5 = sum(1 for a in analyzed if a["mode"] == "top5"
                   and (a["red_count"] > 0 or a["blue_hit"]))
    hit_wheel = sum(1 for a in analyzed if a["mode"] == "wheel"
                    and (a["red_count"] > 0 or a["blue_hit"]))
    style = 'border="1" cellpadding="4" cellspacing="0"'
    parts = [
        "<hr>",
        f"<h3>📊 与推荐号码核对(第{period}期, 开奖前已发出的推荐)</h3>",
        f"<p>开奖号: 红球 <b>{' '.join(f'{x:02d}' for x in drawn_reds)}</b> "
        f"蓝球 <b>{drawn_blue:02d}</b> ｜ 本信命中数: "
        f"A 组 {hit_top5}/{n_top5} 注有奖 · B 组 {hit_wheel}/{n_wheel} 注有奖</p>",
    ]
    if n_top5:
        parts.append(
            f'<h4>A. 常规 5 组</h4><table {style}>'
            f"<tr><th>组</th><th>红球(命中✅)</th><th>蓝球</th><th>结果</th></tr>"
            f"{_rows_html(analyzed, 'top5')}</table>")
    if n_wheel:
        parts.append(
            f'<h4>B. 旋转矩阵 wheel 30 注</h4><table {style}>'
            f"<tr><th>#</th><th>红球(命中✅)</th><th>蓝球</th><th>结果</th></tr>"
            f"{_rows_html(analyzed, 'wheel')}</table>")
    parts.append("<p>仅供娱乐参考; 中奖以官方开奖公告为准。</p>")
    return "\n".join(parts)


def build_reconcile_block(conn: Any, period: str,
                          drawn_reds: list[int], drawn_blue: int) -> Optional[str]:
    """对外主入口: 查 PG + 比对 + 渲染。无推荐记录返回 None(调用方写占位)。"""
    picks = fetch_picks(conn, period)
    if not picks:
        return None
    analyzed = analyze_hits(drawn_reds, drawn_blue, picks)
    return render_hits_html(period, drawn_reds, drawn_blue, analyzed)
