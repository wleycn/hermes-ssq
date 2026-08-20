#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双色球开奖记录 多源轮询更新器（cron 调用 / 手动调用）。

功能：
  1. 多源轮询最新开奖：中彩网官方接口(主源/最权威) + EastMoney(列表) + 网易(单期页)。
  2. 中彩网为权威主源，其余两源交叉校验/兜底；取交叉一致的最新一期。
  3. 增量写入 1.csv：尾部查重 + 单期幂等（依赖 append_ssq.append_records）。
  4. 邮件统一由 send_email.py 中枢发送 (To=126 + Cc=163 由 .env 兜底)。

退出码：0=成功(无论有无新增) | 2=参数错误 | 1=运行异常
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 让脚本能 import 同目录的 append_ssq
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import append_ssq as store  # noqa: E402

# 项目根(SSQ), 供 import reconcile_picks
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# ----------------------------- 数据源解析 -----------------------------

def _get(url: str, timeout: int = 15) -> str:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    # 尝试 utf-8，失败退回 gbk（网易/部分站点）
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def parse_eastmoney_latest(html: str) -> dict | None:
    """从 EastMoney 历史列表页解析最新一期。
    返回 {'dNum':int,'yNum':int,'mNum':int,'dDate':str,'reds':[..6],'blue':int}
    EastMoney 结构：每行 <td><a ..id=2026XXX>..</a></td> 表头，红球在 pellet span。
    日期取真实开奖日期字段（期号里的月/日不可靠）。
    """
    ids = re.findall(r'/Result/Category/ssq\?type=ssq&id=(\d+)', html)
    if not ids:
        return None
    latest = max(int(x) for x in ids)
    # 该期所在 <tr> 块
    block = re.search(r'<tr>.*?id=%s.*?</tr>' % latest, html, re.S)
    block_txt = block.group(0) if block else html
    reds = re.findall(r'pellet-sm red">(\d{2})</span>', block_txt)
    blue = re.findall(r'pellet-sm blue">(\d{2})</span>', block_txt)
    if len(reds) < 6 or not blue:
        return None
    # 真实日期：block 内 "YYYY-MM-DD(星期X)"
    dm = re.search(r'(\d{4}-\d{2}-\d{2})\(', block_txt)
    if not dm:
        return None
    dDate = dm.group(1)
    y, mo, _ = dDate.split("-")
    s = str(latest)
    return {
        "dNum": latest, "yNum": int(y), "mNum": int(mo),
        "dDate": dDate, "reds": [int(x) for x in reds[:6]], "blue": int(blue[0]),
    }


def parse_cwl_latest() -> dict | None:
    """中彩网官方接口（民政部直属，最权威）。
    接口: GET https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice
          ?name=ssq&pageNo=1&pageSize=1
    Headers 必须带 Referer: https://www.cwl.gov.cn/ygkj/kjgg/ (否则 404)。
    返回 {'dNum','yNum','mNum','dDate','reds':[6],'blue'}。"""
    import json
    url = ("https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
           "?name=ssq&pageNo=1&pageSize=1")
    # 复用 _get 拿 HTML 文本（内部已带 UA）；再补 Referer 头需单独请求
    import urllib.request
    headers = {
        "User-Agent": UA,
        "Referer": "https://www.cwl.gov.cn/ygkj/kjgg/",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))
    if data.get("state") != 0 or not data.get("result"):
        return None
    res = data["result"][0]
    code = int(res["code"])
    reds = [int(x) for x in res["red"].split(",")]
    blue = int(res["blue"])
    if len(reds) != 6:
        return None
    # 日期形如 "2026-08-11(二)"，提取真实日期
    dm = re.search(r"(\d{4}-\d{2}-\d{2})", res.get("date", ""))
    dDate = dm.group(1) if dm else ""
    if not dDate:
        return None
    y, mo, _ = dDate.split("-")
    return {
        "dNum": code, "yNum": int(y), "mNum": int(mo),
        "dDate": dDate, "reds": reds, "blue": blue,
    }


def parse_163_issue(html: str, issue: int) -> dict | None:
    """网易单期页：红球在 class 含 red 的 span，蓝球在 class 含 blue 的 span。
    日期取真实「开奖日期: YYYY-MM-DD」字段。"""
    m_iss = re.search(r'(\d{7})期', html)
    if not m_iss:
        return None
    page_iss = int(m_iss.group(1))
    if issue and page_iss != issue:
        return None
    reds = re.findall(r'class="[^"]*red[^"]*"[^>]*>(\d{1,2})</span>', html)
    blues = re.findall(r'class="[^"]*blue[^"]*"[^>]*>(\d{1,2})</span>', html)
    if len(reds) < 6 or not blues:
        return None
    dm = re.search(r'开奖日期:\s*(\d{4}-\d{2}-\d{2})', html)
    if not dm:
        return None
    dDate = dm.group(1)
    y, mo, _ = dDate.split("-")
    return {
        "dNum": page_iss, "yNum": int(y), "mNum": int(mo),
        "dDate": dDate, "reds": [int(x) for x in reds[:6]], "blue": int(blues[0]),
    }


def _infer_date_unused():
    pass


def fetch_latest() -> dict | None:
    """多源轮询，返回交叉校验通过的最新一期 dict；无一致结果返回 None。

    优先级/策略：
      - 中彩网官方接口(cwl.gov.cn) 为权威主源（民政部直属，最可靠）。
      - EastMoney 列表 + 网易单期页 作为交叉校验与兜底。
      - 中彩网成功即采用；若另有源与其号码一致则信心最高；
        中彩网不可达时，退回 EastMoney/网易交叉校验结果。
    """
    candidates: list[tuple[str, dict]] = []

    # 源1：中彩网官方（主源）
    try:
        c = parse_cwl_latest()
        if c:
            candidates.append(("cwl", c))
    except Exception as e:
        print(f"[warn] 中彩网失败: {e}")

    # 源2：EastMoney 列表（结构稳，但可能滞后）
    try:
        em = _get("https://caipiao.eastmoney.com/pub/Result/History/ssq?page=1")
        c = parse_eastmoney_latest(em)
        if c:
            candidates.append(("eastmoney", c))
    except Exception as e:
        print(f"[warn] eastmoney 失败: {e}")

    # 源3：网易单期页 —— 以「候选里最大期号」为基准，探测该期及后两期
    if candidates:
        base = max(c["dNum"] for _, c in candidates)
        for iss in range(base, base + 3):
            try:
                h = _get(f"https://sports.163.com/caipiao/lottery/ssq/{iss}")
                c = parse_163_issue(h, iss)
                if c:
                    candidates.append(("163", c))
            except Exception as e:
                print(f"[warn] 163 期 {iss} 失败: {e}")

    if not candidates:
        return None

    # 交叉校验：统计每个 (dNum,reds,blue) 出现的次数
    from collections import Counter
    sig = Counter((c["dNum"], tuple(c["reds"]), c["blue"]) for _, c in candidates)
    best_sig, best_cnt = sig.most_common(1)[0]

    # 主源中彩网命中即采用
    cwl_hit = [c for src, c in candidates if src == "cwl"
               and (c["dNum"], tuple(c["reds"]), c["blue"]) == best_sig]
    if cwl_hit:
        print(f"[ok] 中彩网权威源命中 期{best_sig[0]} (交叉源数={best_cnt})")
        return cwl_hit[0]

    # 中彩网未命中：需至少两个非官方源一致才采用（防脏数据）
    if best_cnt >= 2:
        alt = [c for _, c in candidates
               if (c["dNum"], tuple(c["reds"]), c["blue"]) == best_sig][0]
        print(f"[ok] 非官方源交叉一致 期{best_sig[0]} (源数={best_cnt})")
        return alt

    print(f"[warn] 期 {best_sig[0]} 仅单源命中且非中彩网，跳过写入")
    return None


# ----------------------------- 邮件发送 -----------------------------

def read_env() -> dict:
    env_file = Path(os.path.expanduser("~/.hermes/.env"))
    vars_: dict = {}
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                vars_[k.strip()] = v.strip()
    return vars_


def send_email(subject: str, body: str, to_addr: str | None = None,
               html: bool = False) -> bool:
    """统一走中枢 CLI: 默认收件 (To=126 + Cc=163); 显式 to_addr 时覆盖.
    html=True 时按 HTML 正文发送(命中核对表用)。"""
    from pathlib import Path as _P
    cli = _P.home() / ".hermes/skills/email/send-email/send_email.py"
    tmp = _P("/tmp/ssq_update_body.html" if html else "/tmp/ssq_update_body.txt")
    tmp.write_text(body, encoding="utf-8")
    cmd = [sys.executable, str(cli), "--subject", subject, "--body-file", str(tmp)]
    if html:
        cmd += ["--html"]
    if to_addr:
        cmd += ["--to", to_addr]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if cp.returncode != 0:
            print(f"✗ 邮件发送失败: {cp.stderr.strip()[:200]}")
            return False
        print(cp.stdout.strip())
        return True
    except Exception as e:
        print(f"✗ 邮件发送异常: {e!r}")
        return False


# ----------------------------- 主流程 -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-email", action="store_true", help="只更新 CSV，不发送邮件")
    ap.add_argument("--to", default=None, help="收件人地址(默认 EMAIL_HOME_ADDRESS)")
    args = ap.parse_args()

    # 锁定 CSV 写入目标：环境变量 SSQ_CSV 优先，默认仓库真实数据文件
    import os as _os
    from pathlib import Path as _P
    _csv_target = _os.environ.get("SSQ_CSV",
                                 "/home/hermes/workspace/python/SSQ/ml/data/1.csv")
    store.CSV_PATH = _P(_csv_target).resolve()

    print(f"[run] {datetime.now():%Y-%m-%d %H:%M:%S} 开始检查双色球开奖")
    latest = fetch_latest()
    if not latest:
        print("[result] 未获取到可靠的新开奖数据")
        if not args.no_email:
            send_email("双色球开奖检查 - 无更新",
                       f"{datetime.now():%Y-%m-%d %H:%M:%S}\n本次检查未获取到新的开奖数据（或数据源尚未更新）。",
                       args.to)
        return 0

    rec = {
        "dNum": latest["dNum"], "yNum": latest["yNum"], "mNum": latest["mNum"],
        "dDate": latest["dDate"],
        "Red1": latest["reds"][0], "Red2": latest["reds"][1], "Red3": latest["reds"][2],
        "Red4": latest["reds"][3], "Red5": latest["reds"][4], "Red6": latest["reds"][5],
        "Blue1": latest["blue"],
    }
    added, skipped = store.append_records([rec])

    if added:
        line = (f"第{latest['dNum']}期 ({latest['dDate']}) 开奖: "
                f"红球 {' '.join(f'{x:02d}' for x in latest['reds'])}  蓝球 {latest['blue']:02d}")
        print(f"[result] 已新增 {added} 期: {line}")
        subject = f"🎯 双色球第{latest['dNum']}期开奖结果"
        body = (f"双色球自动检查更新<br>时间: {datetime.now():%Y-%m-%d %H:%M:%S}<br><br>"
                f"新增期号: 第{latest['dNum']}期 ({latest['dDate']})<br>"
                f"红球: <b>{' '.join(f'{x:02d}' for x in latest['reds'])}</b><br>"
                f"蓝球: <b>{latest['blue']:02d}</b><br><br>"
                f"已写入: {store.CSV_PATH}<br>(数据以官方开奖公告为准)")
        # 追加"与推荐号码核对"块(读 PG predicted_picks, 只读; 失败不影响开奖入库/发信)
        try:
            import reconcile_picks as rc
            conn = rc.connect()
            try:
                block = rc.build_reconcile_block(
                    conn, str(latest["dNum"]), latest["reds"], latest["blue"])
            finally:
                conn.close()
            if block:
                body += "<br><br>" + block
            else:
                body += (f"<br><hr>⚠️ 第{latest['dNum']}期在 predicted_picks 中"
                         "无推荐记录(可能上次发预测未成功), 无法核对推荐命中。")
        except Exception as e:
            print(f"[warn] 推荐核对失败(不影响开奖入库): {e}")
            body += "<br><hr><p>推荐核对暂不可用, 仅报告开奖结果。</p>"
        if not args.no_email:
            send_email(subject, body, args.to, html=True)
    else:
        print(f"[result] 期 {latest['dNum']} 已存在，跳过")
        if not args.no_email:
            send_email("双色球开奖检查 - 已是最新",
                       f"{datetime.now():%Y-%m-%d %H:%M:%S}\n"
                       f"最新期 {latest['dNum']} 已在 CSV 中，无需更新。",
                       args.to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
