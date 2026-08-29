#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【只读探针 · 不写库不落盘】探查中彩网 SSQ 开奖接口返回的完整字段,
确认是否含奖池(pool)/销量(sales)/各奖级中奖注数(winCounts)。
目的: 为 B 档「扩 schema + 爬虫补奖金数据」可行性做前置验证。
"""
from __future__ import annotations
import json
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

URL = ("https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
       "?name=ssq&pageNo=1&pageSize=3")


def probe() -> None:
    headers = {
        "User-Agent": UA,
        "Referer": "https://www.cwl.gov.cn/ygkj/kjgg/",
    }
    req = urllib.request.Request(URL, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    print("state:", data.get("state"))
    print("message:", data.get("message"))
    print("total (result count):", len(data.get("result", [])))
    print("=" * 60)
    for i, res in enumerate(data.get("result", [])[:3]):
        print(f"\n--- 第 {i+1} 条 (期号={res.get('code')}) 完整字段 ---")
        for k, v in res.items():
            # 打印所有键，特别标注奖池/销量/注数相关
            mark = ""
            if any(w in k.lower() for w in ("pool", "sale", "sales", "prize", "win", "award", "count", "money", "amount")):
                mark = "  <<< 奖金/注数相关"
            print(f"  {k!r}: {v!r}{mark}")


if __name__ == "__main__":
    probe()
