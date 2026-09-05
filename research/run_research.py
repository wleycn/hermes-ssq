#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ 预测方法研究 · 每周2次自动研究 + 邮件简报。

修复：在 Hermes 全局 web backend=tavily 但 runtime 未注册该 provider 的环境下，
改为直接调用 Tavily API 做外部检索，绕过 Hermes web tools。
"""
import subprocess
import json
import os
import sys
import datetime
from pathlib import Path

# hermes 所在目录补进 PATH, 防止子进程再 fork 其他 hermes 相关命令时找不到
HERMES_DIR = "/home/hermes/.local/bin"
if HERMES_DIR not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = HERMES_DIR + ":" + os.environ.get("PATH", "")

sys.path.insert(0, "/home/hermes/workspace/.infra")
from path_anchor import SEND_EMAIL_CLI  # noqa: E402

ROOT = Path(__file__).resolve().parent
DC_SSQ = Path("/home/hermes/workspace/data-center/ssq")  # 产出真源 2026-08-30 迁移
REPORTS = DC_SSQ / "research" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TODAY = datetime.date.today().strftime("%Y-%m-%d")
REPORT_PATH = REPORTS / f"{TODAY}.md"
ARCH_PATH = ROOT / "ARCHITECTURE.md"  # 2026-08-22 并入 SSQ 项目目录后动态拼接, 不再写死绝对路径

# Tavily 端点
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


def load_hermes_env() -> str:
    """把 ~/.hermes/.env 注入当前进程环境变量，返回 Tavily API key。"""
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("\"'")
            if k and k not in os.environ:
                os.environ[k] = v
    return (os.environ.get("TAVILY_API_KEY") or "").strip()


def tavily_search(api_key: str, query: str, max_results: int = 5) -> dict:
    import requests

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": True,
    }
    r = requests.post(TAVILY_SEARCH_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def tavily_extract(api_key: str, urls: list[str]) -> dict:
    import requests

    payload = {"api_key": api_key, "urls": urls, "extract_depth": "advanced"}
    r = requests.post(TAVILY_EXTRACT_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


def load_context() -> tuple[str, list[str]]:
    """读架构文档 + 最近 2-3 份简报, 防止重复研究。"""
    arch = ""
    if ARCH_PATH.exists():
        arch = ARCH_PATH.read_text(encoding="utf-8", errors="ignore")

    recent = []
    for p in sorted(REPORTS.glob("*.md"), reverse=True)[:3]:
        try:
            recent.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return arch, recent


def build_queries() -> list[str]:
    """研究查询词：覆盖英文/中文 + 多个方向。"""
    return [
        "lottery prediction model new method 2025 2026",
        "双色球预测 机器学习 新算法 2025 2026",
        "彩票随机性检测 新方法 统计检验",
        "sequence prediction lottery Transformer GNN reinforcement learning",
        "双色球 特征工程 冷热号 遗漏值 新组合",
        "ensemble stacking bayesian model average lottery prediction",
    ]


def research(api_key: str) -> tuple[list[dict], list[dict]]:
    """执行外部检索, 返回 (search_results, extracts)。"""
    queries = build_queries()
    search_results: list[dict] = []
    extracts: list[dict] = []

    for q in queries:
        try:
            data = tavily_search(api_key, q, max_results=5)
            items = data.get("results") or data.get("data", {}).get("web") or []
            search_results.extend(items)
        except Exception as e:
            print(f"[research] search failed: {q}: {e}", flush=True)

    # 去重 URL
    seen = set()
    top_urls = []
    for item in search_results:
        url = item.get("url")
        if url and url not in seen:
            seen.add(url)
            top_urls.append(url)
        if len(top_urls) >= 6:
            break

    # 批量 extract 前 6 个 URL
    if top_urls:
        try:
            data = tavily_extract(api_key, top_urls)
            extracts.extend(data.get("results") or data.get("data") or [])
        except Exception as e:
            print(f"[research] extract failed: {e}", flush=True)

    return search_results, extracts


def synthesize(arch: str, recent: list[str], search_results: list[dict], extracts: list[dict]) -> str:
    """本地轻量合成简报：直接基于检索结果生成。"""
    findings = []
    for idx, item in enumerate(search_results[:10], 1):
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("description") or item.get("content") or ""
        findings.append(f"### [{title}]({url})\n- 来源: {url}\n- 摘要: {snippet[:300]}")

    extract_texts = []
    for ex in extracts[:5]:
        content = ex.get("content") or ex.get("raw_content") or ""
        extract_texts.append(content[:800])

    # 判断是否有新发现
    has_new = bool(findings)

    report = f"""# SSQ 预测方法研究简报 - {TODAY}

## 今日新发现 ({len(findings)} 条)
"""
    if findings:
        for f in findings:
            report += f"{f}\n\n"
    else:
        report += "- 外部检索未产出可验证新方案。\n- 内部已有证据已覆盖主流方向（详见总体评估）。\n\n"

    report += """## 总体评估
"""
    if has_new:
        report += f"""今日通过 Tavily 直接检索获取 {len(search_results)} 条搜索结果，提取 {len(extracts)} 份网页内容。结合当前项目内部状态（`ARCHITECTURE.md` §3 已证伪/已验证 + §3.1 探针14家族饱和 + §4 待办方向），**净新增仍需人工评估**：近期已系统覆盖了频谱/熵/分形/随机矩阵/交叉相关/序数等主流检验家族，且对 ML/DL/集成类新方案（Transformer/LSTM-CRF/GraphNN/RL/EBMA/贝叶斯堆叠）已做回退或判定“无适用空间”。

**下一步建议**：
1. 对上述新发现中感兴趣的方向，走 `evaluate.py` walk-forward + MC 显著性门槛，不显著即弃。
2. 维持探针月度重跑（cron `98164fe6c0d6`）即可，不再扩展新探针家族。
3. 如需深度精读，可手动打开上述链接阅读原文后决定是否接入现有 SSQ 管线。
"""
    else:
        report += """今日 Tavily 直接检索未返回有效结果。结合当前项目内部状态（`ARCHITECTURE.md` §3 已证伪/已验证 + §3.1 探针14家族饱和 + §4 待办方向），即使外部可检索，净新增也很可能继续趋近于零。

**下一步建议**：
1. 先确认 Tavily API key 与网络连通性是否正常。
2. 09 月可继续做“低优先”项中的 P2 方向（agreement 一致性度量、constrained_decode 替代评估），但均须走 `evaluate.py` walk-forward + MC 显著性门槛，不显著即弃。
3. 维持探针月度重跑（cron `98164fe6c0d6`）即可，不再扩展新探针家族。
"""

    return report


def send_email(report_path: Path):
    SEND_CLI = SEND_EMAIL_CLI
    subject = f"SSQ预测方法研究简报 - {TODAY}"
    cmd = [sys.executable, str(SEND_CLI), "--subject", subject, "--body-file", str(report_path)]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if cp.returncode != 0:
            print(f"[send_email] 失败: {cp.stderr.strip()[:200]}", flush=True)
            raise RuntimeError("EMAIL_SEND_FAILED")
        print(f"[send_email] 已提交中枢: {subject}", flush=True)
    except Exception as e:
        print(f"[send_email] 异常: {e}", flush=True)
        raise


def main():
    api_key = load_hermes_env()
    if not api_key:
        print("[main] 未找到 TAVILY_API_KEY, 无法进行外部检索", flush=True)
        sys.exit(1)

    arch, recent = load_context()

    print(f"[main] 启动 Tavily 直接研究 @ {datetime.datetime.now()}", flush=True)
    search_results, extracts = research(api_key)
    report = synthesize(arch, recent, search_results, extracts)

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"[main] 报告就绪: {REPORT_PATH}", flush=True)

    try:
        send_email(REPORT_PATH)
    except Exception as e:
        print(f"[main] 邮件发送失败: {e}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
