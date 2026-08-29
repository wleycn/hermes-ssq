#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSQ 预测方法研究 · 每周2次自动研究 + 邮件简报。

每周六/日 04:00 由系统 crontab(/var/spool/cron/crontabs/hermes)触发：
  1. 调用 Hermes CLI (-z 模式) 做无限制多源研究(互联网/GitHub/arXiv/文库)，
     寻找可改进双色球号码预测概率的新方案(新ML算法/数学模型/特征工程思路)。
  2. 研究结果写入 reports/YYYY-MM-DD.md。
  3. 邮件统一由 send_email.py 中枢发送 (To=126 + Cc=163 由 .env 兜底)。

频率说明: 2026-08-22 起由"每日"降频为"每周六/日" (Rocky 拍板:
SSQ 研究进入维护期, 方向转向 fin-risk 方法论迁移)。
⚠️ 修改 cron 调度须用 docker root 改 /var/spool/cron/crontabs/hermes
(hermes 用户对 /var/spool/cron 无写权限, crontab 命令被拒)。

用法:
  ./run_research.sh            # cron 调用
  hermes -z "$(cat prompt.txt)" --cli -t web,terminal,file   # 手动研究(内部命令)

依赖: hermes CLI 在 PATH; python3
"""
import os
import sys
import shutil
import subprocess
import datetime
from pathlib import Path

# hermes CLI 绝对路径: cron 环境 PATH 往往不含 ~/.local/bin, 必须写死,
# 否则 subprocess.Popen(["hermes", ...]) 会 FileNotFoundError。
HERMES_BIN = shutil.which("hermes") or "/home/hermes/.local/bin/hermes"

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TODAY = datetime.date.today().strftime("%Y-%m-%d")
REPORT_PATH = REPORTS / f"{TODAY}.md"
ARCH_PATH = ROOT / "ARCHITECTURE.md"  # 2026-08-22 并入 SSQ 项目目录后动态拼接, 不再写死绝对路径

# Hermes 研究 prompt: 无限制深挖, 产出简报格式(Q3=a)
# 关键: 强制要求 Hermes 把最终简报写入 REPORT_PATH, 不要输出到 stdout 或别处
RESEARCH_PROMPT = f"""你是双色球(SSQ)预测系统的研究方法论专家。今天的日期是 {TODAY}。

任务: 在互联网、GitHub、arXiv、Google Scholar、知乎/CSDN/博客园等所有公开可访问的来源中，
**无限制地搜索**可能改进我们双色球号码预测概率的新方案。花 15-30 分钟深入搜索，不要浅尝辄止。

**开跑前必读(强制性, 防止重复研究)**:
1. 先 read_file: {ARCH_PATH}
   ——重点读第 3 节「已证伪/已验证清单」与第 4 节「待办方向」。
2. 再快速浏览 reports/ 目录最近 2-3 份简报, 了解已研究过什么。
3. 任何新想法不得与已证伪清单重复(已测并 rollback: T13 五特征 ac/entropy/hot_cold/crf/diversity、
   质数偏好、EBMA 集成、马尔可夫特征、wheel ROI、频谱探针 FLAT/SCALAR_BIAS——这些不要再次推荐)。
4. 简报「总体评估」必须说明: 新发现相对已证伪清单的**净增量**在哪; 若净新增为低/无, 如实说明。

重点寻找(不限于):
- 新的 ML/DL 算法(如 Transformer/GraphNN/强化学习/生成模型在彩票/随机序列预测的应用)
- 数学模型(如马尔可夫链/泊松过程/信息熵/混沌理论/数论在 lottery 的应用)
- 特征工程创新(频率/遗漏/冷热号/跨度/和值的新型组合)
- 集成方法(多模型融合、stacking、贝叶斯模型平均)
- 任何声称在 lottery 预测上有突破的论文/项目/竞赛方案(Kaggle 等)

研究方法(自主决定深度):
- 用 web_search 多源检索, 包括英文和中文关键词
- 对最相关的 2-3 个发现, 深入阅读(web_extract 原文)理解其方法
- 批判性评估: 哪些是真有潜力, 哪些只是营销软文/伪科学(彩票本质随机, 警惕'100%准确率'骗局)
- 优先关注**可实际接入我们现有 SSQ 管线**(Python/PyTorch/sklearn)的方案

**输出要求(必须严格遵守)**:
1. 把最终简报用 write_file 工具写入这个绝对路径: {REPORT_PATH}
2. 不要只输出到对话/终端, 必须写入上面的文件
3. 写完后, 在终端只打印一行: DONE: {REPORT_PATH}
4. 不要在别处(如 /home/hermes/workspace/doc/)另存简报

简报格式(写入文件的内容, 这是要发邮件的):
# SSQ 预测方法研究简报 - {TODAY}

## 今日新发现 (N 条)
每条格式:
### [发现标题]
- 类型: ML算法 / 数学模型 / 特征工程 / 集成方法 / 其他
- 来源: [链接或引用]
- 一句话可行性评估: (高/中/低潜力 + 为什么)
- 核心思路: (1-2句技术要点)

## 总体评估
(一段话: 今天搜索的整体质量, 是否有值得立即尝试的方案, 下一步建议)

注意: 只报告**今天新找到的、且你认为有真实参考价值**的方案。不要凑数。若今天没找到有价值的, 如实说明。
"""


def run_research() -> bool:
    """调用 Hermes CLI 做研究, 简报由 Hermes 写入 REPORT_PATH。返回是否成功。"""
    cmd = [
        HERMES_BIN, "-z", RESEARCH_PROMPT, "--cli",
        "-t", "web,terminal,file",
    ]
    print(f"[run_research] 启动 Hermes 研究 @ {datetime.datetime.now()} (bin={HERMES_BIN})", flush=True)
    # cron 环境下把 hermes 所在目录补进 PATH, 防止子进程再 fork 其他 hermes 相关命令时找不到
    env = dict(os.environ)
    hermes_dir = os.path.dirname(HERMES_BIN)
    if hermes_dir not in env.get("PATH", "").split(":"):
        env["PATH"] = hermes_dir + ":" + env.get("PATH", "")
    try:
        # 不重定向 stdout: Hermes 自行用 write_file 写 REPORT_PATH
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, cwd=str(ROOT), env=env)
        proc.wait(timeout=1800)  # 30 分钟硬上限
        print(f"[run_research] Hermes 退出码 {proc.returncode}", flush=True)
    except subprocess.TimeoutExpired:
        print("[run_research] 超过30分钟上限, 强制结束", flush=True)
    # 无论退出码, 检查简报是否生成
    if REPORT_PATH.exists() and REPORT_PATH.stat().st_size > 0:
        print(f"[run_research] 报告就绪: {REPORT_PATH}", flush=True)
        return True
    print(f"[run_research] 报告未生成: {REPORT_PATH}", flush=True)
    return False


def send_email(report_path: Path):
    """读报告, 统一走中枢 send_email.py (To=126 + Cc=163 由 .env 兜底). 凭据由中枢读取."""
    SEND_CLI = Path.home() / "workspace/ng/skills/common/send-email/send_email.py"
    subject = f"SSQ预测方法研究简报 - {TODAY}"
    cmd = [sys.executable, str(SEND_CLI),
           "--subject", subject, "--body-file", str(report_path)]
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
    ok = run_research()
    if not ok:
        print("[main] 研究未产出有效报告, 跳过邮件", flush=True)
        sys.exit(1)
    try:
        send_email(REPORT_PATH)
    except Exception as e:
        print(f"[main] 邮件发送失败: {e}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
