"""发送 SSQ 工程化改进最终报告给 Rocky（统一走 send_email.py 中枢）。

收件标准: To=126 + Cc=163 由 .env 兜底 (EMAIL_TO/EMAIL_CC 或默认)。
"""
import sys
from pathlib import Path

REPORT = Path(__file__).resolve().parent / "SSQ工程化改进_final_report.md"
SEND_EMAIL_CLI = Path.home() / "workspace/ng/skills/common/send-email/send_email.py"
SUBJECT = "【SSQ 双色球预测项目】工程化改进与对比测试最终报告 (Hermes Agent / 奇思)"


def main():
    if not REPORT.exists():
        print(f"FAIL: 报告不存在 {REPORT}", file=sys.stderr)
        sys.exit(1)
    cmd = [sys.executable, str(SEND_EMAIL_CLI),
           "--subject", SUBJECT, "--body-file", str(REPORT)]
    import subprocess
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if cp.returncode != 0:
        print(f"FAIL: {cp.stderr.strip()[:200]}", file=sys.stderr)
        sys.exit(1)
    print(cp.stdout.strip())


if __name__ == "__main__":
    main()
