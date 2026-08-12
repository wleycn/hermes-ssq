"""通过 163 SMTP 发送 SSQ 深度分析报告给 Rocky。
凭据从 ~/.hermes/.env 读取；收件人 wleycn@126.com（在邮箱网关 allowlist 内）。
纯标准库 smtplib + ssl，无第三方依赖。
"""
import os, smtplib, ssl, sys
from email.message import EmailMessage
from pathlib import Path

ENV_PATH = Path.home() / ".hermes/.env"
REPORT = Path(__file__).resolve().parent / "SSQ深度分析报告.md"

def load_env(p: Path) -> dict:
    d = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        d[k.strip()] = v.strip()
    return d

def main():
    env = load_env(ENV_PATH)
    user = env.get("EMAIL_ADDRESS")
    pwd = env.get("EMAIL_PASSWORD")
    host = env.get("EMAIL_SMTP_HOST", "smtp.163.com")
    port = int(env.get("EMAIL_SMTP_PORT", "465"))
    if not (user and pwd):
        print("FAIL: 缺少 EMAIL_ADDRESS / EMAIL_PASSWORD", file=sys.stderr); sys.exit(1)

    to = "wleycn@126.com"   # Rocky 个人邮箱（allowlist 内）
    body = REPORT.read_text(encoding="utf-8")
    subject = "【SSQ 双色球预测项目】深度分析与改进建议 (Hermes Agent)"

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(host, port, context=ctx) as s:
        s.login(user, pwd)
        s.send_message(msg)
    print(f"OK: 报告已发送至 {to} (主题: {subject})")

if __name__ == "__main__":
    main()
