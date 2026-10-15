import os
import sys
import smtplib
import socket
from datetime import datetime
from email.message import EmailMessage

exit_code = int(sys.argv[1])

sender = os.environ["SMTP_USER"]
password = os.environ["SMTP_PASSWORD"]
receiver = os.environ["SMTP_TO"]
host = os.environ["SMTP_HOST"]

status = "成功完成" if exit_code == 0 else "运行失败"

msg = EmailMessage()
msg["From"] = sender
msg["To"] = receiver
msg["Subject"] = f"[GPTQ] 实验{status}"

msg.set_content(
    f"实验状态：{status}\n"
    f"退出码：{exit_code}\n"
    f"服务器：{socket.gethostname()}\n"
    f"结束时间：{datetime.now().astimezone()}\n"
    f"日志目录：./logs/\n"
)

with smtplib.SMTP_SSL(host, 465, timeout=30) as smtp:
    smtp.login(sender, password)
    smtp.send_message(msg)

print("Email notification sent.")