# alerting.py
import time, json, os
import requests
from config import WEBHOOK_URL, SMTP_ENABLED, SMTP_CONFIG, CLIPS_DIR, CAMERA_ID

def send_webhook(payload: dict):
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        return r.ok
    except Exception as e:
        print("Webhook error:", e)
        return False

def send_email(subject: str, body: str):
    if not SMTP_ENABLED:
        return False
    import smtplib
    from email.mime.text import MIMEText
    cfg = SMTP_CONFIG
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = cfg['user']
    msg['To'] = ','.join(cfg['to'])
    try:
        s = smtplib.SMTP(cfg['host'], cfg['port'], timeout=10)
        s.starttls()
        s.login(cfg['user'], cfg['password'])
        s.sendmail(cfg['user'], cfg['to'], msg.as_string())
        s.quit()
        return True
    except Exception as e:
        print("Email send failed", e)
        return False

def raise_alert(camera_id: str, clip_path: str, details: dict):
    ts = time.time()
    payload = {
        "camera_id": camera_id,
        "timestamp": ts,
        "clip_path": clip_path,
        "details": details
    }
    ok = send_webhook(payload)
    if SMTP_ENABLED:
        subject = f"PPE Violation - {camera_id} - {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"
        body = json.dumps(payload, indent=2)
        send_email(subject, body)
    # Save metadata
    md_path = clip_path + ".json"
    with open(md_path, "w") as f:
        json.dump(payload, f)
    return ok
