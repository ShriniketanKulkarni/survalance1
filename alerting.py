# ─── alerting.py ──────────────────────────────────────────────────────────────
"""Alert manager — stores history, plays sound, deduplicates, sends email."""

import threading
import time
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque
import config

MAX_HISTORY = 200

SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}

# Severity → colour for UI
SEV_COLOR = {
    "HIGH":   "#FF4136",
    "MEDIUM": "#FF851B",
    "LOW":    "#FFDC00",
}

# ── Email sender ───────────────────────────────────────────────────────────────
class _EmailSender:
    """Thread-pool-free background email dispatcher."""
    def __init__(self):
        self._last_sent: dict[str, float] = {}  # alert_type → timestamp
        self._lock = threading.Lock()

    def should_send(self, alert_type: str, severity: str) -> bool:
        if not config.EMAIL_ENABLED:
            return False
        if not config.EMAIL_SENDER or not config.EMAIL_PASSWORD:
            return False
        if not config.EMAIL_RECIPIENTS:
            return False
        # Severity gate
        min_order = SEV_ORDER.get(config.EMAIL_MIN_SEVERITY, 1)
        alert_order = SEV_ORDER.get(severity, 3)
        if alert_order > min_order:
            return False
        # Cooldown per alert type
        with self._lock:
            last = self._last_sent.get(alert_type, 0)
            if time.time() - last < config.EMAIL_COOLDOWN_SEC:
                return False
            self._last_sent[alert_type] = time.time()
        return True

    def send_async(self, alert: dict, snapshot_b64: str | None = None):
        t = threading.Thread(target=self._send, args=(dict(alert), snapshot_b64), daemon=True)
        t.start()

    def _send(self, alert: dict, snapshot_b64: str | None):
        try:
            sev   = alert.get("severity", "")
            atype = alert.get("type", "Alert")
            msg_text = alert.get("message", "")
            cam   = alert.get("label", alert.get("cam_id", ""))
            ts    = alert.get("time_str", time.strftime("%H:%M:%S"))

            sev_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "⚠")

            subject = f"[{sev}] {sev_emoji} Surveillance Alert — {atype}"

            html_body = f"""
<html><body style="font-family:Arial,sans-serif;background:#0f1521;color:#c9d4ed;padding:20px">
<div style="max-width:540px;margin:auto;background:#131c2b;border-radius:12px;overflow:hidden;border:1px solid #1a2540">
  <div style="background:linear-gradient(135deg,#3d7fff,#a855f7);padding:18px 22px">
    <h2 style="margin:0;color:#fff;font-size:1.1rem">🛡️ Smart Surveillance System</h2>
    <p style="margin:4px 0 0;color:rgba(255,255,255,0.75);font-size:0.8rem">CodeFiesta 6.0 · Real-time Alert</p>
  </div>
  <div style="padding:22px">
    <div style="background:#0c1018;border-radius:8px;padding:14px 16px;margin-bottom:16px;border-left:4px solid {'#ef4444' if sev=='CRITICAL' else '#f97316' if sev=='HIGH' else '#fbbf24' if sev=='MEDIUM' else '#22c55e'}">
      <div style="font-size:0.7rem;color:#8896b3;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Alert Type</div>
      <div style="font-size:1.15rem;font-weight:700;color:#fff">{sev_emoji} {atype}</div>
    </div>
    <table style="width:100%;border-collapse:collapse;font-size:0.85rem">
      <tr><td style="padding:6px 0;color:#8896b3;width:110px">Severity</td><td style="color:#fff;font-weight:600">{sev}</td></tr>
      <tr><td style="padding:6px 0;color:#8896b3">Camera</td><td style="color:#fff">{cam or '—'}</td></tr>
      <tr><td style="padding:6px 0;color:#8896b3">Time</td><td style="color:#fff">{ts}</td></tr>
      <tr><td style="padding:6px 0;color:#8896b3;vertical-align:top">Details</td><td style="color:#c9d4ed">{msg_text}</td></tr>
    </table>
    {'<div style="margin-top:16px"><img src="cid:snapshot" style="width:100%;border-radius:8px;border:1px solid #1a2540" alt="Snapshot"/></div>' if snapshot_b64 else ''}
  </div>
  <div style="padding:12px 22px;border-top:1px solid #1a2540;font-size:0.7rem;color:#4a5578;text-align:center">
    Sent automatically by Smart Surveillance System · Do not reply
  </div>
</div>
</body></html>"""

            mime = MIMEMultipart("related")
            mime["Subject"] = subject
            mime["From"]    = f"Smart Surveillance <{config.EMAIL_SENDER}>"
            mime["To"]      = ", ".join(config.EMAIL_RECIPIENTS)

            alt_part = MIMEMultipart("alternative")
            alt_part.attach(MIMEText(f"[{sev}] {atype}: {msg_text} (Camera: {cam}) at {ts}", "plain"))
            alt_part.attach(MIMEText(html_body, "html"))
            mime.attach(alt_part)

            # Attach snapshot inline if provided
            if snapshot_b64:
                import base64
                from email.mime.image import MIMEImage
                img_data = base64.b64decode(snapshot_b64)
                img_part = MIMEImage(img_data, _subtype="jpeg")
                img_part.add_header("Content-ID", "<snapshot>")
                img_part.add_header("Content-Disposition", "inline", filename="snapshot.jpg")
                mime.attach(img_part)

            ctx = ssl.create_default_context()
            with smtplib.SMTP(config.EMAIL_SMTP_HOST, config.EMAIL_SMTP_PORT, timeout=10) as s:
                s.ehlo()
                s.starttls(context=ctx)
                s.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
                s.sendmail(config.EMAIL_SENDER, config.EMAIL_RECIPIENTS, mime.as_string())
            print(f"[Email] ✉ Sent '{atype}' alert to {config.EMAIL_RECIPIENTS}")
        except Exception as e:
            print(f"[Email] ✗ Failed to send email: {e}")


_emailer = _EmailSender()


class AlertManager:
    def __init__(self):
        self.history = deque(maxlen=MAX_HISTORY)   # list of dicts
        self._lock   = threading.Lock()
        self._last_beep = 0

    def push(self, alert, frame_snapshot=None):
        """
        Deduplicate: don't re-push same alert type+id within 5 seconds.
        """
        now = time.time()
        key = (alert["type"], alert.get("track_id"))

        with self._lock:
            # Check dedupe
            for prev in reversed(list(self.history)):
                if (prev["type"], prev.get("track_id")) == key:
                    if now - prev["ts"] < 5:
                        return   # duplicate within 5s — skip
                    break

            entry = {**alert, "ts": now, "time_str": time.strftime("%H:%M:%S")}
            if frame_snapshot is not None:
                entry["snapshot"] = frame_snapshot   # base64 JPEG
            self.history.appendleft(entry)

        sev = alert.get("severity", "LOW")

        # Beep (Windows only) — throttle to 1s
        if config.ALERT_SOUND and now - self._last_beep > 1:
            self._last_beep = now
            threading.Thread(target=self._beep, args=(sev,), daemon=True).start()

        # Email notification
        if _emailer.should_send(alert["type"], sev):
            _emailer.send_async(alert, frame_snapshot)

    def get_history(self, n=50):
        with self._lock:
            return list(self.history)[:n]

    def clear(self):
        with self._lock:
            self.history.clear()

    @staticmethod
    def _beep(severity):
        try:
            import winsound
            freq = 1200 if severity in ("CRITICAL", "HIGH") else 800
            winsound.Beep(freq, 200)
        except Exception:
            pass
