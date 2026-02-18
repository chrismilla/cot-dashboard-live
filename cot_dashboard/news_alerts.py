from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import requests

DEFAULT_TOPIC = os.getenv("NEWS_ALERT_TOPIC", "cot-live-chrismilla-8c73")
DEFAULT_LEAD_MINUTES = int(os.getenv("NEWS_ALERT_LEAD_MINUTES", "30"))
DEFAULT_DASHBOARD_URL = os.getenv("NEWS_ALERT_DASHBOARD_URL", "https://chrismilla.github.io/cot-dashboard-live/")


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _event_key(event: dict[str, Any]) -> str:
    return "|".join(
        [
            str(event.get("datetime_utc") or ""),
            str(event.get("country") or ""),
            str(event.get("title") or ""),
        ]
    )


@dataclass
class AlertState:
    sent: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "AlertState":
        if not path.exists():
            return cls(sent={})
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return cls(sent={})
        sent = payload.get("sent")
        if not isinstance(sent, dict):
            sent = {}
        return cls(sent={str(k): str(v) for k, v in sent.items()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"updated_at_utc": _fmt_utc(datetime.now(timezone.utc)), "sent": self.sent}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def prune(self, now_utc: datetime) -> None:
        keep: dict[str, str] = {}
        cutoff = now_utc - timedelta(days=3)
        for key, sent_at in self.sent.items():
            try:
                sent_dt = _parse_iso_utc(sent_at)
            except Exception:
                continue
            if sent_dt >= cutoff:
                keep[key] = sent_at
        self.sent = keep


def _load_snapshot_events(snapshot_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    events = payload.get("forex_factory", {}).get("upcoming_red_folder", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _minutes_to_event(event_utc: datetime, now_utc: datetime) -> int:
    return int(round((event_utc - now_utc).total_seconds() / 60))


def _send_ntfy(topic: str, event: dict[str, Any], minutes_left: int, dashboard_url: str) -> None:
    title = f"{event.get('country', 'USD')} red news in {minutes_left}m"
    body = f"{event.get('title', 'Major event')} at {event.get('datetime_utc', '--')} UTC"
    headers = {
        "Title": title,
        "Priority": "high",
        "Tags": "warning,chart_with_upwards_trend",
        "Click": dashboard_url,
    }
    url = f"https://ntfy.sh/{quote(topic, safe='')}"
    response = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=20)
    response.raise_for_status()


def run(
    snapshot: Path,
    state_path: Path,
    topic: str,
    lead_minutes: int,
    dashboard_url: str,
    dry_run: bool = False,
) -> int:
    if lead_minutes <= 0:
        raise ValueError("lead_minutes must be positive")

    if not topic:
        print("No push topic configured. Skipping alerts.")
        return 0

    now_utc = datetime.now(timezone.utc)
    events = _load_snapshot_events(snapshot)
    state = AlertState.load(state_path)
    state.prune(now_utc)

    sent_count = 0
    for event in events:
        stamp = str(event.get("datetime_utc") or "")
        if not stamp:
            continue
        try:
            event_utc = _parse_iso_utc(stamp)
        except Exception:
            continue

        minutes_left = _minutes_to_event(event_utc, now_utc)
        if minutes_left < 0 or minutes_left > lead_minutes:
            continue

        key = _event_key(event)
        if key in state.sent:
            continue

        if dry_run:
            print(f"[dry-run] Would alert: {event.get('country')} {event.get('title')} in {minutes_left}m")
        else:
            _send_ntfy(topic=topic, event=event, minutes_left=minutes_left, dashboard_url=dashboard_url)
            print(f"Sent alert: {event.get('country')} {event.get('title')} in {minutes_left}m")

        state.sent[key] = _fmt_utc(now_utc)
        sent_count += 1

    if not dry_run:
        state.save(state_path)
    return sent_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send push alerts before major red-folder events")
    parser.add_argument("--snapshot", type=Path, default=Path("data/cot_snapshot.json"))
    parser.add_argument("--state", type=Path, default=Path("data/news_alert_state.json"))
    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC)
    parser.add_argument("--lead-minutes", type=int, default=DEFAULT_LEAD_MINUTES)
    parser.add_argument("--dashboard-url", type=str, default=DEFAULT_DASHBOARD_URL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        sent = run(
            snapshot=args.snapshot,
            state_path=args.state,
            topic=args.topic.strip(),
            lead_minutes=args.lead_minutes,
            dashboard_url=args.dashboard_url,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        parser.error(str(exc))
        return

    print(f"Alert check complete. Sent {sent} alert(s).")


if __name__ == "__main__":
    main()
