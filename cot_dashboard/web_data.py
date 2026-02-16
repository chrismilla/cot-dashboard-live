import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .data import (
    COTDataError,
    DEFAULT_COT_URL,
    DEFAULT_INDEX_ORDER,
    INDEX_MARKETS,
    filter_market_with_code,
    load_financials,
    normalize_frame,
    summarize_latest,
)


def _to_history_rows(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for _, row in df.iterrows():
        rows.append(
            {
                "report_date": row["report_date"].date().isoformat(),
                "asset_mgr_net": int(row["asset_mgr_net"]),
                "leveraged_net": int(row["leveraged_net"]),
                "open_interest": int(row["open_interest"]),
            }
        )
    return rows


def _build_market_payload(df: pd.DataFrame, market_key: str, history_weeks: int) -> dict:
    filtered, contract_code = filter_market_with_code(df, market_key=market_key)
    filtered = filtered.sort_values("report_date").tail(history_weeks).copy()

    summary = summarize_latest(filtered)
    latest_row = filtered.iloc[-1]
    previous_row = filtered.iloc[-2] if len(filtered) > 1 else None

    return {
        "key": market_key,
        "label": INDEX_MARKETS[market_key].label,
        "contract_code": contract_code,
        "market_name": str(latest_row["market_name"]),
        "latest_report_date": summary.iso_date,
        "latest": {
            "asset_mgr_net": summary.asset_manager_net,
            "leveraged_net": summary.leveraged_net,
            "open_interest": summary.open_interest,
            "asset_mgr_change": summary.asset_manager_change,
            "leveraged_change": summary.leveraged_change,
            "open_interest_change": summary.open_interest_change,
        },
        "previous_report_date": previous_row["report_date"].date().isoformat() if previous_row is not None else None,
        "history": _to_history_rows(filtered),
    }


def build_snapshot(
    output: Path,
    force_refresh: bool = False,
    url: Optional[str] = None,
    history_weeks: int = 104,
) -> Path:
    if history_weeks < 2:
        raise ValueError("history_weeks must be >= 2")

    raw = load_financials(force_refresh=force_refresh, url=url)
    normalized = normalize_frame(raw)

    markets = {
        key: _build_market_payload(normalized, market_key=key, history_weeks=history_weeks) for key in DEFAULT_INDEX_ORDER
    }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_url": url or DEFAULT_COT_URL,
        "history_weeks": history_weeks,
        "markets": markets,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build static JSON snapshot for the multi-index COT web dashboard")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("web/data/cot_snapshot.json"),
        help="Path to write JSON snapshot",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Ignore cache and refetch from the CFTC endpoint",
    )
    parser.add_argument("--url", type=str, help="Override CFTC data URL")
    parser.add_argument(
        "--history-weeks",
        type=int,
        default=104,
        help="Number of weekly rows to include per market",
    )
    return parser


def main(argv: Optional[list[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_file = build_snapshot(
            output=args.output,
            force_refresh=args.force_refresh,
            url=args.url,
            history_weeks=args.history_weeks,
        )
    except (COTDataError, ValueError) as exc:
        parser.error(str(exc))
        return

    print(f"Wrote dashboard snapshot to {output_file}")


if __name__ == "__main__":
    main()
