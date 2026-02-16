import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import requests

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

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
INTRADAY_TICKERS = {
    "dow_jones": "DIA",
    "nasdaq_100": "QQQ",
    "sp_500": "SPY",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_from_epoch(epoch_seconds: Union[int, float]) -> str:
    return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _percentile_rank(series: pd.Series, value: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return 50.0
    return float((clean <= value).sum()) * 100.0 / float(len(clean))


def _bias_label(score: float) -> str:
    if score >= 35:
        return "Strong Bullish"
    if score >= 15:
        return "Bullish"
    if score <= -35:
        return "Strong Bearish"
    if score <= -15:
        return "Bearish"
    return "Neutral"


def _bias_playbook(score: float) -> str:
    if score >= 25:
        return "Buy dips and prioritize long continuation setups."
    if score >= 10:
        return "Leaning long; fade only weak rallies."
    if score <= -25:
        return "Sell pops and prioritize short continuation setups."
    if score <= -10:
        return "Leaning short; avoid aggressive long chases."
    return "Two-way environment; trade reaction levels, reduce size."


def _signed(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _compute_bias(df: pd.DataFrame) -> dict:
    asset = pd.to_numeric(df["asset_mgr_net"], errors="coerce")
    leveraged = pd.to_numeric(df["leveraged_net"], errors="coerce")
    open_interest = pd.to_numeric(df["open_interest"], errors="coerce")

    latest_asset = float(asset.iloc[-1])
    latest_lev = float(leveraged.iloc[-1])
    latest_oi = float(open_interest.iloc[-1])

    asset_pct = _percentile_rank(asset, latest_asset)
    lev_pct = _percentile_rank(leveraged, latest_lev)

    asset_4w_delta = float(asset.iloc[-1] - asset.iloc[-5]) if len(asset) >= 5 else float(asset.iloc[-1] - asset.iloc[0])
    lev_4w_delta = (
        float(leveraged.iloc[-1] - leveraged.iloc[-5]) if len(leveraged) >= 5 else float(leveraged.iloc[-1] - leveraged.iloc[0])
    )
    oi_4w_delta = (
        float(open_interest.iloc[-1] - open_interest.iloc[-5])
        if len(open_interest) >= 5
        else float(open_interest.iloc[-1] - open_interest.iloc[0])
    )

    asset_4w_series = (asset - asset.shift(4)).dropna()
    lev_4w_series = (leveraged - leveraged.shift(4)).dropna()

    asset_momentum_pct = _percentile_rank(asset_4w_series, asset_4w_delta)
    lev_momentum_pct = _percentile_rank(lev_4w_series, lev_4w_delta)

    institutional_pressure = (asset_pct - 50.0) * 2.0
    fast_money_pressure = (lev_pct - 50.0) * 2.0
    institutional_momentum = (asset_momentum_pct - 50.0) * 2.0
    fast_money_momentum = (lev_momentum_pct - 50.0) * 2.0

    same_direction = _signed(latest_asset) == _signed(latest_lev) and _signed(latest_asset) != 0
    alignment_bonus = 8.0 if same_direction else -8.0

    raw_score = (
        institutional_pressure * 0.45
        + fast_money_pressure * 0.20
        + institutional_momentum * 0.20
        + fast_money_momentum * 0.15
        + alignment_bonus
    )
    bias_score = _clamp(raw_score, -100.0, 100.0)

    return {
        "score": round(bias_score, 1),
        "label": _bias_label(bias_score),
        "playbook": _bias_playbook(bias_score),
        "conviction": round(_clamp(abs(bias_score), 0.0, 100.0), 1),
        "asset_percentile": round(asset_pct, 1),
        "leveraged_percentile": round(lev_pct, 1),
        "asset_4w_delta": int(round(asset_4w_delta)),
        "leveraged_4w_delta": int(round(lev_4w_delta)),
        "open_interest_4w_delta": int(round(oi_4w_delta)),
        "components": {
            "institutional_pressure": round(institutional_pressure, 1),
            "fast_money_pressure": round(fast_money_pressure, 1),
            "institutional_momentum": round(institutional_momentum, 1),
            "fast_money_momentum": round(fast_money_momentum, 1),
            "alignment": round(alignment_bonus, 1),
        },
        "divergence": {
            "asset_vs_leveraged": int(round(latest_asset - latest_lev)),
            "same_direction": same_direction,
        },
        "as_of_report_date": df.iloc[-1]["report_date"].date().isoformat(),
        "open_interest_latest": int(round(latest_oi)),
    }


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        parsed = float(value)
        if math.isnan(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _fetch_intraday_snapshot(symbol: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 (cot-dashboard bot)"}
    params = {"interval": "5m", "range": "1d", "includePrePost": "false", "events": "history"}

    try:
        response = requests.get(YAHOO_CHART_URL.format(symbol=symbol), params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return {
            "symbol": symbol,
            "source": "yahoo",
            "as_of_utc": _utc_now_iso(),
            "error": str(exc),
        }

    result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
    if not result:
        return {
            "symbol": symbol,
            "source": "yahoo",
            "as_of_utc": _utc_now_iso(),
            "error": "No result payload",
        }

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    bars = []
    for idx, timestamp in enumerate(timestamps):
        close_value = _safe_float(closes[idx] if idx < len(closes) else None)
        if close_value is None:
            continue

        open_value = _safe_float(opens[idx] if idx < len(opens) else None) or close_value
        high_value = _safe_float(highs[idx] if idx < len(highs) else None) or max(open_value, close_value)
        low_value = _safe_float(lows[idx] if idx < len(lows) else None) or min(open_value, close_value)
        volume_value = _safe_float(volumes[idx] if idx < len(volumes) else None) or 0.0

        bars.append(
            {
                "t": _iso_from_epoch(timestamp),
                "o": round(open_value, 4),
                "h": round(high_value, 4),
                "l": round(low_value, 4),
                "c": round(close_value, 4),
                "v": int(volume_value),
            }
        )

    if not bars:
        return {
            "symbol": symbol,
            "source": "yahoo",
            "as_of_utc": _utc_now_iso(),
            "error": "No price bars returned",
        }

    day_open = bars[0]["o"]
    day_high = max(bar["h"] for bar in bars)
    day_low = min(bar["l"] for bar in bars)
    last_price = bars[-1]["c"]
    day_volume = int(sum(bar["v"] for bar in bars))

    previous_close = _safe_float(meta.get("previousClose")) or _safe_float(meta.get("chartPreviousClose")) or day_open
    change = float(last_price - previous_close)
    change_pct = float((change / previous_close) * 100.0) if previous_close else 0.0
    from_open_pct = float(((last_price - day_open) / day_open) * 100.0) if day_open else 0.0
    range_span = float(day_high - day_low)
    range_position_pct = float(((last_price - day_low) / range_span) * 100.0) if range_span > 0 else 50.0

    return {
        "symbol": symbol,
        "source": "yahoo",
        "as_of_utc": _utc_now_iso(),
        "market_state": meta.get("marketState"),
        "timezone": meta.get("exchangeTimezoneName"),
        "regular_market_time_utc": _iso_from_epoch(meta["regularMarketTime"]) if meta.get("regularMarketTime") else None,
        "previous_close": round(previous_close, 4),
        "last": round(last_price, 4),
        "change": round(change, 4),
        "change_pct": round(change_pct, 4),
        "from_open_pct": round(from_open_pct, 4),
        "day_open": round(day_open, 4),
        "day_high": round(day_high, 4),
        "day_low": round(day_low, 4),
        "day_volume": day_volume,
        "range_position_pct": round(_clamp(range_position_pct, 0.0, 100.0), 2),
        # Keep chart payload light for fast page loads.
        "bars": [{"t": bar["t"], "c": bar["c"]} for bar in bars],
    }


def _fetch_intraday_by_market() -> dict[str, dict]:
    snapshots: dict[str, dict] = {}
    for market_key, symbol in INTRADAY_TICKERS.items():
        snapshots[market_key] = _fetch_intraday_snapshot(symbol)
    return snapshots


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


def _build_market_payload(
    df: pd.DataFrame,
    market_key: str,
    history_weeks: int,
    intraday_by_market: dict[str, dict],
) -> dict:
    filtered, contract_code = filter_market_with_code(df, market_key=market_key)
    filtered = filtered.sort_values("report_date").tail(history_weeks).copy()

    summary = summarize_latest(filtered)
    latest_row = filtered.iloc[-1]
    previous_row = filtered.iloc[-2] if len(filtered) > 1 else None
    bias = _compute_bias(filtered)

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
        "bias": bias,
        "intraday": intraday_by_market.get(market_key),
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
    intraday_by_market = _fetch_intraday_by_market()

    markets = {
        key: _build_market_payload(
            normalized,
            market_key=key,
            history_weeks=history_weeks,
            intraday_by_market=intraday_by_market,
        )
        for key in DEFAULT_INDEX_ORDER
    }

    payload = {
        "generated_at_utc": _utc_now_iso(),
        "source_url": url or DEFAULT_COT_URL,
        "intraday_source": "Yahoo Finance chart API",
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
