from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

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

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
MARKET_SYMBOLS: dict[str, str] = {
    "dow_jones": "DIA",
    "nasdaq_100": "QQQ",
    "sp_500": "SPY",
}

WATCHLIST_CANDIDATES: list[dict[str, Optional[str]]] = [
    {"symbol": "DIA", "label": "Dow (DIA)", "market_key": "dow_jones"},
    {"symbol": "QQQ", "label": "Nasdaq-100 (QQQ)", "market_key": "nasdaq_100"},
    {"symbol": "SPY", "label": "S&P 500 (SPY)", "market_key": "sp_500"},
    {"symbol": "ES=F", "label": "E-mini S&P (ES)", "market_key": None},
    {"symbol": "NQ=F", "label": "E-mini Nasdaq (NQ)", "market_key": None},
    {"symbol": "YM=F", "label": "Mini Dow (YM)", "market_key": None},
    {"symbol": "RTY=F", "label": "Russell 2000 (RTY)", "market_key": None},
    {"symbol": "CL=F", "label": "Crude Oil (CL)", "market_key": None},
    {"symbol": "GC=F", "label": "Gold (GC)", "market_key": None},
    {"symbol": "6E=F", "label": "Euro FX (6E)", "market_key": None},
]

FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_label(score: int) -> str:
    if score >= 60:
        return "BULLISH"
    if score <= 40:
        return "BEARISH"
    return "NEUTRAL"


def _fmt_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _next_cot_release_utc(now_utc: datetime) -> str:
    now_et = now_utc.astimezone(NY_TZ)
    days_until_friday = (4 - now_et.weekday()) % 7
    release_et = (now_et + timedelta(days=days_until_friday)).replace(hour=15, minute=30, second=0, microsecond=0)
    if now_et.weekday() == 4 and now_et >= release_et:
        release_et = release_et + timedelta(days=7)
    return _fmt_utc(release_et)


def _to_history_rows(df: pd.DataFrame) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
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


def _percentile_rank(values: pd.Series, value: float) -> int:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return 50
    return int(round((series.le(value).sum() / len(series)) * 100))


def _to_bias_score(raw: float) -> int:
    return int(round((_clamp(raw, -1.0, 1.0) + 1.0) * 50))


def _compute_cot_bias(filtered: pd.DataFrame) -> dict[str, Any]:
    latest = filtered.iloc[-1]
    previous = filtered.iloc[-2] if len(filtered) > 1 else latest

    open_interest = max(float(latest["open_interest"]), 1.0)
    asset_ratio = float(latest["asset_mgr_net"]) / open_interest
    leveraged_ratio = float(latest["leveraged_net"]) / open_interest

    net_change = (float(latest["asset_mgr_net"]) - float(previous["asset_mgr_net"])) + (
        float(latest["leveraged_net"]) - float(previous["leveraged_net"])
    )
    change_ratio = net_change / open_interest

    institutional_raw = _clamp(asset_ratio / 0.16, -1.0, 1.0)
    fast_money_raw = _clamp(leveraged_ratio / 0.16, -1.0, 1.0)
    momentum_raw = _clamp(change_ratio / 0.04, -1.0, 1.0)

    if abs(asset_ratio) < 0.004 or abs(leveraged_ratio) < 0.004:
        alignment_raw = 0.0
    else:
        alignment_raw = 1.0 if asset_ratio * leveraged_ratio > 0 else -1.0

    score_raw = 0.42 * institutional_raw + 0.28 * fast_money_raw + 0.2 * momentum_raw + 0.1 * alignment_raw
    score = _to_bias_score(score_raw)

    lookback = filtered.tail(156)
    asset_percentile = _percentile_rank(lookback["asset_mgr_net"], float(latest["asset_mgr_net"]))
    leveraged_percentile = _percentile_rank(lookback["leveraged_net"], float(latest["leveraged_net"]))

    return {
        "score": score,
        "label": _score_label(score),
        "components": {
            "institutional_pressure": _to_bias_score(institutional_raw),
            "fast_money_pressure": _to_bias_score(fast_money_raw),
            "position_momentum": _to_bias_score(momentum_raw),
            "alignment": 100 if alignment_raw > 0 else 20 if alignment_raw < 0 else 50,
        },
        "extremes": {
            "asset_manager_percentile_3y": asset_percentile,
            "leveraged_percentile_3y": leveraged_percentile,
        },
    }


def _pct_delta(current: float, reference: float) -> float:
    if reference == 0:
        return 0.0
    return ((current - reference) / abs(reference)) * 100


def _fetch_yahoo_intraday(symbol: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "interval": "5m",
        "range": "5d",
        "includePrePost": "false",
        "events": "div,splits",
    }
    headers = {"User-Agent": "cot-dashboard/1.1"}

    response = requests.get(url, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()

    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"No Yahoo data for {symbol}")

    blob = result[0]
    timestamps = blob.get("timestamp") or []
    quote = (blob.get("indicators", {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    opens = quote.get("open") or []
    volumes = quote.get("volume") or []

    rows: list[dict[str, Any]] = []
    for idx, ts in enumerate(timestamps):
        close_value = closes[idx] if idx < len(closes) else None
        if close_value is None:
            continue
        rows.append(
            {
                "timestamp": int(ts),
                "open": float(opens[idx]) if idx < len(opens) and opens[idx] is not None else float(close_value),
                "high": float(highs[idx]) if idx < len(highs) and highs[idx] is not None else float(close_value),
                "low": float(lows[idx]) if idx < len(lows) and lows[idx] is not None else float(close_value),
                "close": float(close_value),
                "volume": int(volumes[idx]) if idx < len(volumes) and volumes[idx] is not None else 0,
            }
        )

    if not rows:
        raise RuntimeError(f"Yahoo payload for {symbol} had no usable bars")

    df = pd.DataFrame(rows)
    df["dt_utc"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)

    meta = blob.get("meta") or {}
    return df, meta


def _regular_session(df: pd.DataFrame, exchange_tz: str) -> pd.DataFrame:
    timezone_name = exchange_tz or "America/New_York"
    working = df.copy()
    working["dt_local"] = working["dt_utc"].dt.tz_convert(timezone_name)
    working["session_date"] = working["dt_local"].dt.strftime("%Y-%m-%d")
    working["local_time"] = working["dt_local"].dt.time

    regular = working.loc[
        working["local_time"].between(time(9, 30), time(16, 0), inclusive="both")
    ].copy()
    if regular.empty:
        regular = working
    return regular


def _spark_rows(frame: pd.DataFrame, max_points: int = 120) -> list[dict[str, Any]]:
    subset = frame.tail(max_points)
    points: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        points.append(
            {
                "t": _fmt_utc(row["dt_utc"]),
                "p": round(float(row["close"]), 4),
                "v": int(row["volume"]),
            }
        )
    return points


def _build_intraday_payload(symbol: str, label: str) -> dict[str, Any]:
    bars, meta = _fetch_yahoo_intraday(symbol)
    exchange_tz = str(meta.get("exchangeTimezoneName") or "America/New_York")
    regular = _regular_session(bars, exchange_tz=exchange_tz)

    sessions = sorted(regular["session_date"].dropna().unique())
    if not sessions:
        raise RuntimeError(f"No session bars for {symbol}")

    today = regular.loc[regular["session_date"] == sessions[-1]].copy()
    yesterday = regular.loc[regular["session_date"] == sessions[-2]].copy() if len(sessions) > 1 else pd.DataFrame()

    if today.empty:
        today = regular.tail(78).copy()

    price = float(today["close"].iloc[-1])

    prev_close = float(meta.get("previousClose") or 0)
    if prev_close <= 0 and not yesterday.empty:
        prev_close = float(yesterday["close"].iloc[-1])
    if prev_close <= 0:
        prev_close = float(today["close"].iloc[0])

    day_change_pct = _pct_delta(price, prev_close)

    vols = pd.to_numeric(today["volume"], errors="coerce").fillna(0)
    if float(vols.sum()) > 0:
        vwap_series = (today["close"] * vols).cumsum() / vols.cumsum().replace(0, pd.NA)
        vwap = float(vwap_series.ffill().iloc[-1])
    else:
        vwap = float(today["close"].mean())

    opening_window = today.head(min(6, len(today)))
    opening_high = float(opening_window["high"].max())
    opening_low = float(opening_window["low"].min())

    ema20 = float(today["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(today["close"].ewm(span=50, adjust=False).mean().iloc[-1])

    if not yesterday.empty:
        prior_high = float(yesterday["high"].max())
        prior_low = float(yesterday["low"].min())
        prior_close = float(yesterday["close"].iloc[-1])
    else:
        prior_high = float(today["high"].max())
        prior_low = float(today["low"].min())
        prior_close = float(today["close"].iloc[0])

    pivot = (prior_high + prior_low + prior_close) / 3.0

    if price > opening_high:
        opening_state = "Above OR High"
    elif price < opening_low:
        opening_state = "Below OR Low"
    else:
        opening_state = "Inside OR"

    if price > ema20 > ema50:
        ema_stack = "Bull Stack"
    elif price < ema20 < ema50:
        ema_stack = "Bear Stack"
    else:
        ema_stack = "Mixed Stack"

    if opening_state == "Above OR High" and price > vwap:
        structure = "AM Range Breakout Up"
    elif opening_state == "Below OR Low" and price < vwap:
        structure = "AM Range Breakout Down"
    elif abs(_pct_delta(price, vwap)) < 0.15:
        structure = "Rotation Around VWAP"
    else:
        structure = "Trend Continuation"

    score = 50
    score += 18 if price > vwap else -18
    score += 18 if ema_stack == "Bull Stack" else -18 if ema_stack == "Bear Stack" else 0
    score += 12 if price > pivot else -12
    score += 12 if opening_state == "Above OR High" else -12 if opening_state == "Below OR Low" else 0
    score += 8 if day_change_pct >= 0 else -8
    score += 6 if structure == "AM Range Breakout Up" else -6 if structure == "AM Range Breakout Down" else 0
    pulse_score = int(round(_clamp(score, 0, 100)))

    return {
        "symbol": symbol,
        "label": label,
        "as_of_utc": _fmt_utc(today["dt_utc"].iloc[-1]),
        "last_price": round(price, 4),
        "day_change_pct": round(day_change_pct, 2),
        "pulse_score": pulse_score,
        "pulse_label": _score_label(pulse_score),
        "metrics": {
            "vwap": round(vwap, 4),
            "price_vs_vwap_pct": round(_pct_delta(price, vwap), 2),
            "opening_range_high": round(opening_high, 4),
            "opening_range_low": round(opening_low, 4),
            "opening_range_state": opening_state,
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema_stack": ema_stack,
            "pivot": round(pivot, 4),
            "price_vs_pivot_pct": round(_pct_delta(price, pivot), 2),
            "prior_day_high": round(prior_high, 4),
            "prior_day_low": round(prior_low, 4),
            "price_vs_prior_high_pct": round(_pct_delta(price, prior_high), 2),
            "price_vs_prior_low_pct": round(_pct_delta(price, prior_low), 2),
            "session_structure": structure,
        },
        "spark": {
            "today": _spark_rows(today),
            "yesterday": _spark_rows(yesterday),
        },
    }


def _build_playbook(label: str, intraday: Optional[dict[str, Any]]) -> tuple[str, list[str]]:
    if intraday is None:
        if label == "BULLISH":
            return "Bias bullish: follow-through favored if pullbacks hold.", [
                "Buy pullbacks into trend support.",
                "Avoid chasing late extensions into weekly highs.",
                "Flip neutral if intraday structure weakens.",
            ]
        if label == "BEARISH":
            return "Bias bearish: rallies are fade candidates until structure repairs.", [
                "Sell failed bounces into resistance.",
                "Protect if price accepts back above key resistance.",
                "Avoid forcing shorts into high-impact news windows.",
            ]
        return "Bias neutral: expect two-way trade until a clean break confirms.", [
            "Trade edges, not the middle.",
            "Require confirmation before size-up.",
            "Respect scheduled high-impact news.",
        ]

    m = intraday["metrics"]
    vwap = m["vwap"]
    opening_high = m["opening_range_high"]
    opening_low = m["opening_range_low"]

    if label == "BULLISH":
        headline = f"Bias bullish: expect dips bought while price holds above {vwap:.2f}."
        bullets = [
            f"Long bias above VWAP {vwap:.2f}.",
            f"Momentum confirmation on breaks through OR high {opening_high:.2f}.",
            "If structure rotates below VWAP, reduce long aggression.",
        ]
    elif label == "BEARISH":
        headline = f"Bias bearish: favor failed bounces below {vwap:.2f}."
        bullets = [
            f"Short bias below VWAP {vwap:.2f}.",
            f"Acceleration setup under OR low {opening_low:.2f}.",
            "If price reclaims and holds above VWAP, stand down.",
        ]
    else:
        headline = f"Bias neutral: balance around {vwap:.2f}, wait for range break."
        bullets = [
            f"Respect OR boundaries {opening_low:.2f}-{opening_high:.2f}.",
            "Take quicker profits in choppy structure.",
            "Switch directional only after clean acceptance outside range.",
        ]

    return headline, bullets


def _build_confluence(cot_bias: dict[str, Any], intraday: Optional[dict[str, Any]]) -> dict[str, Any]:
    cot_score = int(cot_bias["score"])
    pulse_score = int(intraday["pulse_score"]) if intraday else cot_score
    confluence_score = int(round((cot_score * 0.55) + (pulse_score * 0.45)))
    label = _score_label(confluence_score)
    headline, bullets = _build_playbook(label=label, intraday=intraday)

    return {
        "score": confluence_score,
        "label": label,
        "cot_score": cot_score,
        "technical_score": pulse_score,
        "cot_weight": 55,
        "technical_weight": 45,
        "headline": headline,
        "bullets": bullets,
    }


def _parse_forex_factory_datetime(date_text: str, time_text: str) -> Optional[datetime]:
    date_clean = (date_text or "").strip()
    time_clean = (time_text or "").strip().lower()
    if not date_clean or not time_clean:
        return None
    if time_clean in {"all day", "tentative"}:
        return None

    compact = time_clean.replace(" ", "")
    try:
        parsed = datetime.strptime(f"{date_clean} {compact.upper()}", "%m-%d-%Y %I:%M%p")
    except ValueError:
        return None
    return parsed.replace(tzinfo=NY_TZ)


def _fetch_red_folder_events(limit: int = 10) -> dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    response = requests.get(FF_CALENDAR_URL, timeout=20, headers={"User-Agent": "cot-dashboard/1.1"})
    response.raise_for_status()

    root = ET.fromstring(response.text.encode("utf-8", errors="ignore"))
    all_events: list[dict[str, Any]] = []
    red_events: list[dict[str, Any]] = []

    for event in root.findall("event"):
        title = (event.findtext("title") or "").strip()
        country = (event.findtext("country") or "").strip()
        date_text = (event.findtext("date") or "").strip()
        time_text = (event.findtext("time") or "").strip()
        impact = (event.findtext("impact") or "").strip().title()
        forecast = (event.findtext("forecast") or "").strip()
        previous = (event.findtext("previous") or "").strip()
        url = (event.findtext("url") or "").strip()

        dt_et = _parse_forex_factory_datetime(date_text, time_text)
        dt_utc_iso = _fmt_utc(dt_et) if dt_et else None

        item = {
            "title": title,
            "country": country,
            "date": date_text,
            "time": time_text,
            "impact": impact,
            "forecast": forecast,
            "previous": previous,
            "url": url,
            "datetime_utc": dt_utc_iso,
        }

        all_events.append(item)
        if impact == "High":
            red_events.append(item)

    def sort_key(row: dict[str, Any]) -> tuple[int, datetime]:
        stamp = row.get("datetime_utc")
        if not stamp:
            return (1, datetime.max.replace(tzinfo=timezone.utc))
        try:
            dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            return (1, datetime.max.replace(tzinfo=timezone.utc))
        return (0, dt)

    red_events = sorted(red_events, key=sort_key)

    upcoming: list[dict[str, Any]] = []
    for event in red_events:
        stamp = event.get("datetime_utc")
        if not stamp:
            continue
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        if dt >= (now_utc - timedelta(hours=1)):
            upcoming.append(event)

    selected = (upcoming or red_events)[:limit]

    return {
        "source": FF_CALENDAR_URL,
        "fetched_at_utc": _fmt_utc(now_utc),
        "red_folder_count": len(red_events),
        "upcoming_red_folder": selected,
    }


def _build_market_payload(df: pd.DataFrame, market_key: str, history_weeks: int) -> dict[str, Any]:
    filtered, contract_code = filter_market_with_code(df, market_key=market_key)
    filtered = filtered.sort_values("report_date").tail(history_weeks).copy()

    summary = summarize_latest(filtered)
    latest_row = filtered.iloc[-1]
    previous_row = filtered.iloc[-2] if len(filtered) > 1 else None

    cot_bias = _compute_cot_bias(filtered)

    symbol = MARKET_SYMBOLS[market_key]
    intraday: Optional[dict[str, Any]] = None
    try:
        intraday = _build_intraday_payload(symbol=symbol, label=INDEX_MARKETS[market_key].label)
    except Exception:
        intraday = None

    confluence = _build_confluence(cot_bias=cot_bias, intraday=intraday)

    return {
        "key": market_key,
        "label": INDEX_MARKETS[market_key].label,
        "symbol": symbol,
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
        "cot_bias": cot_bias,
        "intraday": intraday,
        "confluence": confluence,
    }


def _build_watchlist(markets: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    watchlist: dict[str, dict[str, Any]] = {}
    market_by_symbol = {payload.get("symbol"): payload for payload in markets.values()}

    for candidate in WATCHLIST_CANDIDATES:
        symbol = str(candidate["symbol"])
        linked_market = market_by_symbol.get(symbol)

        if linked_market:
            watchlist[symbol] = {
                "symbol": symbol,
                "label": str(candidate["label"]),
                "market_key": linked_market["key"],
                "cot_score": linked_market["cot_bias"]["score"],
                "cot_label": linked_market["cot_bias"]["label"],
                "pulse_score": (linked_market.get("intraday") or {}).get("pulse_score"),
                "pulse_label": (linked_market.get("intraday") or {}).get("pulse_label"),
                "confluence_score": linked_market["confluence"]["score"],
                "confluence_label": linked_market["confluence"]["label"],
                "intraday": linked_market.get("intraday"),
            }
            continue

        try:
            intraday = _build_intraday_payload(symbol=symbol, label=str(candidate["label"]))
        except Exception:
            intraday = None

        watchlist[symbol] = {
            "symbol": symbol,
            "label": str(candidate["label"]),
            "market_key": None,
            "cot_score": None,
            "cot_label": "N/A",
            "pulse_score": intraday.get("pulse_score") if intraday else None,
            "pulse_label": intraday.get("pulse_label") if intraday else "N/A",
            "confluence_score": intraday.get("pulse_score") if intraday else None,
            "confluence_label": intraday.get("pulse_label") if intraday else "N/A",
            "intraday": intraday,
        }

    return watchlist


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

    now_utc = datetime.now(timezone.utc)

    markets = {
        key: _build_market_payload(normalized, market_key=key, history_weeks=history_weeks) for key in DEFAULT_INDEX_ORDER
    }

    try:
        red_folder_payload = _fetch_red_folder_events(limit=12)
    except Exception:
        red_folder_payload = {
            "source": FF_CALENDAR_URL,
            "fetched_at_utc": _fmt_utc(now_utc),
            "red_folder_count": 0,
            "upcoming_red_folder": [],
        }

    payload = {
        "generated_at_utc": _fmt_utc(now_utc),
        "source_url": url or DEFAULT_COT_URL,
        "history_weeks": history_weeks,
        "market_order": list(DEFAULT_INDEX_ORDER),
        "meta": {
            "timezone": "America/New_York",
            "next_cot_release_utc": _next_cot_release_utc(now_utc),
            "refresh_policy": "Hourly during US cash session, plus post-CFTC Friday refresh.",
        },
        "markets": markets,
        "watchlist_candidates": WATCHLIST_CANDIDATES,
        "watchlist": _build_watchlist(markets=markets),
        "forex_factory": red_folder_payload,
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
