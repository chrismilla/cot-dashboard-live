import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlparse

import pandas as pd
import requests

DEFAULT_COT_JSON_URL = os.getenv("COT_FINANCIAL_JSON_URL", "https://publicreporting.cftc.gov/resource/udgc-27he.json")
DEFAULT_COT_TEXT_URL = os.getenv("COT_FINANCIAL_TEXT_URL", "https://www.cftc.gov/dea/newcot/FinFutWk.txt")
DEFAULT_COT_URL = os.getenv("COT_FINANCIAL_URL", DEFAULT_COT_JSON_URL)
CACHE_PATH = Path(os.getenv("COT_CACHE_PATH", "build/cot_financial.csv"))
SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_financial_futures.csv"

DEFAULT_INDEX_ORDER = ("dow_jones", "nasdaq_100", "sp_500")

# Focus the Socrata query to the three requested equity indexes.
INDEX_QUERY_WHERE = (
    "(futonly_or_combined = 'FutOnly' OR futonly_or_combined is null) AND ("
    "cftc_contract_market_code in "
    "('12460+','124603','124601','20974+','209742','13874+','13874A','138741') "
    "OR market_and_exchange_names like '%DJIA Consolidated%' "
    "OR market_and_exchange_names like '%NASDAQ-100 Consolidated%' "
    "OR market_and_exchange_names like '%S&P 500 Consolidated%')"
)


@dataclass(frozen=True)
class IndexMarket:
    key: str
    label: str
    preferred_codes: tuple[str, ...]
    name_patterns: tuple[str, ...]


INDEX_MARKETS: dict[str, IndexMarket] = {
    "dow_jones": IndexMarket(
        key="dow_jones",
        label="Dow Jones",
        preferred_codes=("12460+", "124603", "124601"),
        name_patterns=(r"DJIA Consolidated", r"DOW JONES INDUSTRIAL AVG", r"DOW JONES INDUSTRIAL AVERAGE"),
    ),
    "nasdaq_100": IndexMarket(
        key="nasdaq_100",
        label="Nasdaq-100",
        preferred_codes=("20974+", "209742"),
        name_patterns=(r"NASDAQ-100 Consolidated", r"NASDAQ-100 STOCK INDEX"),
    ),
    "sp_500": IndexMarket(
        key="sp_500",
        label="S&P 500",
        preferred_codes=("13874+", "13874A", "138741"),
        name_patterns=(r"S&P 500 Consolidated", r"E-MINI S&P 500", r"S&P 500 STOCK INDEX"),
    ),
}


@dataclass
class COTSummary:
    report_date: pd.Timestamp
    asset_manager_net: int
    leveraged_net: int
    open_interest: int
    asset_manager_change: Optional[int]
    leveraged_change: Optional[int]
    open_interest_change: Optional[int]

    @property
    def iso_date(self) -> str:
        return self.report_date.date().isoformat()


class COTDataError(RuntimeError):
    """Raised when no COT data can be loaded."""


def _is_json_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".json") or "publicreporting.cftc.gov/resource" in lowered


def _download_text(url: str) -> Optional[str]:
    headers = {"User-Agent": "cot-dashboard/1.0"}
    try:
        response = requests.get(url, timeout=30, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception:
        return None


def _download_json(url: str) -> Optional[list[dict]]:
    headers = {"User-Agent": "cot-dashboard/1.0", "Accept": "application/json"}
    params = None
    parsed = urlparse(url)
    if not parsed.query and "publicreporting.cftc.gov" in parsed.netloc:
        params = {
            "$limit": "50000",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$where": INDEX_QUERY_WHERE,
        }
    try:
        response = requests.get(url, timeout=30, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data
    except Exception:
        return None
    return None


def _has_any_column(df: pd.DataFrame, options: list[str]) -> bool:
    return any(col in df.columns for col in options)


def _looks_like_cot_frame(df: pd.DataFrame) -> bool:
    return (
        _has_any_column(df, ["market_and_exchange_names", "Market_and_Exchange_Names"])
        and _has_any_column(df, ["open_interest_all", "Open_Interest_All"])
        and _has_any_column(df, ["asset_mgr_positions_long", "Asset_Mgr_Long_All"])
        and _has_any_column(df, ["lev_money_positions_long", "Lev_Money_Long_All"])
    )


def load_financials(force_refresh: bool = False, url: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame of the financial futures disaggregated report.

    Attempts to read from cache, then the remote URL, and finally falls back to the
    bundled sample dataset.
    """
    if CACHE_PATH.exists() and not force_refresh:
        cached = pd.read_csv(CACHE_PATH)
        if _looks_like_cot_frame(cached):
            return cached

    url = url or DEFAULT_COT_URL
    df = None
    if _is_json_url(url):
        rows = _download_json(url)
        if rows:
            df = pd.DataFrame(rows)
    else:
        text = _download_text(url)
        if text:
            df = pd.read_csv(io.StringIO(text))

    if df is None and _is_json_url(url) and url == DEFAULT_COT_URL:
        text = _download_text(DEFAULT_COT_TEXT_URL)
        if text:
            df = pd.read_csv(io.StringIO(text))

    if df is not None and _looks_like_cot_frame(df):
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        return df

    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)

    raise COTDataError("Unable to load COT data from remote source or local sample.")


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "report_date_as_yyyy_mm_dd" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"], errors="coerce")
    elif "Report_Date_as_MM_DD_YYYY" in df.columns:
        df["report_date"] = pd.to_datetime(df["Report_Date_as_MM_DD_YYYY"], errors="coerce")
    elif "As_of_Date_In_Form_YYMMDD" in df.columns:
        df["report_date"] = pd.to_datetime(df["As_of_Date_In_Form_YYMMDD"], format="%y%m%d", errors="coerce")
    else:
        raise COTDataError("Unexpected schema: missing report date columns")

    market_col = None
    if "market_and_exchange_names" in df.columns:
        market_col = "market_and_exchange_names"
    elif "Market_and_Exchange_Names" in df.columns:
        market_col = "Market_and_Exchange_Names"
    if not market_col:
        raise COTDataError("Unexpected schema: missing market column")
    df["market_name"] = df[market_col].astype(str)

    if "cftc_contract_market_code" in df.columns:
        df["contract_code"] = df["cftc_contract_market_code"].astype(str).str.strip()
    elif "CFTC_Contract_Market_Code" in df.columns:
        df["contract_code"] = df["CFTC_Contract_Market_Code"].astype(str).str.strip()
    else:
        raise COTDataError("Unexpected schema: missing contract code column")

    if "futonly_or_combined" in df.columns:
        fut_only = df.loc[df["futonly_or_combined"].fillna("").isin(["FutOnly", ""])].copy()
        if not fut_only.empty:
            df = fut_only

    column_map = {
        "asset_mgr_long": ["Asset_Mgr_Long_All", "asset_mgr_positions_long"],
        "asset_mgr_short": ["Asset_Mgr_Short_All", "asset_mgr_positions_short"],
        "leveraged_long": ["Lev_Money_Long_All", "lev_money_positions_long"],
        "leveraged_short": ["Lev_Money_Short_All", "lev_money_positions_short"],
        "open_interest": ["Open_Interest_All", "open_interest_all"],
    }
    for target, candidates in column_map.items():
        source = next((col for col in candidates if col in df.columns), None)
        if not source:
            raise COTDataError(f"Unexpected schema: missing columns {candidates}")
        df[target] = pd.to_numeric(df[source], errors="coerce")

    df["asset_mgr_net"] = df["asset_mgr_long"] - df["asset_mgr_short"]
    df["leveraged_net"] = df["leveraged_long"] - df["leveraged_short"]

    df = df.dropna(subset=["report_date"])
    df = df.sort_values(["contract_code", "report_date", "open_interest"])
    df = df.drop_duplicates(subset=["contract_code", "report_date"], keep="last")
    df = df.sort_values("report_date")
    return df


def _find_market_slice(df: pd.DataFrame, spec: IndexMarket) -> tuple[pd.DataFrame, str]:
    for contract_code in spec.preferred_codes:
        matches = df.loc[df["contract_code"] == contract_code].copy()
        if not matches.empty:
            return matches, contract_code

    for pattern in spec.name_patterns:
        regex = re.compile(pattern, re.IGNORECASE)
        matches = df.loc[df["market_name"].astype(str).str.contains(regex, na=False)].copy()
        if matches.empty:
            continue

        # If fallback pattern matching catches multiple contracts, keep the dominant one.
        dominant_code = matches["contract_code"].value_counts().index[0]
        matches = matches.loc[matches["contract_code"] == dominant_code].copy()
        return matches, str(dominant_code)

    raise COTDataError(f"No rows found for {spec.label} in the dataset.")


def filter_market_with_code(df: pd.DataFrame, market_key: str) -> tuple[pd.DataFrame, str]:
    spec = INDEX_MARKETS.get(market_key)
    if spec is None:
        choices = ", ".join(DEFAULT_INDEX_ORDER)
        raise COTDataError(f"Unknown market '{market_key}'. Expected one of: {choices}")

    filtered, contract_code = _find_market_slice(df, spec)
    filtered = filtered.sort_values("report_date")
    if filtered.empty:
        raise COTDataError(f"No {spec.label} rows found in the dataset.")
    return filtered, contract_code


def filter_market(df: pd.DataFrame, market_key: str) -> pd.DataFrame:
    filtered, _ = filter_market_with_code(df, market_key=market_key)
    return filtered


def filter_djia(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias used by older scripts."""
    return filter_market(df, market_key="dow_jones")


def summarize_latest(df: pd.DataFrame) -> COTSummary:
    if df.empty:
        raise COTDataError("Cannot summarize an empty dataframe.")

    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else None
    return COTSummary(
        report_date=latest["report_date"],
        asset_manager_net=int(latest["asset_mgr_net"]),
        leveraged_net=int(latest["leveraged_net"]),
        open_interest=int(latest["open_interest"]),
        asset_manager_change=int(latest["asset_mgr_net"] - previous["asset_mgr_net"]) if previous is not None else None,
        leveraged_change=int(latest["leveraged_net"] - previous["leveraged_net"]) if previous is not None else None,
        open_interest_change=int(latest["open_interest"] - previous["open_interest"]) if previous is not None else None,
    )


def get_market_frame(market_key: str, force_refresh: bool = False, url: Optional[str] = None) -> pd.DataFrame:
    raw = load_financials(force_refresh=force_refresh, url=url)
    normalized = normalize_frame(raw)
    return filter_market(normalized, market_key=market_key)


def get_index_frames(
    indexes: Optional[Sequence[str]] = None,
    force_refresh: bool = False,
    url: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    raw = load_financials(force_refresh=force_refresh, url=url)
    normalized = normalize_frame(raw)
    keys = tuple(indexes) if indexes else DEFAULT_INDEX_ORDER
    return {key: filter_market(normalized, market_key=key) for key in keys}


def get_djia_frame(force_refresh: bool = False, url: Optional[str] = None) -> pd.DataFrame:
    return get_market_frame("dow_jones", force_refresh=force_refresh, url=url)
