#!/usr/bin/env python3
"""
triangle_breakout_scanner_v2.py

Improved IAK-style ascending-triangle / volatility-compression breakout scanner.

What it looks for:
- A real resistance *level* built from clustered pivot highs, not a raw high percentile.
- Rising swing lows using timestamp-aware regression.
- Volatility compression using Bollinger Band width percentile.
- Fresh breakout with breakout-day volume expansion.
- Trend confirmation and extension controls.

Install:
    pip install yfinance pandas numpy openpyxl

Examples:
    python triangle_breakout_scanner_v2.py --tickers IAK,KIE,XLF,SMH,SOXX --period 5y
    python triangle_breakout_scanner_v2.py --universe-file ticker_universe_etfs_only.csv --period 5y --min-score 70
    python triangle_breakout_scanner_v2.py --universe-file ticker_universe_stocks_only.csv --period 5y --min-score 75 --min-avg-dollar-volume-m 25

This is a scanner/watchlist generator, not a trading recommendation system.
Manually review charts, volume, news, earnings dates, and options liquidity before trading.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


PRICE_FIELDS = {"Open", "High", "Low", "Close", "Volume"}


@dataclass
class ScanResult:
    ticker: str
    setup_state: str
    score: float

    last_close: float
    resistance: Optional[float]
    breakout_pct: Optional[float]
    breakout_days_ago: Optional[int]

    pivot_touch_count: int
    resistance_flatness_pct: Optional[float]
    low_slope_pct_per_day: Optional[float]
    swing_low_count: int
    ascending_triangle: bool

    bb_width_pctile: Optional[float]
    min_bb_width_pctile_20d: Optional[float]
    squeeze_recent: bool

    breakout_volume_ratio: Optional[float]
    volume_confirmed: bool
    last_complete_week_return_pct: Optional[float]

    close_vs_52w_high_pct: Optional[float]
    above_sma50: bool
    above_sma200: bool
    avg_dollar_volume_20d_m: float

    notes: str


def parse_tickers_from_file(path: str | Path) -> list[str]:
    p = Path(path)
    text = p.read_text(errors="ignore")

    if p.suffix.lower() in [".csv", ".tsv"]:
        sep = "\t" if p.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        possible_cols = [c for c in df.columns if c.lower() in {"ticker", "symbol", "etf", "fund"}]
        if possible_cols:
            vals = df[possible_cols[0]].astype(str).tolist()
        else:
            vals = df.iloc[:, 0].astype(str).tolist()
        tickers = vals
    else:
        tickers = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^\$?([A-Z][A-Z0-9.\-]{0,12})\b", line.upper())
            if m:
                tickers.append(m.group(1))

    cleaned: list[str] = []
    seen: set[str] = set()
    for t in tickers:
        t = str(t).strip().upper().replace("$", "")
        if not t or t in seen:
            continue
        cleaned.append(t)
        seen.add(t)
    return cleaned


def normalize_yfinance_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        level0 = set(str(x) for x in df.columns.get_level_values(0))
        level1 = set(str(x) for x in df.columns.get_level_values(1))

        if PRICE_FIELDS.issubset(level0):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        elif PRICE_FIELDS.issubset(level1):
            df = df.copy()
            df.columns = df.columns.get_level_values(1)
        else:
            raise ValueError(f"Unexpected yfinance MultiIndex columns: {df.columns.tolist()[:5]}")

    missing = PRICE_FIELDS - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


def download_one(
    ticker: str,
    period: str,
    interval: str = "1d",
    retry_attempts: int = 3,
    retry_backoff: float = 1.5,
) -> pd.DataFrame:
    last_exc: Optional[Exception] = None

    for attempt in range(1, retry_attempts + 1):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            df = normalize_yfinance_columns(df)
            df = df.dropna(subset=["High", "Low", "Close"])
            if not df.empty:
                return df
        except Exception as exc:
            last_exc = exc

        if attempt < retry_attempts:
            time.sleep(retry_backoff ** attempt)

    if last_exc:
        raise last_exc
    return pd.DataFrame()


def round_opt(value: Optional[float], digits: int = 2) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def strict_local_extrema(series: pd.Series, order: int = 5, kind: str = "low") -> pd.Series:
    """
    Strict pivot detector.

    A pivot low must be strictly lower than the previous N and next N bars.
    A pivot high must be strictly higher than the previous N and next N bars.

    This intentionally avoids counting every bar in a flat trough/top as a separate pivot.
    """
    if order < 1:
        raise ValueError("order must be >= 1")

    values = series.astype(float).to_numpy()
    idx = series.index
    pivots: list[tuple[pd.Timestamp, float]] = []

    for i in range(order, len(values) - order):
        v = values[i]
        if np.isnan(v):
            continue

        left = values[i - order : i]
        right = values[i + 1 : i + order + 1]

        if np.isnan(left).any() or np.isnan(right).any():
            continue

        if kind == "low":
            if v < left.min() and v < right.min():
                pivots.append((idx[i], float(v)))
        elif kind == "high":
            if v > left.max() and v > right.max():
                pivots.append((idx[i], float(v)))
        else:
            raise ValueError("kind must be 'low' or 'high'")

    if not pivots:
        return pd.Series(dtype=float)

    out_idx, out_vals = zip(*pivots)
    return pd.Series(out_vals, index=pd.DatetimeIndex(out_idx))


def cluster_pivot_highs(
    pivot_highs: pd.Series,
    zone_pct: float,
    min_touch_separation_days: int = 10,
    upper_quantile_floor: float = 0.55,
) -> dict[str, object]:
    """
    Cluster pivot highs into resistance zones.

    Resistance is chosen from the upper portion of pivot highs and must have repeated
    separated touches. This is closer to chart resistance than a raw percentile of all highs.
    """
    if pivot_highs.empty:
        return {
            "resistance": None,
            "touch_count": 0,
            "flatness_pct": None,
            "touch_dates": [],
            "method": "none",
        }

    pivot_highs = pivot_highs.sort_index()
    floor = float(pivot_highs.quantile(upper_quantile_floor))
    candidates = pivot_highs[pivot_highs >= floor].sort_values()

    if candidates.empty:
        candidates = pivot_highs.sort_values()

    raw_clusters: list[list[tuple[pd.Timestamp, float]]] = []
    current: list[tuple[pd.Timestamp, float]] = []

    for dt, value in candidates.items():
        value = float(value)
        if not current:
            current = [(dt, value)]
            continue

        current_level = float(np.median([v for _, v in current]))
        if abs(value / current_level - 1.0) <= zone_pct:
            current.append((dt, value))
        else:
            raw_clusters.append(current)
            current = [(dt, value)]

    if current:
        raw_clusters.append(current)

    clusters: list[dict[str, object]] = []
    for c in raw_clusters:
        dates = sorted([dt for dt, _ in c])
        values = [v for _, v in c]
        level = float(np.median(values))

        # De-duplicate touches that occur very close together in time.
        separated_dates: list[pd.Timestamp] = []
        separated_values: list[float] = []
        for dt, v in sorted(c, key=lambda x: x[0]):
            if not separated_dates or (dt - separated_dates[-1]).days >= min_touch_separation_days:
                separated_dates.append(dt)
                separated_values.append(v)

        if separated_values:
            flatness = float((max(separated_values) - min(separated_values)) / level * 100.0)
        else:
            flatness = None

        clusters.append(
            {
                "level": level,
                "touch_count": len(separated_values),
                "flatness_pct": flatness,
                "touch_dates": separated_dates,
                "values": separated_values,
            }
        )

    # Prefer higher levels with more touches and tighter clustering.
    clusters = sorted(
        clusters,
        key=lambda c: (
            int(c["touch_count"]),
            float(c["level"]),
            -999.0 if c["flatness_pct"] is None else -float(c["flatness_pct"]),
        ),
        reverse=True,
    )

    best = clusters[0]
    return {
        "resistance": float(best["level"]),
        "touch_count": int(best["touch_count"]),
        "flatness_pct": None if best["flatness_pct"] is None else float(best["flatness_pct"]),
        "touch_dates": best["touch_dates"],
        "method": "pivot_cluster",
    }


def slope_pct_per_day(series: pd.Series) -> Optional[float]:
    if len(series) < 2:
        return None

    s = series.dropna().sort_index()
    if len(s) < 2:
        return None

    y = s.astype(float).to_numpy()
    x = np.array([(dt - s.index[0]).days for dt in s.index], dtype=float)

    if np.nanmean(y) == 0 or len(np.unique(x)) < 2:
        return None

    slope = float(np.polyfit(x, y, 1)[0])
    return float((slope / np.nanmean(y)) * 100.0)


def bb_percentile_vector(reference: pd.Series, values: pd.Series) -> pd.Series:
    ref = reference.dropna().astype(float).to_numpy()
    vals = values.dropna().astype(float)

    if len(ref) == 0 or vals.empty:
        return pd.Series(dtype=float)

    pctiles = (ref[:, None] <= vals.to_numpy()[None, :]).mean(axis=0) * 100.0
    return pd.Series(pctiles, index=vals.index)


def last_complete_week_return(close: pd.Series) -> Optional[float]:
    if close.empty:
        return None

    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < 3:
        return None

    last_daily_date = close.index[-1].date()

    # If resample created a partial current week ending in the future, drop it.
    if weekly.index[-1].date() > last_daily_date:
        weekly = weekly.iloc[:-1]

    if len(weekly) < 2:
        return None

    return float(weekly.pct_change().iloc[-1] * 100.0)


def analyze_ticker(
    ticker: str,
    period: str = "5y",
    consolidation_days: int = 252,
    breakout_lookback_days: int = 10,
    resistance_zone_pct: float = 3.0,
    min_touch_count: int = 3,
    max_resistance_flatness_pct: float = 5.0,
    min_avg_dollar_volume_m: float = 2.0,
    breakout_buffer_pct: float = 0.5,
    volume_expansion_min: float = 1.3,
    swing_order: int = 5,
    min_low_slope_pct_per_day: float = 0.01,
    max_breakout_extension_pct: float = 8.0,
    retry_attempts: int = 3,
    retry_backoff: float = 1.5,
) -> Optional[ScanResult]:
    if consolidation_days < 80:
        raise ValueError("consolidation_days should be at least 80")
    if breakout_lookback_days < 1:
        raise ValueError("breakout_lookback_days must be >= 1")
    if resistance_zone_pct <= 0:
        raise ValueError("resistance_zone_pct must be positive")

    df = download_one(
        ticker,
        period=period,
        retry_attempts=retry_attempts,
        retry_backoff=retry_backoff,
    )

    min_rows = max(260, consolidation_days + breakout_lookback_days + 60)
    if len(df) < min_rows:
        return None

    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_WIDTH"] = (4.0 * df["STD20"]) / df["SMA20"]
    df["DOLLAR_VOLUME"] = df["Close"] * df["Volume"]

    latest = df.iloc[-1]
    last_close = float(latest["Close"])
    avg_dollar_volume_20d_m = float(df["DOLLAR_VOLUME"].tail(20).mean() / 1_000_000.0)

    if avg_dollar_volume_20d_m < min_avg_dollar_volume_m:
        return None

    base_start = -(consolidation_days + breakout_lookback_days)
    base_end = -breakout_lookback_days

    base = df.iloc[base_start:base_end]
    recent = df.tail(breakout_lookback_days)

    if len(base) < consolidation_days * 0.75:
        return None

    zone_decimal = resistance_zone_pct / 100.0

    pivot_highs = strict_local_extrema(base["High"], order=swing_order, kind="high")
    resistance_info = cluster_pivot_highs(pivot_highs, zone_pct=zone_decimal)
    resistance = resistance_info["resistance"]

    # Conservative fallback: if strict pivots are sparse, use high percentile but label it.
    if resistance is None:
        resistance = float(base["High"].quantile(0.97))
        resistance_info = {
            "resistance": resistance,
            "touch_count": 0,
            "flatness_pct": None,
            "touch_dates": [],
            "method": "percentile_fallback",
        }

    resistance = float(resistance)
    pivot_touch_count = int(resistance_info["touch_count"])
    resistance_flatness_pct = resistance_info["flatness_pct"]

    breakout_level = resistance * (1.0 + breakout_buffer_pct / 100.0)
    breakout_rows = recent[recent["Close"] > breakout_level]
    has_breakout = not breakout_rows.empty
    still_above_resistance = last_close > resistance
    breakout_pct = float((last_close / resistance - 1.0) * 100.0)

    breakout_days_ago: Optional[int] = None
    breakout_volume_ratio: Optional[float] = None
    volume_confirmed = False

    if has_breakout:
        breakout_idx = breakout_rows.index[0]
        breakout_loc = df.index.get_loc(breakout_idx)
        breakout_days_ago = int(len(df) - breakout_loc - 1)

        prior_volume = df["Volume"].iloc[max(0, breakout_loc - 50) : breakout_loc]
        if not prior_volume.empty and prior_volume.mean() > 0:
            breakout_volume_ratio = float(df.loc[breakout_idx, "Volume"] / prior_volume.mean())
            volume_confirmed = breakout_volume_ratio >= volume_expansion_min

    # Swing lows should be in the same base window and ideally after the first resistance-zone touch.
    swing_lows = strict_local_extrema(base["Low"], order=swing_order, kind="low")

    touch_dates = resistance_info.get("touch_dates", [])
    if touch_dates:
        first_touch_date = min(touch_dates)
        swing_lows_for_slope = swing_lows[swing_lows.index >= first_touch_date]
        if len(swing_lows_for_slope) < 3:
            swing_lows_for_slope = swing_lows.tail(12)
    else:
        swing_lows_for_slope = swing_lows.tail(12)

    swing_lows_for_slope = swing_lows_for_slope.tail(12)
    low_slope = slope_pct_per_day(swing_lows_for_slope)
    swing_low_count = int(len(swing_lows_for_slope))

    rising_lows = low_slope is not None and low_slope >= min_low_slope_pct_per_day
    base_has_flat_resistance = (
        pivot_touch_count >= min_touch_count
        and resistance_flatness_pct is not None
        and resistance_flatness_pct <= max_resistance_flatness_pct
    )
    ascending_triangle = bool(base_has_flat_resistance and rising_lows)

    # Volatility compression.
    bb_ref = df["BB_WIDTH"].tail(252)
    recent_bb_pctiles = bb_percentile_vector(bb_ref, df["BB_WIDTH"].tail(20))
    current_bb_pctile = None
    if not df["BB_WIDTH"].tail(1).dropna().empty:
        current_pct = bb_percentile_vector(bb_ref, df["BB_WIDTH"].tail(1))
        if not current_pct.empty:
            current_bb_pctile = float(current_pct.iloc[-1])

    min_bb_pctile_20d = float(recent_bb_pctiles.min()) if not recent_bb_pctiles.empty else None
    squeeze_recent = bool(min_bb_pctile_20d is not None and min_bb_pctile_20d <= 35.0)

    # Trend and high confirmation.
    above_sma50 = bool(last_close > float(latest["SMA50"])) if not pd.isna(latest["SMA50"]) else False
    above_sma200 = bool(last_close > float(latest["SMA200"])) if not pd.isna(latest["SMA200"]) else False

    high_52w = float(df["High"].tail(252).max())
    close_vs_52w_high_pct = float((last_close / high_52w - 1.0) * 100.0) if high_52w > 0 else None

    week_return = last_complete_week_return(df["Close"])

    # Setup state.
    if has_breakout and still_above_resistance and breakout_pct > max_breakout_extension_pct:
        setup_state = "extended_breakout"
    elif has_breakout and still_above_resistance:
        setup_state = "fresh_breakout"
    elif -1.5 <= breakout_pct <= breakout_buffer_pct:
        setup_state = "testing_resistance"
    elif breakout_pct < -3.0:
        setup_state = "below_resistance"
    else:
        setup_state = "near_resistance"

    # Scoring: more conjunctive than v1; fewer points for price position alone.
    score = 0.0
    notes: list[str] = []

    # Resistance/base quality: 35 pts.
    if pivot_touch_count >= min_touch_count:
        # 3 touches gets meaningful credit; 5+ gets full credit.
        score += min(18.0, 6.0 + (pivot_touch_count - min_touch_count + 1) * 4.0)
        notes.append(f"{pivot_touch_count} separated pivot resistance touches")
    elif pivot_touch_count > 0:
        score += 4.0
        notes.append(f"only {pivot_touch_count} pivot resistance touch(es)")

    if resistance_flatness_pct is not None and resistance_flatness_pct <= max_resistance_flatness_pct:
        score += 10.0
        notes.append("flat pivot resistance")
    elif resistance_flatness_pct is not None:
        score += max(0.0, 10.0 * (1.0 - resistance_flatness_pct / (max_resistance_flatness_pct * 2.0)))
        notes.append("messy resistance")

    if rising_lows:
        score += 18.0
        notes.append("rising swing lows")
    elif low_slope is not None and low_slope > 0:
        score += 8.0
        notes.append("slightly rising lows")
    elif low_slope is not None:
        score -= 12.0
        notes.append("falling/non-rising lows")

    if ascending_triangle:
        score += 5.0
        notes.append("valid ascending triangle structure")
    else:
        notes.append("ascending triangle not fully confirmed")

    # Volatility compression: 15 pts.
    if current_bb_pctile is not None and current_bb_pctile <= 25:
        score += 15.0
        notes.append("current volatility squeeze")
    elif current_bb_pctile is not None and current_bb_pctile <= 35:
        score += 10.0
        notes.append("current volatility compression")
    elif squeeze_recent:
        score += 7.0
        notes.append("recent volatility compression")

    # Breakout / setup timing: 25 pts.
    if setup_state == "fresh_breakout":
        score += 18.0
        notes.append("fresh breakout")
        if volume_confirmed:
            score += 10.0
            notes.append("breakout volume expansion")
        else:
            score -= 6.0
            notes.append("breakout lacks volume confirmation")
    elif setup_state == "testing_resistance":
        score += 12.0
        notes.append("testing resistance before breakout")
    elif setup_state == "extended_breakout":
        score += 10.0
        score -= 10.0
        notes.append("breakout already extended")
    elif setup_state == "near_resistance":
        score += 6.0
        notes.append("near resistance")

    # Trend / high quality: 20 pts.
    if above_sma200:
        score += 8.0
    else:
        score -= 15.0
        notes.append("below SMA200")

    if above_sma50:
        score += 5.0
    else:
        score -= 5.0
        notes.append("below SMA50")

    if close_vs_52w_high_pct is not None and close_vs_52w_high_pct >= -2.0:
        score += 7.0
        notes.append("near/new 52-week high")

    # Complete-week momentum: 5 pts.
    if week_return is not None:
        if 1.5 <= week_return <= 8.0:
            score += 5.0
            notes.append("healthy last complete week momentum")
        elif week_return > 8.0:
            score += 2.0
            notes.append("very strong but possibly extended weekly momentum")

    # Conjunctive caps/penalties to avoid weak charts scoring too high from one dimension.
    if not base_has_flat_resistance:
        score = min(score, 68.0)
    if not rising_lows:
        score = min(score, 65.0)
    if not ascending_triangle:
        score = min(score, 72.0)
    if not above_sma200:
        score = min(score, 55.0)
    if breakout_pct > max_breakout_extension_pct:
        score = min(score, 70.0)

    score = max(0.0, min(100.0, round(score, 1)))

    return ScanResult(
        ticker=ticker,
        setup_state=setup_state,
        score=score,
        last_close=round(last_close, 2),
        resistance=round_opt(resistance, 2),
        breakout_pct=round_opt(breakout_pct, 2),
        breakout_days_ago=breakout_days_ago,
        pivot_touch_count=pivot_touch_count,
        resistance_flatness_pct=round_opt(resistance_flatness_pct, 2),
        low_slope_pct_per_day=round_opt(low_slope, 4),
        swing_low_count=swing_low_count,
        ascending_triangle=ascending_triangle,
        bb_width_pctile=round_opt(current_bb_pctile, 1),
        min_bb_width_pctile_20d=round_opt(min_bb_pctile_20d, 1),
        squeeze_recent=squeeze_recent,
        breakout_volume_ratio=round_opt(breakout_volume_ratio, 2),
        volume_confirmed=volume_confirmed,
        last_complete_week_return_pct=round_opt(week_return, 2),
        close_vs_52w_high_pct=round_opt(close_vs_52w_high_pct, 2),
        above_sma50=above_sma50,
        above_sma200=above_sma200,
        avg_dollar_volume_20d_m=round(avg_dollar_volume_20d_m, 2),
        notes="; ".join(notes),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="Comma-separated tickers, e.g. IAK,XLF,KIE")
    parser.add_argument("--universe-file", help="CSV/TXT file containing tickers")
    parser.add_argument("--period", default="5y", help="yfinance period, e.g. 3y, 5y, 10y")
    parser.add_argument("--output", default="triangle_breakout_scan_v2.csv")
    parser.add_argument("--min-score", type=float, default=70.0)
    parser.add_argument("--min-avg-dollar-volume-m", type=float, default=2.0)

    parser.add_argument("--consolidation-days", type=int, default=252)
    parser.add_argument("--breakout-lookback-days", type=int, default=10)
    parser.add_argument("--resistance-zone-pct", type=float, default=3.0)
    parser.add_argument("--min-touch-count", type=int, default=3)
    parser.add_argument("--max-resistance-flatness-pct", type=float, default=5.0)
    parser.add_argument("--breakout-buffer-pct", type=float, default=0.5)
    parser.add_argument("--volume-expansion-min", type=float, default=1.3)
    parser.add_argument("--swing-order", type=int, default=5)
    parser.add_argument("--min-low-slope-pct-per-day", type=float, default=0.01)
    parser.add_argument("--max-breakout-extension-pct", type=float, default=8.0)

    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between ticker downloads")
    args = parser.parse_args()

    tickers: list[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper().replace("$", "") for t in args.tickers.split(",") if t.strip()])
    if args.universe_file:
        tickers.extend(parse_tickers_from_file(args.universe_file))

    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("Provide --tickers or --universe-file")

    rows: list[dict[str, object]] = []
    errors: list[tuple[str, str]] = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] scanning {ticker}...")
        try:
            result = analyze_ticker(
                ticker=ticker,
                period=args.period,
                consolidation_days=args.consolidation_days,
                breakout_lookback_days=args.breakout_lookback_days,
                resistance_zone_pct=args.resistance_zone_pct,
                min_touch_count=args.min_touch_count,
                max_resistance_flatness_pct=args.max_resistance_flatness_pct,
                min_avg_dollar_volume_m=args.min_avg_dollar_volume_m,
                breakout_buffer_pct=args.breakout_buffer_pct,
                volume_expansion_min=args.volume_expansion_min,
                swing_order=args.swing_order,
                min_low_slope_pct_per_day=args.min_low_slope_pct_per_day,
                max_breakout_extension_pct=args.max_breakout_extension_pct,
                retry_attempts=args.retry_attempts,
                retry_backoff=args.retry_backoff,
            )
            if result is not None:
                rows.append(asdict(result))
        except Exception as exc:
            errors.append((ticker, str(exc)))
        time.sleep(args.sleep)

    if not rows:
        print("No valid rows returned.")
        if errors:
            print("Errors:")
            for t, e in errors[:25]:
                print(f"  {t}: {e}")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["score", "setup_state", "breakout_pct", "last_complete_week_return_pct"],
        ascending=[False, True, False, False],
    )

    all_output = Path(args.output)
    filtered_output = all_output.with_name(all_output.stem + "_filtered" + all_output.suffix)

    df.to_csv(all_output, index=False)
    filtered = df[df["score"] >= args.min_score].copy()
    filtered.to_csv(filtered_output, index=False)

    display_cols = [
        "ticker",
        "setup_state",
        "score",
        "last_close",
        "resistance",
        "breakout_pct",
        "pivot_touch_count",
        "resistance_flatness_pct",
        "low_slope_pct_per_day",
        "bb_width_pctile",
        "breakout_volume_ratio",
        "volume_confirmed",
        "last_complete_week_return_pct",
        "notes",
    ]

    print("\nTop candidates:")
    if filtered.empty:
        print("No tickers met the minimum score. Showing top 30 unfiltered rows instead:")
        print(df[display_cols].head(30).to_string(index=False))
    else:
        print(filtered[display_cols].head(30).to_string(index=False))

    print(f"\nSaved full results to: {all_output}")
    print(f"Saved filtered results to: {filtered_output}")

    if errors:
        print(f"\nSkipped/error tickers: {len(errors)}")
        for t, e in errors[:15]:
            print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
