#!/usr/bin/env python3
"""
triangle_breakout_scanner.py

Find ETF/stock setups similar to the IAK chart:
- long consolidation
- flat/horizontal resistance
- rising lows / ascending base
- volatility compression
- fresh breakout / new high
- strong recent momentum

Install:
    pip install yfinance pandas numpy scipy openpyxl

Examples:
    python triangle_breakout_scanner.py --tickers IAK,XLF,KIE,SPY,QQQ --period 5y
    python triangle_breakout_scanner.py --universe-file WealthfrontETFs.txt --period 5y --output scan_results.csv
    python triangle_breakout_scanner.py --universe-file tickers.csv --min-score 70

Notes:
- This is a screening tool, not a trading recommendation engine.
- Review charts manually before trading.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class ScanResult:
    ticker: str
    score: float
    last_close: float
    resistance: float
    breakout_pct: float
    breakout_days_ago: int
    touch_count: int
    resistance_flatness_pct: float
    low_slope_pct: float
    bb_width_pctile: float
    squeeze_recent: bool
    close_vs_52w_high_pct: float
    close_vs_period_high_pct: float
    last_week_return_pct: float
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
        # Accept lines like "IAK", "$IAK", "IAK - iShares Insurance ETF"
        tickers = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"^\$?([A-Z][A-Z0-9.\-]{0,9})\b", line.upper())
            if m:
                tickers.append(m.group(1))

    cleaned = []
    seen = set()
    for t in tickers:
        t = str(t).strip().upper().replace("$", "")
        if not t or t in seen:
            continue
        # Yahoo uses BRK-B style as BRK-B; some sources use BRK.B.
        cleaned.append(t)
        seen.add(t)
    return cleaned


def download_one(ticker: str, period: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        return pd.DataFrame()

    df = df.dropna(subset=["High", "Low", "Close"])
    return df


def pct_rank(series: pd.Series, value: float) -> float:
    clean = series.dropna().astype(float)
    if clean.empty or math.isnan(value):
        return np.nan
    return float((clean <= value).mean() * 100)


def linreg_slope_pct(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=float)
    if len(y) < 2 or np.any(np.isnan(y)) or np.nanmean(y) == 0:
        return np.nan
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float((slope / np.nanmean(y)) * 100)


def find_swing_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    lows = df["Low"]
    rolling_min = lows.rolling(window * 2 + 1, center=True).min()
    swings = lows.where(lows == rolling_min)
    return swings.dropna()


def analyze_ticker(
    ticker: str,
    period: str = "5y",
    consolidation_days: int = 252,
    exclude_recent_days: int = 10,
    breakout_lookback_days: int = 5,
    resistance_zone_pct: float = 0.035,
    min_avg_dollar_volume_m: float = 2.0,
) -> ScanResult | None:
    df = download_one(ticker, period=period)
    if len(df) < max(260, consolidation_days + 40):
        return None

    df = df.copy()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["STD20"] = df["Close"].rolling(20).std()
    df["BB_WIDTH"] = (4 * df["STD20"]) / df["SMA20"]
    df["DOLLAR_VOLUME"] = df["Close"] * df["Volume"]

    latest = df.iloc[-1]
    last_close = float(latest["Close"])
    avg_dollar_volume_20d_m = float(df["DOLLAR_VOLUME"].tail(20).mean() / 1_000_000)

    if avg_dollar_volume_20d_m < min_avg_dollar_volume_m:
        return None

    pre_breakout = df.iloc[-(consolidation_days + exclude_recent_days): -exclude_recent_days]
    if len(pre_breakout) < consolidation_days * 0.75:
        return None

    # Use the 97th percentile of prior highs as practical resistance.
    # This avoids one bad intraday spike dominating the level.
    resistance = float(pre_breakout["High"].quantile(0.97))

    if resistance <= 0:
        return None

    # Touch count near resistance.
    touch_zone = resistance * (1 - resistance_zone_pct)
    touches = pre_breakout[pre_breakout["High"] >= touch_zone]
    touch_count = int(len(touches))

    if len(touches) >= 3:
        resistance_flatness_pct = float((touches["High"].max() - touches["High"].min()) / resistance * 100)
    else:
        resistance_flatness_pct = 99.0

    # Rising lows: fit a slope across swing lows in the consolidation window.
    swing_lows = find_swing_lows(pre_breakout, window=5)
    # Keep last 8-15 swing lows to capture the current base instead of ancient structure.
    swing_lows_recent = swing_lows.tail(12)
    low_slope_pct = linreg_slope_pct(swing_lows_recent.values)

    # Volatility compression: Bollinger Band width percentile over the past 252 days.
    bb_ref = df["BB_WIDTH"].tail(252)
    current_bb_pctile = pct_rank(bb_ref, float(latest["BB_WIDTH"]))
    min_bb_pctile_last_20 = min(
        pct_rank(bb_ref, float(v))
        for v in df["BB_WIDTH"].tail(20).dropna().values
    ) if df["BB_WIDTH"].tail(20).dropna().size else np.nan
    squeeze_recent = bool(min_bb_pctile_last_20 <= 35) if not np.isnan(min_bb_pctile_last_20) else False

    # Breakout detection: any close in the past N days above resistance by 0.5%+.
    recent = df.tail(breakout_lookback_days)
    breakout_mask = recent["Close"] > resistance * 1.005
    if breakout_mask.any():
        breakout_idx = breakout_mask[breakout_mask].index[-1]
        breakout_days_ago = int((df.index[-1] - breakout_idx).days)
    else:
        breakout_days_ago = 999

    breakout_pct = float((last_close / resistance - 1) * 100)

    high_52w = float(df["High"].tail(252).max())
    period_high = float(df["High"].max())
    close_vs_52w_high_pct = float((last_close / high_52w - 1) * 100)
    close_vs_period_high_pct = float((last_close / period_high - 1) * 100)

    weekly_close = df["Close"].resample("W-FRI").last().dropna()
    last_week_return_pct = float(weekly_close.pct_change().iloc[-1] * 100) if len(weekly_close) > 2 else np.nan

    above_sma50 = bool(last_close > float(latest["SMA50"])) if not np.isnan(latest["SMA50"]) else False
    above_sma200 = bool(last_close > float(latest["SMA200"])) if not np.isnan(latest["SMA200"]) else False

    # Score components, 0-100.
    score = 0.0
    notes = []

    # Fresh breakout and position near highs.
    if breakout_pct > 0.5 and breakout_days_ago <= 10:
        score += 25
        notes.append("fresh breakout")
    elif breakout_pct > -1.0:
        score += 12
        notes.append("testing resistance")

    if close_vs_52w_high_pct >= -2.0:
        score += 12
        notes.append("near/new 52w high")
    if close_vs_period_high_pct >= -3.0:
        score += 8
        notes.append("near period high")

    # Ascending triangle structure.
    if touch_count >= 3:
        score += min(15, touch_count * 2)
        notes.append(f"{touch_count} resistance touches")
    if resistance_flatness_pct <= 6:
        score += 10
        notes.append("flat resistance")
    if not np.isnan(low_slope_pct) and low_slope_pct > 0.02:
        score += 15
        notes.append("rising lows")

    # Volatility compression.
    if not np.isnan(current_bb_pctile):
        if current_bb_pctile <= 35:
            score += 10
            notes.append("current volatility compression")
        elif squeeze_recent:
            score += 7
            notes.append("recent volatility compression")

    # Momentum confirmation.
    if above_sma50:
        score += 5
    if above_sma200:
        score += 5
    if not np.isnan(last_week_return_pct) and last_week_return_pct >= 3:
        score += 10
        notes.append("strong weekly momentum")

    score = min(100.0, round(score, 1))

    return ScanResult(
        ticker=ticker,
        score=score,
        last_close=round(last_close, 2),
        resistance=round(resistance, 2),
        breakout_pct=round(breakout_pct, 2),
        breakout_days_ago=breakout_days_ago,
        touch_count=touch_count,
        resistance_flatness_pct=round(resistance_flatness_pct, 2),
        low_slope_pct=round(low_slope_pct, 4) if not np.isnan(low_slope_pct) else np.nan,
        bb_width_pctile=round(current_bb_pctile, 1) if not np.isnan(current_bb_pctile) else np.nan,
        squeeze_recent=squeeze_recent,
        close_vs_52w_high_pct=round(close_vs_52w_high_pct, 2),
        close_vs_period_high_pct=round(close_vs_period_high_pct, 2),
        last_week_return_pct=round(last_week_return_pct, 2) if not np.isnan(last_week_return_pct) else np.nan,
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
    parser.add_argument("--output", default="triangle_breakout_scan.csv")
    parser.add_argument("--min-score", type=float, default=65)
    parser.add_argument("--min-avg-dollar-volume-m", type=float, default=2.0)
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between downloads")
    args = parser.parse_args()

    tickers: list[str] = []
    if args.tickers:
        tickers.extend([t.strip().upper().replace("$", "") for t in args.tickers.split(",") if t.strip()])
    if args.universe_file:
        tickers.extend(parse_tickers_from_file(args.universe_file))

    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("Provide --tickers or --universe-file")

    rows = []
    errors = []

    for i, ticker in enumerate(tickers, start=1):
        print(f"[{i}/{len(tickers)}] scanning {ticker}...")
        try:
            result = analyze_ticker(
                ticker,
                period=args.period,
                min_avg_dollar_volume_m=args.min_avg_dollar_volume_m,
            )
            if result:
                rows.append(asdict(result))
        except Exception as exc:
            errors.append((ticker, str(exc)))
        time.sleep(args.sleep)

    if not rows:
        print("No valid rows returned.")
        if errors:
            print("Errors:")
            for t, e in errors[:20]:
                print(f"  {t}: {e}")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["score", "breakout_pct", "last_week_return_pct"], ascending=[False, False, False])

    all_output = Path(args.output)
    filtered_output = all_output.with_name(all_output.stem + "_filtered" + all_output.suffix)

    df.to_csv(all_output, index=False)
    filtered = df[df["score"] >= args.min_score].copy()
    filtered.to_csv(filtered_output, index=False)

    print("\nTop candidates:")
    cols = [
        "ticker",
        "score",
        "last_close",
        "resistance",
        "breakout_pct",
        "touch_count",
        "bb_width_pctile",
        "last_week_return_pct",
        "notes",
    ]
    print(filtered[cols].head(30).to_string(index=False))

    print(f"\nSaved full results to: {all_output}")
    print(f"Saved filtered results to: {filtered_output}")

    if errors:
        print(f"\nSkipped/error tickers: {len(errors)}")
        for t, e in errors[:10]:
            print(f"  {t}: {e}")


if __name__ == "__main__":
    main()
