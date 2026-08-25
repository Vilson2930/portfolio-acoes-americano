# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_valuation_cell33.py
# ======================================================================================
#
# Audita a CÉLULA 33 — VALUATION HISTÓRICO POINT-IN-TIME
# contra a implementação atual do GitHub.
#
# Valida:
#   • carteira fixa de 15 ações
#   • 140 snapshots por ação
#   • 2.100 snapshots totais
#   • 1.704 snapshots elegíveis
#   • 2015-01-31 -> 2026-08-24
#   • cobertura de P/E, P/B, P/S, P/OCF e P/FCF
#   • primeira data elegível por ticker
#   • 13/15 empresas com >=24 meses válidos
#   • snapshot atual
#
# Regra da Célula 33:
#       0 < múltiplo <= 500
#       mínimo 2 métricas válidas por snapshot
# ======================================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from data import (
    build_base_universe,
    enrich_sectors,
    filter_target_sectors,
    download_fundamentals,
    download_prices,
)

from entry import build_entry_signal_history


REFERENCE_DATE = pd.Timestamp("2026-08-24")
START_DATE = "2015-01-31"
PRICE_START = "2014-12-12"

MIN_VALID_METRICS = 2
MIN_HISTORY_MONTHS = 24
MAX_MULTIPLE = 500.0

VALUATION_METRICS = ["pe", "pb", "ps", "p_ocf", "p_fcf"]

EXPECTED_TOTAL_SNAPSHOTS = 2100
EXPECTED_SNAPSHOTS_PER_STOCK = 140
EXPECTED_ELIGIBLE_SNAPSHOTS = 1704
EXPECTED_HISTORY_OK = 13

EXPECTED_COVERAGE = {
    "pe": 1433,
    "pb": 1399,
    "ps": 1085,
    "p_ocf": 1622,
    "p_fcf": 1470,
}

EXPECTED_FIRST_VALID = {
    "EW":   ("2015-01-31", 140, True),
    "DXCM": ("2015-02-28", 127, True),
    "WST":  ("2015-01-31", 140, True),
    "MRNA": ("2020-02-29", 76, True),
    "RMD":  ("2015-01-31", 140, True),
    "VRT":  ("2020-05-31", 69, True),
    "FIX":  ("2015-01-31", 140, True),
    "GE":   ("2015-01-31", 140, True),
    "GEV":  ("2025-02-28", 19, False),
    "RTX":  ("2015-01-31", 140, True),
    "VRSN": ("2015-01-31", 140, True),
    "NVDA": ("2015-01-31", 140, True),
    "SNDK": ("2025-08-31", 13, False),
    "WDC":  ("2015-01-31", 140, True),
    "FICO": ("2015-01-31", 140, True),
}

FIXED_PORTFOLIO = pd.DataFrame(
    [
        ("Health Care", "EW"),
        ("Health Care", "DXCM"),
        ("Health Care", "WST"),
        ("Health Care", "MRNA"),
        ("Health Care", "RMD"),
        ("Industrials", "VRT"),
        ("Industrials", "FIX"),
        ("Industrials", "GE"),
        ("Industrials", "GEV"),
        ("Industrials", "RTX"),
        ("Information Technology", "VRSN"),
        ("Information Technology", "NVDA"),
        ("Information Technology", "SNDK"),
        ("Information Technology", "WDC"),
        ("Information Technology", "FICO"),
    ],
    columns=["sector", "ticker"],
)

EXPECTED_CURRENT = {
    "EW":   {"price": 91.08,   "pe": 49.77, "pb": 4.95,  "ps": 8.66,   "p_ocf": 32.92, "p_fcf": 39.34, "valid": 5},
    "DXCM": {"price": 91.56,   "pe": 43.81, "pb": 13.18, "ps": 7.41,   "p_ocf": 23.98, "p_fcf": 32.08, "valid": 5},
    "WST":  {"price": 350.57,  "pe": 51.63, "pb": 8.25,  "ps": 8.03,   "p_ocf": 32.70, "p_fcf": 52.63, "valid": 5},
    "MRNA": {"price": 139.36,  "pe": np.nan, "pb": 8.22,  "ps": 28.60, "p_ocf": np.nan, "p_fcf": np.nan, "valid": 2},
    "RMD":  {"price": 232.31,  "pe": 22.27, "pb": 5.09,  "ps": 5.93,   "p_ocf": 18.56, "p_fcf": 20.31, "valid": 5},
    "VRT":  {"price": 254.21,  "pe": 74.55, "pb": 20.57, "ps": 9.57,  "p_ocf": 46.29, "p_fcf": 51.67, "valid": 5},
    "FIX":  {"price": 1609.66, "pe": 55.74, "pb": 17.61, "ps": 19.83, "p_ocf": 47.76, "p_fcf": 54.93, "valid": 5},
    "GE":   {"price": 340.71,  "pe": 41.86, "pb": 20.04, "ps": 3.11,  "p_ocf": 41.41, "p_fcf": 49.33, "valid": 5},
    "GEV":  {"price": 940.47,  "pe": 53.16, "pb": 20.95, "ps": 6.58,  "p_ocf": 50.23, "p_fcf": np.nan, "valid": 4},
    "RTX":  {"price": 209.11,  "pe": 42.16, "pb": 0.00,  "ps": 0.00,   "p_ocf": 0.03,  "p_fcf": 0.04, "valid": 5},
    "VRSN": {"price": 290.92,  "pe": 46.62, "pb": np.nan,"ps": 15.88, "p_ocf": 24.10, "p_fcf": 24.62, "valid": 4},
    "NVDA": {"price": 209.46,  "pe": 42.75, "pb": 26.04, "ps": 189.14,"p_ocf": 49.56, "p_fcf": 49.63, "valid": 5},
    "SNDK": {"price": 1499.94, "pe": 20.34, "pb": 13.92, "ps": 10.82, "p_ocf": 18.76, "p_fcf": 19.05, "valid": 5},
    "WDC":  {"price": 436.83,  "pe": 17.99, "pb": 17.79, "ps": 12.21, "p_ocf": 40.14, "p_fcf": 44.91, "valid": 5},
    "FICO": {"price": 1172.86, "pe": 44.19, "pb": np.nan,"ps": 12.72, "p_ocf": 32.52, "p_fcf": 32.90, "valid": 4},
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def header(title: str):
    print("\n" + "=" * 145)
    print(title)
    print("=" * 145)


def normalize_ticker(value):
    return str(value).strip().upper().replace(".", "-")


def close_enough(actual, expected, tolerance=0.10):
    if pd.isna(actual) and pd.isna(expected):
        return True
    if pd.isna(actual) or pd.isna(expected):
        return False
    return abs(float(actual) - float(expected)) <= tolerance


def build_fixed_universe():
    universe = build_base_universe()
    universe = enrich_sectors(universe)
    universe = filter_target_sectors(universe)
    universe["ticker"] = universe["ticker"].map(normalize_ticker)

    expected = set(FIXED_PORTFOLIO["ticker"])
    fixed = universe[universe["ticker"].isin(expected)].copy()

    missing = sorted(expected - set(fixed["ticker"]))
    if missing:
        raise RuntimeError(
            f"Tickers da Célula 33 não encontrados no universo atual: {missing}"
        )

    sector_map = FIXED_PORTFOLIO.set_index("ticker")["sector"].to_dict()
    fixed["sector"] = fixed["ticker"].map(sector_map)
    return fixed


def build_github_cell33_history(fundamentals, prices):
    history = build_entry_signal_history(
        portfolio=FIXED_PORTFOLIO,
        prices=prices,
        fundamentals_history=fundamentals,
        as_of_date=REFERENCE_DATE,
        start_date=START_DATE,
    ).copy()

    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
    history["ticker"] = history["ticker"].map(normalize_ticker)
    history["price"] = history["market_price"]

    for metric in VALUATION_METRICS:
        values = pd.to_numeric(history[metric], errors="coerce")
        history[f"{metric}_raw"] = values
        history[metric] = values.where((values > 0) & (values <= MAX_MULTIPLE))

    history["valid_valuation_metrics"] = (
        history[VALUATION_METRICS].notna().sum(axis=1)
    )

    history["valuation_eligible"] = (
        history["valid_valuation_metrics"] >= MIN_VALID_METRICS
    )

    return history.sort_values(
        ["ticker", "snapshot_date"]
    ).reset_index(drop=True)


def build_coverage(history):
    rows = []
    total = len(history)

    for metric in VALUATION_METRICS:
        series = pd.to_numeric(history[metric], errors="coerce")
        available = int(series.notna().sum())

        rows.append(
            {
                "metric": metric,
                "available": available,
                "expected_available": EXPECTED_COVERAGE[metric],
                "total": total,
                "coverage": available / total if total else np.nan,
                "median": series.median(),
                "p95": series.quantile(0.95),
                "available_ok": available == EXPECTED_COVERAGE[metric],
            }
        )

    return pd.DataFrame(rows)


def build_first_valid(history):
    rows = []

    for ticker in FIXED_PORTFOLIO["ticker"]:
        temp = history[
            (history["ticker"] == ticker)
            & history["valuation_eligible"]
        ]

        rows.append(
            {
                "ticker": ticker,
                "first_valid_date": (
                    temp["snapshot_date"].min() if len(temp) else pd.NaT
                ),
                "valid_months": len(temp),
                "history_ok": len(temp) >= MIN_HISTORY_MONTHS,
            }
        )

    out = pd.DataFrame(rows)

    out["expected_first_valid_date"] = out["ticker"].map(
        lambda x: pd.Timestamp(EXPECTED_FIRST_VALID[x][0])
    )
    out["expected_valid_months"] = out["ticker"].map(
        lambda x: EXPECTED_FIRST_VALID[x][1]
    )
    out["expected_history_ok"] = out["ticker"].map(
        lambda x: EXPECTED_FIRST_VALID[x][2]
    )

    out["date_ok"] = (
        out["first_valid_date"] == out["expected_first_valid_date"]
    )
    out["months_ok"] = (
        out["valid_months"] == out["expected_valid_months"]
    )
    out["history_ok_match"] = (
        out["history_ok"] == out["expected_history_ok"]
    )

    return out


def build_current_snapshot(history):
    latest = history["snapshot_date"].max()

    current = history[
        history["snapshot_date"] == latest
    ][
        [
            "sector",
            "ticker",
            "price",
            *VALUATION_METRICS,
            "valid_valuation_metrics",
        ]
    ].copy()

    current["current_ok"] = True

    for idx, row in current.iterrows():
        expected = EXPECTED_CURRENT[row["ticker"]]

        checks = [
            close_enough(row["price"], expected["price"]),
            close_enough(row["pe"], expected["pe"]),
            close_enough(row["pb"], expected["pb"]),
            close_enough(row["ps"], expected["ps"]),
            close_enough(row["p_ocf"], expected["p_ocf"]),
            close_enough(row["p_fcf"], expected["p_fcf"]),
            int(row["valid_valuation_metrics"]) == int(expected["valid"]),
        ]

        current.at[idx, "current_ok"] = all(checks)

    return current.sort_values(
        ["sector", "ticker"]
    ).reset_index(drop=True)


def run_audit():
    header("AUDITORIA VALUATION POINT-IN-TIME — CÉLULA 33 x GITHUB")

    print(f"\nData de referência              : {REFERENCE_DATE.date()}")
    print("Regra múltiplos                : 0 < múltiplo <= 500")
    print(f"Mínimo métricas válidas        : {MIN_VALID_METRICS}")

    header("1. CARTEIRA FIXA DA CÉLULA 33")

    for sector in FIXED_PORTFOLIO["sector"].unique():
        tickers = FIXED_PORTFOLIO.loc[
            FIXED_PORTFOLIO["sector"] == sector,
            "ticker",
        ].tolist()
        print(f"{sector:<31}: {', '.join(tickers)}")

    fixed_universe = build_fixed_universe()

    header("2. FUNDAMENTOS SEC")

    fundamentals, errors = download_fundamentals(
        universe=fixed_universe,
        use_cache=True,
    )

    print(f"\nObservações fundamentais       : {len(fundamentals):,}")
    print(f"Empresas com erro              : {len(errors):,}")

    if fundamentals.empty:
        raise RuntimeError("Nenhum fundamento SEC foi obtido.")

    header("3. PREÇOS")

    prices = download_prices(
        tickers=FIXED_PORTFOLIO["ticker"].tolist(),
        start=PRICE_START,
    )

    print(f"\nPrimeira data                 : {prices.index.min().date()}")
    print(f"Última data                   : {prices.index.max().date()}")
    print(f"Tickers                       : {len(prices.columns)}")

    header("4. VALUATION HISTÓRICO")

    history = build_github_cell33_history(
        fundamentals=fundamentals,
        prices=prices,
    )

    total = len(history)
    ticker_count = history["ticker"].nunique()
    sector_count = history["sector"].nunique()
    first_date = history["snapshot_date"].min()
    last_date = history["snapshot_date"].max()
    duplicate_count = int(
        history.duplicated(
            subset=["ticker", "snapshot_date"]
        ).sum()
    )
    snapshots_by_stock = history.groupby("ticker").size()
    eligible = int(history["valuation_eligible"].sum())

    print(f"\nAções                         : {ticker_count}")
    print(f"Setores                       : {sector_count}")
    print(f"Snapshots totais              : {total:,}")
    print(f"Snapshots elegíveis           : {eligible:,}")
    print(f"Primeira data                 : {first_date.date()}")
    print(f"Última data                   : {last_date.date()}")
    print(f"Duplicatas ticker/data        : {duplicate_count}")

    header("5. COBERTURA DOS MÚLTIPLOS")

    coverage = build_coverage(history)

    print(
        coverage.round(
            {"coverage": 6, "median": 4, "p95": 4}
        ).to_string(index=False)
    )

    header("6. HISTÓRICO ELEGÍVEL")

    first_valid = build_first_valid(history)

    print(
        first_valid[
            [
                "ticker",
                "first_valid_date",
                "expected_first_valid_date",
                "valid_months",
                "expected_valid_months",
                "history_ok",
                "date_ok",
                "months_ok",
                "history_ok_match",
            ]
        ].to_string(index=False)
    )

    header("7. SNAPSHOT ATUAL")

    current = build_current_snapshot(history)

    print(
        current.round(4).to_string(index=False)
    )

    header("8. DIAGNÓSTICO FINAL")

    snapshots_per_stock_ok = bool(
        (snapshots_by_stock == EXPECTED_SNAPSHOTS_PER_STOCK).all()
    )

    structure_ok = bool(
        ticker_count == 15
        and sector_count == 3
        and total == EXPECTED_TOTAL_SNAPSHOTS
        and snapshots_per_stock_ok
        and first_date == pd.Timestamp("2015-01-31")
        and last_date == REFERENCE_DATE
        and duplicate_count == 0
    )

    eligible_ok = eligible == EXPECTED_ELIGIBLE_SNAPSHOTS
    coverage_ok = bool(coverage["available_ok"].all())

    first_valid_ok = bool(
        first_valid[
            ["date_ok", "months_ok", "history_ok_match"]
        ].all(axis=None)
    )

    history_ok_count = int(first_valid["history_ok"].sum())
    history_ok_count_match = history_ok_count == EXPECTED_HISTORY_OK
    current_ok = bool(current["current_ok"].all())

    print(f"\nEstrutura 15/3/5-5-5            : {structure_ok}")
    print(f"2.100 snapshots                 : {total == EXPECTED_TOTAL_SNAPSHOTS}")
    print(f"140 snapshots por ação          : {snapshots_per_stock_ok}")
    print(f"1.704 snapshots elegíveis       : {eligible_ok}")
    print(f"Cobertura exata dos múltiplos   : {coverage_ok}")
    print(f"Primeira data válida por ação   : {first_valid_ok}")
    print(f"Empresas >=24 meses = 13/15     : {history_ok_count_match}")
    print(f"Snapshot atual dentro tolerância : {current_ok}")

    approved = bool(
        structure_ok
        and eligible_ok
        and coverage_ok
        and first_valid_ok
        and history_ok_count_match
        and current_ok
    )

    if approved:
        status = (
            "AUDITORIA APROVADA — "
            "GITHUB REPRODUZ A CÉLULA 33."
        )
    else:
        status = (
            "AUDITORIA NÃO APROVADA — "
            "EXISTEM DIVERGÊNCIAS NA CÉLULA 33."
        )

    print(f"\nSTATUS: {status}")

    history.to_csv(
        OUTPUT_DIR / "audit_valuation_cell33_history.csv",
        index=False,
    )

    coverage.to_csv(
        OUTPUT_DIR / "audit_valuation_cell33_coverage.csv",
        index=False,
    )

    first_valid.to_csv(
        OUTPUT_DIR / "audit_valuation_cell33_first_valid.csv",
        index=False,
    )

    current.to_csv(
        OUTPUT_DIR / "audit_valuation_cell33_current.csv",
        index=False,
    )

    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    run_audit()
