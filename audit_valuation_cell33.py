# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_valuation_cell33.py
# ======================================================================================
#
# AUDITORIA DA CÉLULA 33B — VALUATION HISTÓRICO POINT-IN-TIME CORRIGIDO
#
# IMPORTANTE
# ----------
# A Célula 33 original NÃO é mais a referência desta auditoria.
#
# A referência científica é a CÉLULA 33B, porque ela:
#   • corrigiu a construção do Market Cap / valuation;
#   • foi aprovada;
#   • gerou ENTRY_VALUATION_PIT_CORRECTED.parquet;
#   • liberou explicitamente essa base para a Célula 34;
#   • é a base usada nas células posteriores do estudo.
#
# Esta auditoria NÃO exige mais:
#   • 1.704 snapshots elegíveis da Célula 33 antiga;
#   • coberturas antigas 1433/1399/1085/1622/1470;
#   • first_valid da Célula 33 antiga;
#   • snapshot atual antigo.
#
# REFERÊNCIA 33B:
#   • 15 ações
#   • 3 setores
#   • 5 / 5 / 5
#   • 2.100 snapshots
#   • 140 datas por ação
#   • 0 duplicatas ticker/data
#   • múltiplos válidos: 0 < múltiplo <= 500
#   • shares outstanding para Market Cap
#   • diluted shares apenas como fallback
#
# Cobertura corrigida dos múltiplos:
#   pe    = 1.433
#   pb    = 1.522
#   ps    = 1.637
#   p_ocf = 1.591
#   p_fcf = 1.548
#
# Histórico 33B:
#   ROBUSTO  = >= 60 meses com pelo menos um múltiplo válido
#   MODERADO = 24 a 59
#   CURTO    = < 24
#
# Resultado de referência:
#   ROBUSTO  = 13
#   MODERADO = 1
#   CURTO    = 1
#
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


# ======================================================================================
# 1. CONFIGURAÇÕES — CÉLULA 33B
# ======================================================================================

REFERENCE_DATE = pd.Timestamp("2026-08-24")
START_DATE = "2015-01-31"
PRICE_START = "2014-12-12"

MAX_MULTIPLE = 500.0

MIN_MARKET_CAP = 50_000_000
MAX_MARKET_CAP = 20_000_000_000_000

MIN_SHARE_RATIO = 0.50
MAX_SHARE_RATIO = 2.00

MAX_PE_LOG_GAP = np.log(2.0)

VALUATION_METRICS = [
    "pe",
    "pb",
    "ps",
    "p_ocf",
    "p_fcf",
]

EXPECTED_TOTAL_SNAPSHOTS = 2100
EXPECTED_SNAPSHOTS_PER_STOCK = 140

EXPECTED_COVERAGE = {
    "pe": 1433,
    "pb": 1522,
    "ps": 1637,
    "p_ocf": 1591,
    "p_fcf": 1548,
}

EXPECTED_HISTORY = {
    "EW":   (140, "ROBUSTO"),
    "DXCM": (140, "ROBUSTO"),
    "WST":  (140, "ROBUSTO"),
    "MRNA": (90,  "ROBUSTO"),
    "RMD":  (140, "ROBUSTO"),

    "VRT":  (78,  "ROBUSTO"),
    "FIX":  (140, "ROBUSTO"),
    "GE":   (140, "ROBUSTO"),
    "GEV":  (29,  "MODERADO"),
    "RTX":  (140, "ROBUSTO"),

    "VRSN": (140, "ROBUSTO"),
    "NVDA": (140, "ROBUSTO"),
    "SNDK": (18,  "CURTO"),
    "WDC":  (140, "ROBUSTO"),
    "FICO": (140, "ROBUSTO"),
}

EXPECTED_HISTORY_CLASS_COUNTS = {
    "ROBUSTO": 13,
    "MODERADO": 1,
    "CURTO": 1,
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


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_HISTORY = OUTPUT_DIR / "audit_valuation_cell33_history.csv"
OUTPUT_COVERAGE = OUTPUT_DIR / "audit_valuation_cell33_coverage.csv"
OUTPUT_HISTORY_CLASS = OUTPUT_DIR / "audit_valuation_cell33_first_valid.csv"
OUTPUT_CURRENT = OUTPUT_DIR / "audit_valuation_cell33_current.csv"


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def header(title: str):
    print("\n" + "=" * 145)
    print(title)
    print("=" * 145)


def normalize_ticker(value):
    return str(value).strip().upper().replace(".", "-")


def history_class(valid_months: int) -> str:
    if valid_months >= 60:
        return "ROBUSTO"
    if valid_months >= 24:
        return "MODERADO"
    return "CURTO"


# ======================================================================================
# 3. UNIVERSO FIXO
# ======================================================================================

def build_fixed_universe():

    universe = build_base_universe()
    universe = enrich_sectors(universe)
    universe = filter_target_sectors(universe)

    universe["ticker"] = universe["ticker"].map(normalize_ticker)

    expected = set(FIXED_PORTFOLIO["ticker"])

    fixed = universe[
        universe["ticker"].isin(expected)
    ].copy()

    missing = sorted(
        expected - set(fixed["ticker"])
    )

    if missing:
        raise RuntimeError(
            "Tickers da carteira 33B ausentes no universo: "
            + ", ".join(missing)
        )

    sector_map = (
        FIXED_PORTFOLIO
        .set_index("ticker")["sector"]
        .to_dict()
    )

    fixed["sector"] = fixed["ticker"].map(sector_map)

    return fixed


# ======================================================================================
# 4. CONSTRUIR BASE GITHUB E NORMALIZAR COMO 33B
# ======================================================================================

def build_github_33b_history(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    history = build_entry_signal_history(
        portfolio=FIXED_PORTFOLIO,
        prices=prices,
        fundamentals_history=fundamentals,
        as_of_date=REFERENCE_DATE,
        start_date=START_DATE,
    ).copy()

    if history.empty:
        raise RuntimeError(
            "build_entry_signal_history retornou base vazia."
        )

    history["snapshot_date"] = pd.to_datetime(
        history["snapshot_date"],
        errors="coerce",
    )

    history["ticker"] = history["ticker"].map(
        normalize_ticker
    )

    if "market_price" in history.columns:
        history["price"] = pd.to_numeric(
            history["market_price"],
            errors="coerce",
        )

    elif "price" in history.columns:
        history["price"] = pd.to_numeric(
            history["price"],
            errors="coerce",
        )

    else:
        raise RuntimeError(
            "Histórico não possui market_price nem price."
        )

    missing_metrics = [
        metric
        for metric in VALUATION_METRICS
        if metric not in history.columns
    ]

    if missing_metrics:
        raise RuntimeError(
            "Métricas de valuation ausentes: "
            + ", ".join(missing_metrics)
        )

    # Mesma regra econômica da Célula 33B.
    for metric in VALUATION_METRICS:

        values = pd.to_numeric(
            history[metric],
            errors="coerce",
        )

        history[metric] = values.where(
            (values > 0)
            &
            (values <= MAX_MULTIPLE)
        )

    return (
        history
        .sort_values(
            ["ticker", "snapshot_date"]
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 5. COBERTURA CORRIGIDA 33B
# ======================================================================================

def build_coverage(history: pd.DataFrame) -> pd.DataFrame:

    rows = []
    total = len(history)

    for metric in VALUATION_METRICS:

        series = pd.to_numeric(
            history[metric],
            errors="coerce",
        )

        available = int(
            series.notna().sum()
        )

        rows.append(
            {
                "metric": metric,
                "available": available,
                "expected_available": EXPECTED_COVERAGE[metric],
                "total": total,
                "coverage": (
                    available / total
                    if total
                    else np.nan
                ),
                "median": series.median(),
                "p95": series.quantile(0.95),
                "available_ok": (
                    available
                    ==
                    EXPECTED_COVERAGE[metric]
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================================
# 6. HISTÓRICO DISPONÍVEL — REGRA DA 33B
# ======================================================================================

def build_history_class(history: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for ticker in FIXED_PORTFOLIO["ticker"]:

        temp = history[
            history["ticker"] == ticker
        ].copy()

        valid_any = (
            temp[VALUATION_METRICS]
            .notna()
            .any(axis=1)
        )

        valid_months = int(
            valid_any.sum()
        )

        cls = history_class(
            valid_months
        )

        expected_months, expected_class = (
            EXPECTED_HISTORY[ticker]
        )

        sector = (
            FIXED_PORTFOLIO
            .loc[
                FIXED_PORTFOLIO["ticker"] == ticker,
                "sector",
            ]
            .iloc[0]
        )

        rows.append(
            {
                "ticker": ticker,
                "sector": sector,
                "valid_months": valid_months,
                "expected_valid_months": expected_months,
                "history_class": cls,
                "expected_history_class": expected_class,
                "months_ok": (
                    valid_months
                    ==
                    expected_months
                ),
                "class_ok": (
                    cls
                    ==
                    expected_class
                ),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================================
# 7. SNAPSHOT ATUAL — SANIDADE 33B
# ======================================================================================

def build_current_audit(history: pd.DataFrame) -> pd.DataFrame:

    latest_date = history["snapshot_date"].max()

    current = history[
        history["snapshot_date"] == latest_date
    ].copy()

    shares_col = None

    for candidate in [
        "shares_outstanding",
        "shares",
        "shares_for_market_cap",
    ]:
        if candidate in current.columns:
            shares_col = candidate
            break

    if shares_col is None:
        current["shares_for_audit"] = np.nan
    else:
        current["shares_for_audit"] = pd.to_numeric(
            current[shares_col],
            errors="coerce",
        )

    if "diluted_shares" in current.columns:
        current["diluted_shares_audit"] = pd.to_numeric(
            current["diluted_shares"],
            errors="coerce",
        )
    else:
        current["diluted_shares_audit"] = np.nan

    current["market_cap_audit"] = (
        pd.to_numeric(
            current["price"],
            errors="coerce",
        )
        *
        current["shares_for_audit"]
    )

    current["market_cap_valid"] = (
        current["market_cap_audit"]
        .between(
            MIN_MARKET_CAP,
            MAX_MARKET_CAP,
        )
    )

    current["shares_ratio"] = (
        current["shares_for_audit"]
        /
        current["diluted_shares_audit"]
    )

    current["shares_ratio_ok"] = (
        current["shares_ratio"]
        .between(
            MIN_SHARE_RATIO,
            MAX_SHARE_RATIO,
        )
        |
        current["shares_ratio"].isna()
    )

    current["pe_direct_audit"] = pd.to_numeric(
        current["pe"],
        errors="coerce",
    )

    if "net_income" in current.columns:

        net_income = pd.to_numeric(
            current["net_income"],
            errors="coerce",
        )

        current["pe_income_audit"] = np.where(
            (
                current["market_cap_audit"].notna()
            )
            &
            (
                net_income > 0
            ),
            current["market_cap_audit"]
            /
            net_income,
            np.nan,
        )

    else:
        current["pe_income_audit"] = np.nan

    direct = pd.to_numeric(
        current["pe_direct_audit"],
        errors="coerce",
    )

    implied = pd.to_numeric(
        current["pe_income_audit"],
        errors="coerce",
    )

    both_positive = (
        (direct > 0)
        &
        (implied > 0)
    )

    log_gap = pd.Series(
        np.nan,
        index=current.index,
        dtype=float,
    )

    log_gap.loc[both_positive] = np.abs(
        np.log(
            direct.loc[both_positive]
            /
            implied.loc[both_positive]
        )
    )

    current["pe_consistency_ok"] = (
        (~both_positive)
        |
        (log_gap <= MAX_PE_LOG_GAP)
    )

    current["audit_flags"] = "OK"

    for idx, row in current.iterrows():

        flags = []

        if not bool(row["market_cap_valid"]):
            flags.append("MARKET_CAP")

        if not bool(row["shares_ratio_ok"]):
            flags.append("SHARES")

        if not bool(row["pe_consistency_ok"]):
            flags.append("PE_INCONSISTENTE")

        current.at[idx, "audit_flags"] = (
            ", ".join(flags)
            if flags
            else "OK"
        )

    cols = [
        "sector",
        "ticker",
        "price",
        "shares_for_audit",
        "diluted_shares_audit",
        "shares_ratio",
        "market_cap_audit",
        "market_cap_valid",
        "pe_direct_audit",
        "pe_income_audit",
        "pe_consistency_ok",
        "pb",
        "ps",
        "p_ocf",
        "p_fcf",
        "audit_flags",
    ]

    return (
        current[cols]
        .sort_values(
            ["sector", "ticker"]
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 8. EXECUÇÃO
# ======================================================================================

def run_audit():

    header(
        "AUDITORIA VALUATION POINT-IN-TIME — CÉLULA 33B x GITHUB"
    )

    print(
        f"\nData de referência              : "
        f"{REFERENCE_DATE.date()}"
    )

    print(
        "Referência metodológica         : "
        "CÉLULA 33B — CORRIGIDA"
    )

    print(
        "Market Cap                      : "
        "preço × shares outstanding; diluted shares = fallback"
    )

    print(
        "Regra dos múltiplos             : "
        "0 < múltiplo <= 500"
    )

    # ------------------------------------------------------------------
    # CARTEIRA
    # ------------------------------------------------------------------

    header(
        "1. CARTEIRA FIXA DA CÉLULA 33B"
    )

    for sector in FIXED_PORTFOLIO["sector"].unique():

        tickers = FIXED_PORTFOLIO.loc[
            FIXED_PORTFOLIO["sector"] == sector,
            "ticker",
        ].tolist()

        print(
            f"{sector:<31}: "
            + ", ".join(tickers)
        )

    fixed_universe = build_fixed_universe()

    # ------------------------------------------------------------------
    # FUNDAMENTOS
    # ------------------------------------------------------------------

    header(
        "2. FUNDAMENTOS SEC"
    )

    fundamentals, errors = download_fundamentals(
        universe=fixed_universe,
        use_cache=True,
    )

    print(
        f"\nObservações fundamentais       : "
        f"{len(fundamentals):,}"
    )

    print(
        f"Empresas com erro              : "
        f"{len(errors):,}"
    )

    if fundamentals.empty:
        raise RuntimeError(
            "Nenhum fundamento SEC foi obtido."
        )

    # ------------------------------------------------------------------
    # PREÇOS
    # ------------------------------------------------------------------

    header(
        "3. PREÇOS"
    )

    prices = download_prices(
        tickers=FIXED_PORTFOLIO["ticker"].tolist(),
        start=PRICE_START,
    )

    print(
        f"\nPrimeira data                 : "
        f"{prices.index.min().date()}"
    )

    print(
        f"Última data                   : "
        f"{prices.index.max().date()}"
    )

    print(
        f"Tickers                       : "
        f"{len(prices.columns)}"
    )

    # ------------------------------------------------------------------
    # HISTÓRICO 33B
    # ------------------------------------------------------------------

    header(
        "4. VALUATION HISTÓRICO CORRIGIDO"
    )

    history = build_github_33b_history(
        fundamentals=fundamentals,
        prices=prices,
    )

    total = len(history)

    ticker_count = int(
        history["ticker"].nunique()
    )

    sector_count = int(
        history["sector"].nunique()
    )

    first_date = history["snapshot_date"].min()
    last_date = history["snapshot_date"].max()

    duplicate_count = int(
        history
        .duplicated(
            subset=[
                "ticker",
                "snapshot_date",
            ]
        )
        .sum()
    )

    snapshots_by_stock = (
        history
        .groupby("ticker")
        .size()
    )

    print(
        f"\nAções                         : "
        f"{ticker_count}"
    )

    print(
        f"Setores                       : "
        f"{sector_count}"
    )

    print(
        f"Snapshots totais              : "
        f"{total:,}"
    )

    print(
        f"Primeira data                 : "
        f"{first_date.date()}"
    )

    print(
        f"Última data                   : "
        f"{last_date.date()}"
    )

    print(
        f"Duplicatas ticker/data        : "
        f"{duplicate_count}"
    )

    # ------------------------------------------------------------------
    # COBERTURA
    # ------------------------------------------------------------------

    header(
        "5. COBERTURA CORRIGIDA DOS MÚLTIPLOS"
    )

    coverage = build_coverage(
        history
    )

    print(
        coverage
        .round(
            {
                "coverage": 6,
                "median": 4,
                "p95": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # HISTÓRICO DISPONÍVEL
    # ------------------------------------------------------------------

    header(
        "6. CLASSIFICAÇÃO DO HISTÓRICO — 33B"
    )

    history_audit = build_history_class(
        history
    )

    print(
        history_audit
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # SNAPSHOT ATUAL
    # ------------------------------------------------------------------

    header(
        "7. SNAPSHOT ATUAL — AUDITORIA 33B"
    )

    current = build_current_audit(
        history
    )

    print(
        current
        .round(4)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # DIAGNÓSTICO
    # ------------------------------------------------------------------

    header(
        "8. DIAGNÓSTICO FINAL — CÉLULA 33B"
    )

    snapshots_per_stock_ok = bool(
        (
            snapshots_by_stock
            ==
            EXPECTED_SNAPSHOTS_PER_STOCK
        )
        .all()
    )

    structure_ok = bool(
        ticker_count == 15
        and
        sector_count == 3
        and
        total == EXPECTED_TOTAL_SNAPSHOTS
        and
        snapshots_per_stock_ok
        and
        first_date == pd.Timestamp("2015-01-31")
        and
        last_date == REFERENCE_DATE
        and
        duplicate_count == 0
    )

    coverage_ok = bool(
        coverage["available_ok"].all()
    )

    history_months_ok = bool(
        history_audit["months_ok"].all()
    )

    history_classes_ok = bool(
        history_audit["class_ok"].all()
    )

    actual_class_counts = (
        history_audit[
            "history_class"
        ]
        .value_counts()
        .to_dict()
    )

    class_counts_ok = bool(
        actual_class_counts
        ==
        EXPECTED_HISTORY_CLASS_COUNTS
    )

    invalid_current_marketcap = int(
        (~current["market_cap_valid"])
        .sum()
    )

    invalid_current_pe = int(
        (~current["pe_consistency_ok"])
        .sum()
    )

    current_flags_ok = bool(
        (
            current["audit_flags"]
            ==
            "OK"
        )
        .all()
    )

    print(
        f"\nEstrutura 15/3/5-5-5             : "
        f"{structure_ok}"
    )

    print(
        f"2.100 snapshots                  : "
        f"{total == EXPECTED_TOTAL_SNAPSHOTS}"
    )

    print(
        f"140 snapshots por ação           : "
        f"{snapshots_per_stock_ok}"
    )

    print(
        f"Duplicatas ticker/data = 0       : "
        f"{duplicate_count == 0}"
    )

    print(
        f"Cobertura corrigida exata        : "
        f"{coverage_ok}"
    )

    print(
        f"Meses válidos por ação           : "
        f"{history_months_ok}"
    )

    print(
        f"Classes ROBUSTO/MODERADO/CURTO   : "
        f"{history_classes_ok}"
    )

    print(
        f"Contagem 13/1/1                  : "
        f"{class_counts_ok}"
    )

    print(
        f"Market Caps atuais reprovados    : "
        f"{invalid_current_marketcap}"
    )

    print(
        f"P/E atuais inconsistentes        : "
        f"{invalid_current_pe}"
    )

    print(
        f"Flags atuais todos OK            : "
        f"{current_flags_ok}"
    )

    approved = bool(
        structure_ok
        and
        coverage_ok
        and
        history_months_ok
        and
        history_classes_ok
        and
        class_counts_ok
        and
        invalid_current_marketcap == 0
        and
        invalid_current_pe == 0
        and
        current_flags_ok
    )

    if approved:

        status = (
            "AUDITORIA APROVADA — "
            "GITHUB REPRODUZ A CÉLULA 33B CORRIGIDA."
        )

    else:

        status = (
            "AUDITORIA NÃO APROVADA — "
            "EXISTEM DIVERGÊNCIAS CONTRA A CÉLULA 33B."
        )

    print(
        f"\nSTATUS: {status}"
    )

    # ------------------------------------------------------------------
    # ARQUIVOS
    # ------------------------------------------------------------------

    history.to_csv(
        OUTPUT_HISTORY,
        index=False,
    )

    coverage.to_csv(
        OUTPUT_COVERAGE,
        index=False,
    )

    history_audit.to_csv(
        OUTPUT_HISTORY_CLASS,
        index=False,
    )

    current.to_csv(
        OUTPUT_CURRENT,
        index=False,
    )

    header(
        "9. ARQUIVOS DE AUDITORIA"
    )

    print(
        f"\nHistórico      : {OUTPUT_HISTORY}"
    )

    print(
        f"Cobertura      : {OUTPUT_COVERAGE}"
    )

    print(
        f"Histórico 33B  : {OUTPUT_HISTORY_CLASS}"
    )

    print(
        f"Snapshot atual : {OUTPUT_CURRENT}"
    )

    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    run_audit()
