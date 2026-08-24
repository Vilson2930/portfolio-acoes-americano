# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_industrials.py
# ======================================================================================
#
# Audita VRT, FIX, GE e RTX componente por componente contra a Célula 41.
# Não altera entry.py, data.py, pesos, thresholds ou execução diária.
# ======================================================================================

from __future__ import annotations

from pathlib import Path
import pandas as pd

from data import (
    build_base_universe,
    enrich_sectors,
    download_fundamentals,
    download_prices,
)
from entry import build_entry_signal_history


REFERENCE_DATE = pd.Timestamp("2026-08-24")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CURRENT = OUTPUT_DIR / "audit_industrials_current.csv"
OUTPUT_HISTORY = OUTPUT_DIR / "audit_industrials_history.csv"


REFERENCE_PORTFOLIO = pd.DataFrame(
    [
        {"ticker": "EW", "sector": "Health Care"},
        {"ticker": "DXCM", "sector": "Health Care"},
        {"ticker": "WST", "sector": "Health Care"},
        {"ticker": "MRNA", "sector": "Health Care"},
        {"ticker": "RMD", "sector": "Health Care"},
        {"ticker": "VRT", "sector": "Industrials"},
        {"ticker": "FIX", "sector": "Industrials"},
        {"ticker": "GE", "sector": "Industrials"},
        {"ticker": "GEV", "sector": "Industrials"},
        {"ticker": "RTX", "sector": "Industrials"},
        {"ticker": "VRSN", "sector": "Information Technology"},
        {"ticker": "NVDA", "sector": "Information Technology"},
        {"ticker": "SNDK", "sector": "Information Technology"},
        {"ticker": "WDC", "sector": "Information Technology"},
        {"ticker": "FICO", "sector": "Information Technology"},
    ]
)

REFERENCE = pd.DataFrame(
    [
        {
            "ticker": "VRT",
            "expected_discount_score": 0.781,
            "expected_fundamental_score": 0.943,
            "expected_final_score": 0.911,
            "expected_percentile": 0.951220,
        },
        {
            "ticker": "FIX",
            "expected_discount_score": 0.853,
            "expected_fundamental_score": 0.891,
            "expected_final_score": 0.884,
            "expected_percentile": 0.921951,
        },
        {
            "ticker": "GE",
            "expected_discount_score": 0.489,
            "expected_fundamental_score": 0.947,
            "expected_final_score": 0.855,
            "expected_percentile": 0.907317,
        },
        {
            "ticker": "RTX",
            "expected_discount_score": 0.396,
            "expected_fundamental_score": 0.712,
            "expected_final_score": 0.649,
            "expected_percentile": 0.680488,
        },
    ]
)


def header(title):
    print("\n" + "=" * 140)
    print(title)
    print("=" * 140)


def build_test_universe():
    universe = enrich_sectors(build_base_universe())

    wanted = set(REFERENCE_PORTFOLIO["ticker"])
    test_universe = universe[universe["ticker"].isin(wanted)].copy()

    missing = sorted(wanted - set(test_universe["ticker"]))
    if missing:
        raise RuntimeError(
            "Tickers ausentes no universo atual: " + ", ".join(missing)
        )

    return (
        test_universe
        .drop(columns=["sector"], errors="ignore")
        .merge(REFERENCE_PORTFOLIO, on="ticker", how="left")
    )


def run_audit():
    header("AUDITORIA DETALHADA — INDUSTRIALS / CÉLULA 41")
    print(f"\nData de referência : {REFERENCE_DATE.date()}")

    header("1. FUNDAMENTOS SEC")
    universe = build_test_universe()

    fundamentals, errors = download_fundamentals(
        universe=universe,
        use_cache=True,
    )

    print(f"\nObservações fundamentais : {len(fundamentals):,}")

    if isinstance(errors, pd.DataFrame) and not errors.empty:
        print("\nErros SEC:")
        print(errors.to_string(index=False))

    header("2. PREÇOS")
    prices = download_prices(
        tickers=REFERENCE_PORTFOLIO["ticker"].tolist(),
        start="2013-01-01",
        end=(REFERENCE_DATE + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
    )

    print(f"\nPrimeira data : {prices.index.min().date()}")
    print(f"Última data   : {prices.index.max().date()}")
    print(f"Tickers       : {len(prices.columns)}")

    header("3. RECONSTRUIR ENTRY SIGNAL HISTORY")
    history = build_entry_signal_history(
        portfolio=REFERENCE_PORTFOLIO,
        prices=prices,
        fundamentals_history=fundamentals,
        as_of_date=REFERENCE_DATE,
    )

    latest_date = history["snapshot_date"].max()
    print(f"\nÚltimo snapshot : {latest_date.date()}")

    current = history[
        (history["snapshot_date"] == latest_date)
        & (history["sector"] == "Industrials")
    ].copy()

    wanted_cols = [
        "ticker",
        "drawdown_52w_discount_score",
        "drawdown_3y_discount_score",
        "distance_ma200_discount_score",
        "price_position_3y_discount_score",
        "valid_discount_metrics",
        "price_discount_score",
        "revenue_growth_yoy",
        "revenue_growth_yoy_score",
        "net_income_growth_yoy",
        "net_income_growth_yoy_score",
        "operating_cash_flow_growth_yoy",
        "operating_cash_flow_growth_yoy_score",
        "diluted_eps_growth_yoy",
        "diluted_eps_growth_yoy_score",
        "net_margin",
        "net_margin_score",
        "ocf_margin",
        "ocf_margin_score",
        "fcf_margin",
        "fcf_margin_score",
        "valid_fundamental_components",
        "fundamental_preservation_score",
        "final_signal_score",
        "signal_percentile",
    ]

    cols = [c for c in wanted_cols if c in current.columns]
    audit = REFERENCE.merge(current[cols], on="ticker", how="left")

    audit["discount_diff"] = (
        audit["price_discount_score"] - audit["expected_discount_score"]
    )
    audit["fundamental_diff"] = (
        audit["fundamental_preservation_score"]
        - audit["expected_fundamental_score"]
    )
    audit["final_diff"] = (
        audit["final_signal_score"] - audit["expected_final_score"]
    )
    audit["percentile_diff"] = (
        audit["signal_percentile"] - audit["expected_percentile"]
    )

    audit["recalculated_20_80"] = (
        0.20 * audit["price_discount_score"]
        + 0.80 * audit["fundamental_preservation_score"]
    )
    audit["engine_vs_recalc_diff"] = (
        audit["final_signal_score"] - audit["recalculated_20_80"]
    )

    header("4. RESUMO — ESPERADO x GITHUB")
    summary_cols = [
        "ticker",
        "expected_discount_score",
        "price_discount_score",
        "discount_diff",
        "expected_fundamental_score",
        "fundamental_preservation_score",
        "fundamental_diff",
        "expected_final_score",
        "final_signal_score",
        "final_diff",
        "recalculated_20_80",
        "engine_vs_recalc_diff",
        "expected_percentile",
        "signal_percentile",
        "percentile_diff",
    ]
    print(audit[summary_cols].round(6).to_string(index=False))

    header("5. COMPONENTES DE DESCONTO")
    discount_cols = [
        "ticker",
        "drawdown_52w_discount_score",
        "drawdown_3y_discount_score",
        "distance_ma200_discount_score",
        "price_position_3y_discount_score",
        "valid_discount_metrics",
        "price_discount_score",
        "expected_discount_score",
        "discount_diff",
    ]
    discount_cols = [c for c in discount_cols if c in audit.columns]
    print(audit[discount_cols].round(6).to_string(index=False))

    header("6. COMPONENTES FUNDAMENTAIS")
    fundamental_cols = [
        "ticker",
        "revenue_growth_yoy",
        "revenue_growth_yoy_score",
        "net_income_growth_yoy",
        "net_income_growth_yoy_score",
        "operating_cash_flow_growth_yoy",
        "operating_cash_flow_growth_yoy_score",
        "diluted_eps_growth_yoy",
        "diluted_eps_growth_yoy_score",
        "net_margin",
        "net_margin_score",
        "ocf_margin",
        "ocf_margin_score",
        "fcf_margin",
        "fcf_margin_score",
        "valid_fundamental_components",
        "fundamental_preservation_score",
        "expected_fundamental_score",
        "fundamental_diff",
    ]
    fundamental_cols = [c for c in fundamental_cols if c in audit.columns]
    print(audit[fundamental_cols].round(6).to_string(index=False))

    header("7. HISTÓRICO RECENTE — INDUSTRIALS")
    industrial_history = (
        history[history["sector"] == "Industrials"]
        .copy()
        .sort_values(["snapshot_date", "ticker"])
    )

    recent = industrial_history[
        industrial_history["snapshot_date"]
        >= latest_date - pd.DateOffset(months=12)
    ][
        [
            "snapshot_date",
            "ticker",
            "price_discount_score",
            "fundamental_preservation_score",
            "final_signal_score",
            "signal_percentile",
        ]
    ].copy()

    print(recent.round(6).to_string(index=False))

    audit.to_csv(OUTPUT_CURRENT, index=False)
    industrial_history.to_csv(OUTPUT_HISTORY, index=False)

    header("8. DIAGNÓSTICO AUTOMÁTICO")
    for _, row in audit.iterrows():
        d = row["discount_diff"]
        f = row["fundamental_diff"]
        final = row["final_diff"]

        print(f"\n{row['ticker']}")
        print(f"  Δ desconto       : {d:+.6f}")
        print(f"  Δ fundamentos    : {f:+.6f}")
        print(f"  Δ score final    : {final:+.6f}")

        if abs(f) > abs(d):
            print("  Principal origem : FUNDAMENTOS")
        elif abs(d) > abs(f):
            print("  Principal origem : DESCONTO")
        else:
            print("  Principal origem : MISTA / INDETERMINADA")

    header("9. ARQUIVOS")
    print(f"\nSnapshot atual : {OUTPUT_CURRENT}")
    print(f"Histórico       : {OUTPUT_HISTORY}")
    print("\nSTATUS: AUDITORIA CONCLUÍDA — NENHUMA REGRA FOI ALTERADA.")


if __name__ == "__main__":
    run_audit()
