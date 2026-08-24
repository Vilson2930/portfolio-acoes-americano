# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# main.py — VERSÃO CORRIGIDA PARA ENTRY HISTÓRICO
# ======================================================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import pandas as pd

from config import SECTORS, OUTPUT_DIR, validate_config

from data import (
    build_base_universe,
    enrich_sectors,
    filter_target_sectors,
    download_fundamentals,
    prepare_selection_snapshot,
    download_prices,
)

from selection import select_portfolio, build_frontier_audit
from entry import classify_portfolio_entries, audit_entry_ranking
from report import generate_reports


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def print_header(title: str):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)


def run():

    started_at = datetime.now()

    print_header("PORTFOLIO ACOES AMERICANO — EXECUÇÃO DIÁRIA")

    # 1. CONFIG
    print_header("1. CONFIGURAÇÃO")
    validate_config()
    print("Configuração validada.")

    # 2. UNIVERSO
    print_header("2. UNIVERSO")
    universe = build_base_universe()
    print(f"Empresas no universo: {len(universe):,}")

    # 3. SETORES
    print_header("3. CLASSIFICAÇÃO SETORIAL")
    universe = enrich_sectors(universe)
    universe = filter_target_sectors(universe)

    counts = universe["sector"].value_counts()
    for sector in SECTORS:
        print(f"  {sector:<28} {int(counts.get(sector, 0)):,}")

    # 4. FUNDAMENTOS
    print_header("4. FUNDAMENTOS SEC")
    fundamentals, fundamental_errors = download_fundamentals(
        universe=universe,
        use_cache=True,
    )

    print(f"Observações fundamentais: {len(fundamentals):,}")
    print(f"Empresas com erro: {len(fundamental_errors):,}")

    if fundamentals.empty:
        raise RuntimeError("Nenhum fundamento SEC foi obtido.")

    # 5. SNAPSHOT ATUAL PARA SELEÇÃO
    print_header("5. SNAPSHOT FUNDAMENTAL")
    today = pd.Timestamp.today().normalize()

    snapshot = prepare_selection_snapshot(
        universe=universe,
        fundamentals=fundamentals,
        as_of_date=today,
    )

    print(f"Empresas no snapshot: {len(snapshot):,}")

    # 6. SELEÇÃO
    print_header("6. SELEÇÃO FUNDAMENTAL — TOP 5 POR SETOR")
    portfolio = select_portfolio(
        universe=snapshot,
        use_previous_portfolio=True,
    )

    for sector in SECTORS:
        tickers = portfolio.loc[
            portfolio["sector"] == sector,
            "ticker",
        ].tolist()
        print(f"  {sector:<28}: {', '.join(tickers)}")

    # 7. FRONTEIRA
    print_header("7. AUDITORIA DA FRONTEIRA")
    frontier = build_frontier_audit(snapshot)
    if frontier.empty:
        print("Sem auditoria disponível.")
    else:
        print(frontier.to_string(index=False))

    # 8. PREÇOS
    print_header("8. PREÇOS")
    selected_tickers = portfolio["ticker"].tolist()

    # 2013 garante janela para indicadores de 3 anos antes dos snapshots úteis.
    prices = download_prices(
        tickers=selected_tickers,
        start="2013-01-01",
    )

    print(f"Tickers com preços: {len(prices.columns)}")
    print(f"Primeira data: {prices.index.min().date()}")
    print(f"Última data: {prices.index.max().date()}")

    # 9. ENTRY — AGORA COM HISTÓRICO POINT-IN-TIME
    print_header("9. CLASSIFICAÇÃO DE ENTRADA — HISTÓRICO CÉLULA 41")

    ranking = classify_portfolio_entries(
        portfolio=portfolio,
        prices=prices,
        fundamentals_history=fundamentals,
        as_of_date=today,
    )

    audit = audit_entry_ranking(ranking)

    print("\nAuditoria:")
    print(f"  Ações               : {audit['number_of_stocks']}")
    print(f"  Setores              : {audit['number_of_sectors']}")
    print(f"  Entrada Forte        : {audit['entry_strong']}")
    print(f"  Entrada              : {audit['entry']}")
    print(f"  Aguardar             : {audit['wait']}")
    print(f"  Não comprar agora    : {audit['do_not_buy']}")
    print(f"  Estrutura OK         : {audit['structure_ok']}")

    if not audit["structure_ok"]:
        raise RuntimeError("Auditoria final da carteira falhou.")

    # 10. RANKING
    print_header("10. RANKING FINAL")

    columns = [
        c for c in [
            "sector",
            "buy_priority_sector",
            "ticker",
            "selection_score",
            "valuation_status",
            "discount_status",
            "fundamental_status",
            "relative_valuation_score",
            "price_discount_score",
            "fundamental_preservation_score",
            "final_signal_score",
            "signal_percentile",
            "entry_signal",
        ]
        if c in ranking.columns
    ]

    print(ranking[columns].to_string(index=False))

    # 11. RELATÓRIOS
    print_header("11. RELATÓRIOS")
    files = generate_reports(
        ranking=ranking,
        generated_at=started_at,
    )

    for name, path in files.items():
        print(f"{name:<22}: {path}")

    finished_at = datetime.now()

    print_header("EXECUÇÃO CONCLUÍDA")
    print(f"Início          : {started_at.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Fim             : {finished_at.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Tempo total     : {finished_at - started_at}")
    print("\nSTATUS: SUCESSO")

    return ranking


if __name__ == "__main__":
    run()
