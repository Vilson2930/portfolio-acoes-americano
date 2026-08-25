# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# main.py — PIPELINE FIEL AO ESTUDO
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

from selection import (
    select_portfolio,
    build_frontier_audit,
    get_boundary_test_tickers,
)

from entry import (
    classify_portfolio_entries,
    audit_entry_ranking,
)

from allocation import (
    apply_portfolio_weights,
    audit_portfolio_weights,
)

from report import generate_reports


Path(OUTPUT_DIR).mkdir(
    parents=True,
    exist_ok=True,
)


def print_header(
    title: str,
):
    print(
        "\n"
        +
        "=" * 110
    )

    print(title)

    print(
        "=" * 110
    )


def run():

    started_at = datetime.now()

    print_header(
        "PORTFOLIO ACOES AMERICANO — EXECUÇÃO DIÁRIA"
    )

    # ==================================================================================
    # 1. CONFIG
    # ==================================================================================

    print_header(
        "1. CONFIGURAÇÃO"
    )

    validate_config()

    print(
        "Configuração validada."
    )

    # ==================================================================================
    # 2. UNIVERSO
    # ==================================================================================

    print_header(
        "2. UNIVERSO"
    )

    universe = (
        build_base_universe()
    )

    print(
        f"Empresas no universo: "
        f"{len(universe):,}"
    )

    # ==================================================================================
    # 3. SETORES
    # ==================================================================================

    print_header(
        "3. CLASSIFICAÇÃO SETORIAL"
    )

    universe = (
        enrich_sectors(
            universe
        )
    )

    universe = (
        filter_target_sectors(
            universe
        )
    )

    counts = (
        universe[
            "sector"
        ]
        .value_counts()
    )

    for sector in SECTORS:

        print(
            f"  {sector:<28} "
            f"{int(counts.get(sector, 0)):,}"
        )

    # ==================================================================================
    # 4. FUNDAMENTOS
    # ==================================================================================

    print_header(
        "4. FUNDAMENTOS SEC"
    )

    fundamentals, fundamental_errors = (
        download_fundamentals(
            universe=universe,
            use_cache=True,
        )
    )

    print(
        f"Observações fundamentais: "
        f"{len(fundamentals):,}"
    )

    print(
        f"Empresas com erro: "
        f"{len(fundamental_errors):,}"
    )

    if fundamentals.empty:

        raise RuntimeError(
            "Nenhum fundamento SEC foi obtido."
        )

    # ==================================================================================
    # 5. SNAPSHOT
    # ==================================================================================

    print_header(
        "5. SNAPSHOT FUNDAMENTAL"
    )

    today = (
        pd.Timestamp.today()
        .normalize()
    )

    snapshot = (
        prepare_selection_snapshot(
            universe=universe,
            fundamentals=fundamentals,
            as_of_date=today,
        )
    )

    print(
        f"Empresas no snapshot: "
        f"{len(snapshot):,}"
    )

    # ==================================================================================
    # 6. SELEÇÃO FUNDAMENTAL PURA
    # ==================================================================================

    print_header(
        "6. SELEÇÃO FUNDAMENTAL — TOP 5 PURO"
    )

    preliminary_portfolio = (
        select_portfolio(
            universe=snapshot,
            use_previous_portfolio=False,
            prices=None,
        )
    )

    for sector in SECTORS:

        tickers = (
            preliminary_portfolio.loc[
                preliminary_portfolio[
                    "sector"
                ]
                ==
                sector,
                "ticker",
            ]
            .tolist()
        )

        print(
            f"  {sector:<28}: "
            f"{', '.join(tickers)}"
        )

    # ==================================================================================
    # 7. CANDIDATOS DA FRONTEIRA 5º x 6º
    # ==================================================================================

    print_header(
        "7. FRONTEIRA FUNDAMENTAL — 5º VS 6º"
    )

    frontier_pre = (
        build_frontier_audit(
            snapshot,
            prices=None,
        )
    )

    print(
        frontier_pre.to_string(
            index=False
        )
    )

    test_tickers = (
        get_boundary_test_tickers(
            snapshot
        )
    )

    print(
        f"\nTickers necessários para o teste: "
        f"{len(test_tickers)}"
    )

    print(
        ", ".join(
            test_tickers
        )
    )

    # ==================================================================================
    # 8. PREÇOS
    #
    # Baixamos Top-5 + candidatos 6º antes de fechar a carteira.
    # O início em 2013 atende também o Entry Engine.
    # A fronteira usa internamente apenas dados >= 2024-01-01,
    # exatamente como a Célula 31.
    # ==================================================================================

    print_header(
        "8. PREÇOS — CARTEIRA + CANDIDATOS DA FRONTEIRA"
    )

    prices_all = (
        download_prices(
            tickers=test_tickers,
            start="2013-01-01",
        )
    )

    print(
        f"Tickers com preços: "
        f"{len(prices_all.columns)}"
    )

    print(
        f"Primeira data: "
        f"{prices_all.index.min().date()}"
    )

    print(
        f"Última data: "
        f"{prices_all.index.max().date()}"
    )

    # ==================================================================================
    # 9. TESTE FINAL DA FRONTEIRA — CÉLULA 31
    # ==================================================================================

    print_header(
        "9. TESTE FINAL DA FRONTEIRA — CÉLULA 31"
    )

    frontier = (
        build_frontier_audit(
            snapshot,
            prices=prices_all,
        )
    )

    print(
        frontier.to_string(
            index=False
        )
    )

    # ==================================================================================
    # 10. CARTEIRA FINAL 15 AÇÕES
    # ==================================================================================

    print_header(
        "10. CARTEIRA FINAL — APÓS FRONTEIRA"
    )

    portfolio = (
        select_portfolio(
            universe=snapshot,
            use_previous_portfolio=False,
            prices=prices_all,
        )
    )

    for sector in SECTORS:

        tickers = (
            portfolio.loc[
                portfolio[
                    "sector"
                ]
                ==
                sector,
                "ticker",
            ]
            .tolist()
        )

        print(
            f"  {sector:<28}: "
            f"{', '.join(tickers)}"
        )

    # ==================================================================================
    # 11. PREÇOS DA CARTEIRA FINAL PARA ENTRY
    # ==================================================================================

    selected_tickers = (
        portfolio[
            "ticker"
        ]
        .tolist()
    )

    missing_final = [
        ticker
        for ticker in selected_tickers
        if ticker not in prices_all.columns
    ]

    if missing_final:

        raise RuntimeError(
            f"Tickers finais sem preço: "
            f"{missing_final}"
        )

    prices = (
        prices_all[
            selected_tickers
        ]
        .copy()
    )

    # ==================================================================================
    # 12. ENTRY — HISTÓRICO POINT-IN-TIME
    # ==================================================================================

    print_header(
        "11. CLASSIFICAÇÃO DE ENTRADA — HISTÓRICO CÉLULA 41"
    )

    ranking = (
        classify_portfolio_entries(
            portfolio=portfolio,
            prices=prices,
            fundamentals_history=fundamentals,
            as_of_date=today,
        )
    )

    audit = (
        audit_entry_ranking(
            ranking
        )
    )

    print(
        "\nAuditoria:"
    )

    print(
        f"  Ações               : "
        f"{audit['number_of_stocks']}"
    )

    print(
        f"  Setores              : "
        f"{audit['number_of_sectors']}"
    )

    print(
        f"  Entrada Forte        : "
        f"{audit['entry_strong']}"
    )

    print(
        f"  Entrada              : "
        f"{audit['entry']}"
    )

    print(
        f"  Aguardar             : "
        f"{audit['wait']}"
    )

    print(
        f"  Não comprar agora    : "
        f"{audit['do_not_buy']}"
    )

    print(
        f"  Estrutura OK         : "
        f"{audit['structure_ok']}"
    )

    if not audit[
        "structure_ok"
    ]:

        raise RuntimeError(
            "Auditoria final da carteira falhou."
        )

    # ==================================================================================
    # 13. ALOCAÇÃO DE CAPITAL — ESTUDO CIENTÍFICO
    # ==================================================================================

    print_header(
        "12. ALOCAÇÃO DE CAPITAL"
    )

    ranking = (
        apply_portfolio_weights(
            ranking
        )
    )

    allocation_audit = (
        audit_portfolio_weights(
            ranking
        )
    )

    allocation_summary = (
        ranking
        .groupby(
            "sector"
        )
        .agg(
            stocks=(
                "ticker",
                "nunique",
            ),
            sector_weight=(
                "stock_weight",
                "sum",
            ),
            stock_weight=(
                "stock_weight",
                "first",
            ),
        )
        .reset_index()
    )

    print(
        "\nPesos aprovados pelo estudo:"
    )

    for _, row in allocation_summary.iterrows():

        print(
            f"  {row['sector']:<28}"
            f"{float(row['sector_weight']):>7.2%} "
            f"| {int(row['stocks'])} ações "
            f"× {float(row['stock_weight']):.2%}"
        )

    print(
        f"\nPeso total da carteira   : "
        f"{allocation_audit['total_weight']:.2%}"
    )

    print(
        f"Alocação válida           : "
        f"{allocation_audit['allocation_ok']}"
    )

    if not allocation_audit[
        "allocation_ok"
    ]:

        raise RuntimeError(
            "Falha na auditoria da alocação."
        )

    # ==================================================================================
    # 14. RANKING FINAL
    # ==================================================================================

    print_header(
        "13. RANKING FINAL"
    )

    columns = [
        c
        for c in [
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
            "sector_weight",
            "stock_weight",
            "entry_signal",
        ]
        if c in ranking.columns
    ]

    print(
        ranking[
            columns
        ]
        .to_string(
            index=False
        )
    )

    # ==================================================================================
    # 15. RELATÓRIOS
    # ==================================================================================

    print_header(
        "14. RELATÓRIOS"
    )

    files = (
        generate_reports(
            ranking=ranking,
            generated_at=started_at,
        )
    )

    for name, path in files.items():

        print(
            f"{name:<22}: "
            f"{path}"
        )

    finished_at = datetime.now()

    print_header(
        "EXECUÇÃO CONCLUÍDA"
    )

    print(
        f"Início          : "
        f"{started_at.strftime('%d/%m/%Y %H:%M:%S')}"
    )

    print(
        f"Fim             : "
        f"{finished_at.strftime('%d/%m/%Y %H:%M:%S')}"
    )

    print(
        f"Tempo total     : "
        f"{finished_at - started_at}"
    )

    print(
        "\nSTATUS: SUCESSO"
    )

    return ranking


if __name__ == "__main__":

    run()
