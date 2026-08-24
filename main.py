# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# main.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Orquestrar a execução completa do sistema:
#
#   1. validar config
#   2. construir universo
#   3. classificar setores
#   4. baixar fundamentos
#   5. construir snapshot fundamental
#   6. selecionar 15 ações
#   7. baixar preços
#   8. classificar entrada
#   9. gerar CSV + PDF
#
# ======================================================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from config import (
    SECTORS,
    OUTPUT_DIR,
    validate_config,
)

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
)

from entry import (
    classify_portfolio_entries,
    audit_entry_ranking,
)

from report import (
    generate_reports,
)


# ======================================================================================
# 1. DIRETÓRIOS
# ======================================================================================

Path(
    OUTPUT_DIR
).mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================================
# 2. CABEÇALHO
# ======================================================================================

def print_header(
    title: str,
):

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        title
    )

    print(
        "=" * 110
    )


# ======================================================================================
# 3. EXECUÇÃO PRINCIPAL
# ======================================================================================

def run():

    started_at = (
        datetime.now()
    )

    print_header(
        "PORTFOLIO ACOES AMERICANO — EXECUÇÃO DIÁRIA"
    )

    print(
        f"\nInício: "
        f"{started_at.strftime('%d/%m/%Y %H:%M:%S')}"
    )

    # ==================================================================================
    # CONFIG
    # ==================================================================================

    print_header(
        "1. CONFIGURAÇÃO"
    )

    validate_config()

    print(
        "Configuração validada."
    )

    print(
        f"Setores: "
        f"{', '.join(SECTORS)}"
    )

    # ==================================================================================
    # UNIVERSO
    # ==================================================================================

    print_header(
        "2. UNIVERSO"
    )

    universe = (
        build_base_universe()
    )

    print(
        f"Empresas SEC: "
        f"{len(universe):,}"
    )

    # ==================================================================================
    # SETORES
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

    print(
        f"Empresas nos três setores: "
        f"{len(universe):,}"
    )

    sector_counts = (
        universe[
            "sector"
        ]
        .value_counts()
    )

    print(
        "\nDistribuição:"
    )

    for sector in SECTORS:

        print(
            f"  {sector:<28} "
            f"{int(sector_counts.get(sector, 0)):,}"
        )

    # ==================================================================================
    # FUNDAMENTOS SEC
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
        f"\nObservações fundamentais: "
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
    # SNAPSHOT PARA SELEÇÃO
    # ==================================================================================

    print_header(
        "5. SNAPSHOT FUNDAMENTAL"
    )

    snapshot = (
        prepare_selection_snapshot(
            universe=universe,
            fundamentals=fundamentals,
            as_of_date=pd.Timestamp.today(),
        )
    )

    print(
        f"Empresas no snapshot: "
        f"{len(snapshot):,}"
    )

    if snapshot.empty:

        raise RuntimeError(
            "Snapshot de seleção está vazio."
        )

    # ==================================================================================
    # SELEÇÃO DAS 15 AÇÕES
    # ==================================================================================

    print_header(
        "6. SELEÇÃO FUNDAMENTAL — TOP 5 POR SETOR"
    )

    portfolio = (
        select_portfolio(
            universe=snapshot,
            use_previous_portfolio=True,
        )
    )

    print(
        "\nCarteira selecionada:"
    )

    for sector in SECTORS:

        tickers = (
            portfolio[
                portfolio[
                    "sector"
                ]
                ==
                sector
            ][
                "ticker"
            ]
            .tolist()
        )

        print(
            f"  {sector:<28}: "
            f"{', '.join(tickers)}"
        )

    # ==================================================================================
    # AUDITORIA DA FRONTEIRA
    # ==================================================================================

    print_header(
        "7. AUDITORIA DA FRONTEIRA"
    )

    frontier = (
        build_frontier_audit(
            snapshot
        )
    )

    if frontier.empty:

        print(
            "Sem auditoria de fronteira disponível."
        )

    else:

        print(
            frontier.to_string(
                index=False
            )
        )

    # ==================================================================================
    # PREÇOS
    # ==================================================================================

    print_header(
        "8. PREÇOS"
    )

    selected_tickers = (
        portfolio[
            "ticker"
        ]
        .tolist()
    )

    prices = (
        download_prices(
            tickers=selected_tickers,
            start="2013-01-01",
        )
    )

    print(
        f"Tickers com preços: "
        f"{len(prices.columns)}"
    )

    print(
        f"Primeira data: "
        f"{prices.index.min().date()}"
    )

    print(
        f"Última data: "
        f"{prices.index.max().date()}"
    )

    # ==================================================================================
    # VALUATION HISTÓRICO
    # ==================================================================================
    #
    # Neste primeiro GitHub operacional, o valuation histórico ainda é opcional.
    #
    # Se existir um arquivo persistente produzido posteriormente, ele pode ser
    # carregado aqui.
    #
    # Sem essa base, Health Care utiliza os componentes disponíveis e o relatório
    # indicará N/D quando valuation não puder ser calculado.
    #
    # ==================================================================================

    historical_valuation = None

    # ==================================================================================
    # CLASSIFICAÇÃO DE ENTRADA
    # ==================================================================================

    print_header(
        "9. CLASSIFICAÇÃO DE ENTRADA"
    )

    ranking = (
        classify_portfolio_entries(
            portfolio=portfolio,
            prices=prices,
            historical_valuation=historical_valuation,
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
    # RESULTADO NO TERMINAL
    # ==================================================================================

    print_header(
        "10. RANKING FINAL"
    )

    columns_to_show = [
        col
        for col in [
            "sector",
            "buy_priority_sector",
            "ticker",
            "selection_score",
            "valuation_status",
            "discount_status",
            "fundamental_status",
            "final_signal_score",
            "signal_percentile",
            "entry_signal",
        ]
        if col in ranking.columns
    ]

    print(
        ranking[
            columns_to_show
        ]
        .to_string(
            index=False
        )
    )

    # ==================================================================================
    # RELATÓRIOS
    # ==================================================================================

    print_header(
        "11. RELATÓRIOS"
    )

    report_files = (
        generate_reports(
            ranking=ranking,
            generated_at=started_at,
        )
    )

    for name, path in (
        report_files.items()
    ):

        print(
            f"{name:<22}: "
            f"{path}"
        )

    # ==================================================================================
    # FINAL
    # ==================================================================================

    finished_at = (
        datetime.now()
    )

    elapsed = (
        finished_at
        -
        started_at
    )

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
        f"{elapsed}"
    )

    print(
        "\nSTATUS: SUCESSO"
    )

    return ranking


# ======================================================================================
# 4. EXECUÇÃO
# ======================================================================================

if __name__ == "__main__":

    run()
