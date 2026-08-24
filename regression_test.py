# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# regression_test.py
# ======================================================================================
#
# OBJETIVO
# --------
# Teste de regressão Colab x GitHub.
#
# Este arquivo NÃO altera a carteira, NÃO altera os pesos e NÃO participa
# da execução diária. Ele apenas verifica se o motor operacional consegue
# reproduzir a CÉLULA 41 usando:
#
#   • mesma data de referência;
#   • mesmas 15 ações;
#   • mesmos 3 setores;
#   • mesma arquitetura de entrada.
#
# GABARITO — CÉLULA 41
# --------------------
# Health Care:
#   RMD   0.694   57.0%   ENTRADA
#   WST   0.486   30.5%   AGUARDAR
#   DXCM  0.468   28.8%   AGUARDAR
#   EW    0.401   23.2%   NÃO COMPRAR AGORA
#   MRNA  0.177   10.5%   NÃO COMPRAR AGORA
#
# Industrials:
#   VRT   0.911   95.1%   ENTRADA
#   FIX   0.884   92.2%   ENTRADA
#   GE    0.855   90.7%   ENTRADA
#   RTX   0.649   68.0%   ENTRADA
#   GEV   NaN     NaN     AGUARDAR
#
# Information Technology:
#   SNDK  1.000   91.5%   ENTRADA FORTE
#   WDC   0.800   73.0%   ENTRADA
#   NVDA  0.500   35.9%   AGUARDAR
#   VRSN  0.500   35.9%   AGUARDAR
#   FICO  0.200    0.4%   NÃO COMPRAR AGORA
#
# IMPORTANTE
# ----------
# Este teste NÃO "ajusta" o motor para fazer os números baterem.
# Ele somente mede a fidelidade.
#
# ======================================================================================

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from config import SECTORS

from data import (
    build_base_universe,
    enrich_sectors,
    download_fundamentals,
    download_prices,
)

from entry import (
    classify_portfolio_entries,
)


# ======================================================================================
# 1. CONFIGURAÇÃO DO TESTE
# ======================================================================================

REFERENCE_DATE = pd.Timestamp("2026-08-24")

SCORE_TOLERANCE = 0.025
PERCENTILE_TOLERANCE = 0.035

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_CSV = (
    OUTPUT_DIR
    /
    "regression_cell41_comparison.csv"
)


# ======================================================================================
# 2. CARTEIRA FIXA DA CÉLULA 41
# ======================================================================================

REFERENCE_PORTFOLIO = pd.DataFrame(
    [
        # Health Care
        {
            "ticker": "EW",
            "sector": "Health Care",
        },
        {
            "ticker": "DXCM",
            "sector": "Health Care",
        },
        {
            "ticker": "WST",
            "sector": "Health Care",
        },
        {
            "ticker": "MRNA",
            "sector": "Health Care",
        },
        {
            "ticker": "RMD",
            "sector": "Health Care",
        },

        # Industrials
        {
            "ticker": "VRT",
            "sector": "Industrials",
        },
        {
            "ticker": "FIX",
            "sector": "Industrials",
        },
        {
            "ticker": "GE",
            "sector": "Industrials",
        },
        {
            "ticker": "GEV",
            "sector": "Industrials",
        },
        {
            "ticker": "RTX",
            "sector": "Industrials",
        },

        # Information Technology
        {
            "ticker": "VRSN",
            "sector": "Information Technology",
        },
        {
            "ticker": "NVDA",
            "sector": "Information Technology",
        },
        {
            "ticker": "SNDK",
            "sector": "Information Technology",
        },
        {
            "ticker": "WDC",
            "sector": "Information Technology",
        },
        {
            "ticker": "FICO",
            "sector": "Information Technology",
        },
    ]
)


# ======================================================================================
# 3. GABARITO NUMÉRICO DA CÉLULA 41
# ======================================================================================

REFERENCE = pd.DataFrame(
    [
        {
            "ticker": "RMD",
            "expected_score": 0.694,
            "expected_percentile": 0.569524,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "WST",
            "expected_score": 0.486,
            "expected_percentile": 0.305,
            "expected_signal": "AGUARDAR",
        },
        {
            "ticker": "DXCM",
            "expected_score": 0.468,
            "expected_percentile": 0.288,
            "expected_signal": "AGUARDAR",
        },
        {
            "ticker": "EW",
            "expected_score": 0.401,
            "expected_percentile": 0.232,
            "expected_signal": "NÃO COMPRAR AGORA",
        },
        {
            "ticker": "MRNA",
            "expected_score": 0.177,
            "expected_percentile": 0.105,
            "expected_signal": "NÃO COMPRAR AGORA",
        },

        {
            "ticker": "VRT",
            "expected_score": 0.911,
            "expected_percentile": 0.951220,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "FIX",
            "expected_score": 0.884,
            "expected_percentile": 0.921951,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "GE",
            "expected_score": 0.855,
            "expected_percentile": 0.907317,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "RTX",
            "expected_score": 0.649,
            "expected_percentile": 0.680488,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "GEV",
            "expected_score": np.nan,
            "expected_percentile": np.nan,
            "expected_signal": "AGUARDAR",
        },

        {
            "ticker": "SNDK",
            "expected_score": 1.000,
            "expected_percentile": 0.915480,
            "expected_signal": "ENTRADA FORTE",
        },
        {
            "ticker": "WDC",
            "expected_score": 0.800,
            "expected_percentile": 0.729537,
            "expected_signal": "ENTRADA",
        },
        {
            "ticker": "NVDA",
            "expected_score": 0.500,
            "expected_percentile": 0.359,
            "expected_signal": "AGUARDAR",
        },
        {
            "ticker": "VRSN",
            "expected_score": 0.500,
            "expected_percentile": 0.359,
            "expected_signal": "AGUARDAR",
        },
        {
            "ticker": "FICO",
            "expected_score": 0.200,
            "expected_percentile": 0.004,
            "expected_signal": "NÃO COMPRAR AGORA",
        },
    ]
)


# ======================================================================================
# 4. HELPERS
# ======================================================================================

def _is_close(
    actual,
    expected,
    tolerance,
):
    """
    Compara números preservando NaN esperado.
    """

    actual = pd.to_numeric(
        pd.Series([actual]),
        errors="coerce",
    ).iloc[0]

    expected = pd.to_numeric(
        pd.Series([expected]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(expected):
        return pd.isna(actual)

    if pd.isna(actual):
        return False

    return (
        abs(
            float(actual)
            -
            float(expected)
        )
        <=
        tolerance
    )


def _difference(
    actual,
    expected,
):
    actual = pd.to_numeric(
        pd.Series([actual]),
        errors="coerce",
    ).iloc[0]

    expected = pd.to_numeric(
        pd.Series([expected]),
        errors="coerce",
    ).iloc[0]

    if (
        pd.isna(actual)
        or
        pd.isna(expected)
    ):
        return np.nan

    return float(
        actual
        -
        expected
    )


def print_header(
    title: str,
):

    print(
        "\n"
        +
        "=" * 120
    )

    print(
        title
    )

    print(
        "=" * 120
    )


# ======================================================================================
# 5. VALIDAR ESTRUTURA DO GABARITO
# ======================================================================================

def validate_reference():

    if (
        REFERENCE_PORTFOLIO[
            "ticker"
        ]
        .nunique()
        !=
        15
    ):

        raise RuntimeError(
            "Gabarito não possui 15 ações."
        )

    counts = (
        REFERENCE_PORTFOLIO
        .groupby(
            "sector"
        )[
            "ticker"
        ]
        .nunique()
    )

    for sector in SECTORS:

        if (
            int(
                counts.get(
                    sector,
                    0,
                )
            )
            !=
            5
        ):

            raise RuntimeError(
                f"Gabarito inválido para {sector}."
            )


# ======================================================================================
# 6. RECUPERAR UNIVERSO / CIKs
# ======================================================================================

def build_test_universe():

    universe = (
        build_base_universe()
    )

    universe = (
        enrich_sectors(
            universe
        )
    )

    wanted = set(
        REFERENCE_PORTFOLIO[
            "ticker"
        ]
        .tolist()
    )

    test_universe = universe[
        universe[
            "ticker"
        ]
        .isin(
            wanted
        )
    ].copy()

    found = set(
        test_universe[
            "ticker"
        ]
        .tolist()
    )

    missing = sorted(
        wanted
        -
        found
    )

    if missing:

        raise RuntimeError(
            "Tickers da Célula 41 ausentes no universo atual: "
            +
            ", ".join(
                missing
            )
        )

    # Força os setores exatamente como estavam congelados no estudo.
    test_universe = (
        test_universe
        .drop(
            columns=[
                "sector"
            ],
            errors="ignore",
        )
        .merge(
            REFERENCE_PORTFOLIO,
            on="ticker",
            how="left",
        )
    )

    return test_universe


# ======================================================================================
# 7. EXECUTAR REGRESSÃO
# ======================================================================================

def run_regression_test():

    validate_reference()

    print_header(
        "REGRESSION TEST — CÉLULA 41 x GITHUB"
    )

    print(
        f"\nData de referência             : "
        f"{REFERENCE_DATE.date()}"
    )

    print(
        f"Tolerância final_signal_score  : "
        f"{SCORE_TOLERANCE:.3f}"
    )

    print(
        f"Tolerância signal_percentile   : "
        f"{PERCENTILE_TOLERANCE:.3f}"
    )

    # ------------------------------------------------------------------
    # UNIVERSO FIXO
    # ------------------------------------------------------------------

    print_header(
        "1. CARTEIRA FIXA DA CÉLULA 41"
    )

    test_universe = (
        build_test_universe()
    )

    for sector in SECTORS:

        tickers = (
            REFERENCE_PORTFOLIO[
                REFERENCE_PORTFOLIO[
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
            f"{sector:<28}: "
            f"{', '.join(tickers)}"
        )

    # ------------------------------------------------------------------
    # FUNDAMENTOS
    # ------------------------------------------------------------------

    print_header(
        "2. FUNDAMENTOS SEC"
    )

    fundamentals, errors = (
        download_fundamentals(
            universe=test_universe,
            use_cache=True,
        )
    )

    print(
        f"\nObservações fundamentais       : "
        f"{len(fundamentals):,}"
    )

    print(
        f"Empresas com erro              : "
        f"{len(errors)}"
    )

    if errors:

        print(
            errors.to_string(
                index=False
            )
            if isinstance(
                errors,
                pd.DataFrame,
            )
            else errors
        )

    if fundamentals.empty:

        raise RuntimeError(
            "Fundamentos vazios."
        )

    # ------------------------------------------------------------------
    # PREÇOS
    # ------------------------------------------------------------------

    print_header(
        "3. PREÇOS"
    )

    tickers = (
        REFERENCE_PORTFOLIO[
            "ticker"
        ]
        .tolist()
    )

    prices = (
        download_prices(
            tickers=tickers,
            start="2013-01-01",
            end=(
                REFERENCE_DATE
                +
                pd.Timedelta(
                    days=1
                )
            )
            .strftime(
                "%Y-%m-%d"
            ),
        )
    )

    if prices.empty:

        raise RuntimeError(
            "Preços vazios."
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
    # MOTOR DE ENTRADA
    # ------------------------------------------------------------------

    print_header(
        "4. EXECUTAR ENTRY ENGINE"
    )

    ranking = (
        classify_portfolio_entries(
            portfolio=REFERENCE_PORTFOLIO,
            prices=prices,
            fundamentals_history=fundamentals,
            as_of_date=REFERENCE_DATE,
        )
    )

    # ------------------------------------------------------------------
    # COMPARAÇÃO
    # ------------------------------------------------------------------

    print_header(
        "5. COMPARAÇÃO NUMÉRICA"
    )

    actual = ranking[
        [
            "ticker",
            "sector",
            "final_signal_score",
            "signal_percentile",
            "entry_signal",
        ]
    ].copy()

    comparison = (
        REFERENCE
        .merge(
            actual,
            on="ticker",
            how="left",
        )
    )

    comparison[
        "score_diff"
    ] = comparison.apply(
        lambda row:
            _difference(
                row[
                    "final_signal_score"
                ],
                row[
                    "expected_score"
                ],
            ),
        axis=1,
    )

    comparison[
        "percentile_diff"
    ] = comparison.apply(
        lambda row:
            _difference(
                row[
                    "signal_percentile"
                ],
                row[
                    "expected_percentile"
                ],
            ),
        axis=1,
    )

    comparison[
        "score_ok"
    ] = comparison.apply(
        lambda row:
            _is_close(
                row[
                    "final_signal_score"
                ],
                row[
                    "expected_score"
                ],
                SCORE_TOLERANCE,
            ),
        axis=1,
    )

    comparison[
        "percentile_ok"
    ] = comparison.apply(
        lambda row:
            _is_close(
                row[
                    "signal_percentile"
                ],
                row[
                    "expected_percentile"
                ],
                PERCENTILE_TOLERANCE,
            ),
        axis=1,
    )

    comparison[
        "signal_ok"
    ] = (
        comparison[
            "entry_signal"
        ]
        ==
        comparison[
            "expected_signal"
        ]
    )

    comparison[
        "row_ok"
    ] = (
        comparison[
            "score_ok"
        ]
        &
        comparison[
            "percentile_ok"
        ]
        &
        comparison[
            "signal_ok"
        ]
    )

    # ------------------------------------------------------------------
    # EXIBIÇÃO
    # ------------------------------------------------------------------

    display_cols = [
        "sector",
        "ticker",
        "expected_score",
        "final_signal_score",
        "score_diff",
        "expected_percentile",
        "signal_percentile",
        "percentile_diff",
        "expected_signal",
        "entry_signal",
        "score_ok",
        "percentile_ok",
        "signal_ok",
        "row_ok",
    ]

    print(
        comparison[
            display_cols
        ]
        .sort_values(
            [
                "sector",
                "ticker",
            ]
        )
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # SALVAR
    # ------------------------------------------------------------------

    comparison.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # ------------------------------------------------------------------
    # RESULTADO
    # ------------------------------------------------------------------

    print_header(
        "6. RESULTADO DA REGRESSÃO"
    )

    total = len(
        comparison
    )

    score_passed = int(
        comparison[
            "score_ok"
        ].sum()
    )

    percentile_passed = int(
        comparison[
            "percentile_ok"
        ].sum()
    )

    signal_passed = int(
        comparison[
            "signal_ok"
        ].sum()
    )

    row_passed = int(
        comparison[
            "row_ok"
        ].sum()
    )

    print(
        f"\nScore dentro da tolerância      : "
        f"{score_passed}/{total}"
    )

    print(
        f"Percentil dentro da tolerância  : "
        f"{percentile_passed}/{total}"
    )

    print(
        f"Classificação idêntica          : "
        f"{signal_passed}/{total}"
    )

    print(
        f"Linhas 100% aprovadas           : "
        f"{row_passed}/{total}"
    )

    print(
        f"\nArquivo de auditoria            : "
        f"{OUTPUT_CSV}"
    )

    full_pass = bool(
        comparison[
            "row_ok"
        ].all()
    )

    print(
        "\n"
        +
        "-" * 120
    )

    if full_pass:

        print(
            "STATUS: REGRESSÃO APROVADA — "
            "GITHUB REPRODUZ A CÉLULA 41 DENTRO DAS TOLERÂNCIAS DEFINIDAS."
        )

    else:

        print(
            "STATUS: REGRESSÃO NÃO APROVADA — "
            "EXISTEM DIVERGÊNCIAS A SEREM AUDITADAS."
        )

        failed = comparison[
            ~comparison[
                "row_ok"
            ]
        ][
            [
                "ticker",
                "score_ok",
                "percentile_ok",
                "signal_ok",
            ]
        ]

        print(
            "\nTickers divergentes:"
        )

        print(
            failed.to_string(
                index=False
            )
        )

    print(
        "-" * 120
    )

    return comparison


# ======================================================================================
# 8. EXECUÇÃO
# ======================================================================================

if __name__ == "__main__":

    result = (
        run_regression_test()
    )

    if not result[
        "row_ok"
    ].all():

        # Faz o GitHub Actions marcar o teste como falha
        # quando a reprodução não estiver dentro da tolerância.
        sys.exit(1)

    sys.exit(0)
