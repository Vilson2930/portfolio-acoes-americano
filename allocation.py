# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# allocation.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Aplicar os pesos setoriais aprovados pelo estudo:
#
#   Health Care              -> 25%
#   Industrials              -> 25%
#   Information Technology   -> 50%
#
# A carteira continua tendo:
#
#   5 ações de Health Care
#   5 ações de Industrials
#   5 ações de Information Technology
#
# Logo, com peso igual dentro de cada setor:
#
#   Health Care              -> 5% por ação
#   Industrials              -> 5% por ação
#   Information Technology   -> 10% por ação
#
# IMPORTANTE
# ----------
# Este módulo NÃO altera:
#
#   • seleção fundamental
#   • fronteira 5º x 6º
#   • Entry Engine
#   • sinal de compra
#
# Apenas adiciona a camada de alocação de capital após a carteira já estar definida.
# ======================================================================================

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# ======================================================================================
# 1. PESOS SETORIAIS APROVADOS PELO ESTUDO
# ======================================================================================

SECTOR_WEIGHTS: Dict[str, float] = {
    "Health Care": 0.25,
    "Industrials": 0.25,
    "Information Technology": 0.50,
}


# ======================================================================================
# 2. ESTRUTURA ESPERADA
# ======================================================================================

EXPECTED_STOCKS_PER_SECTOR = {
    "Health Care": 5,
    "Industrials": 5,
    "Information Technology": 5,
}

EXPECTED_TOTAL_STOCKS = 15


# ======================================================================================
# 3. VALIDAR CONFIGURAÇÃO
# ======================================================================================

def validate_allocation_config() -> None:
    """
    Valida a configuração fixa dos pesos setoriais.
    """

    expected_sectors = set(
        EXPECTED_STOCKS_PER_SECTOR.keys()
    )

    actual_sectors = set(
        SECTOR_WEIGHTS.keys()
    )

    if actual_sectors != expected_sectors:

        raise RuntimeError(
            "SECTOR_WEIGHTS possui setores diferentes "
            "da estrutura oficial da carteira."
        )

    total_weight = float(
        sum(
            SECTOR_WEIGHTS.values()
        )
    )

    if not np.isclose(
        total_weight,
        1.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            f"Pesos setoriais não somam 100%. "
            f"Total = {total_weight:.12f}"
        )

    for sector, weight in SECTOR_WEIGHTS.items():

        if (
            not np.isfinite(weight)
            or
            weight <= 0
            or
            weight > 1
        ):

            raise RuntimeError(
                f"{sector}: peso inválido = {weight}"
            )


# ======================================================================================
# 4. APLICAR PESOS À CARTEIRA
# ======================================================================================

def apply_portfolio_weights(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recebe o ranking final já classificado pelo Entry Engine e adiciona:

        sector_weight
        stock_weight

    A distribuição é igual entre as 5 ações de cada setor.

    Não altera:
        selection_score
        entry_signal
        final_signal_score
        signal_percentile
        prioridades
        seleção das ações
    """

    validate_allocation_config()

    if ranking is None or ranking.empty:

        raise ValueError(
            "Ranking vazio. Não é possível aplicar pesos."
        )

    required_columns = {
        "ticker",
        "sector",
    }

    if not required_columns.issubset(
        ranking.columns
    ):

        missing = sorted(
            required_columns
            -
            set(
                ranking.columns
            )
        )

        raise ValueError(
            f"Ranking não possui colunas obrigatórias: {missing}"
        )

    df = ranking.copy()

    # ------------------------------------------------------------------
    # Normalização
    # ------------------------------------------------------------------

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["sector"] = (
        df["sector"]
        .astype(str)
        .str.strip()
    )

    # ------------------------------------------------------------------
    # Validar total de ações
    # ------------------------------------------------------------------

    unique_tickers = int(
        df["ticker"]
        .nunique()
    )

    if unique_tickers != EXPECTED_TOTAL_STOCKS:

        raise RuntimeError(
            f"Carteira possui {unique_tickers} tickers únicos. "
            f"Esperado: {EXPECTED_TOTAL_STOCKS}."
        )

    # ------------------------------------------------------------------
    # Validar estrutura 5 / 5 / 5
    # ------------------------------------------------------------------

    sector_counts = (
        df
        .groupby(
            "sector"
        )[
            "ticker"
        ]
        .nunique()
    )

    for sector, expected_count in EXPECTED_STOCKS_PER_SECTOR.items():

        actual_count = int(
            sector_counts.get(
                sector,
                0,
            )
        )

        if actual_count != expected_count:

            raise RuntimeError(
                f"{sector}: {actual_count} ações. "
                f"Esperado: {expected_count}."
            )

    # ------------------------------------------------------------------
    # Peso setorial
    # ------------------------------------------------------------------

    df["sector_weight"] = (
        df["sector"]
        .map(
            SECTOR_WEIGHTS
        )
    )

    if df[
        "sector_weight"
    ].isna().any():

        unknown = sorted(
            df.loc[
                df[
                    "sector_weight"
                ]
                .isna(),
                "sector",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            f"Setores sem peso definido: {unknown}"
        )

    # ------------------------------------------------------------------
    # Peso por ação
    #
    # setor_weight / quantidade de ações do setor
    # ------------------------------------------------------------------

    df["stock_weight"] = (
        df.apply(
            lambda row:
                float(
                    SECTOR_WEIGHTS[
                        row[
                            "sector"
                        ]
                    ]
                )
                /
                int(
                    EXPECTED_STOCKS_PER_SECTOR[
                        row[
                            "sector"
                        ]
                    ]
                ),
            axis=1,
        )
    )

    # ------------------------------------------------------------------
    # Validar soma da carteira
    # ------------------------------------------------------------------

    total_stock_weight = float(
        df[
            "stock_weight"
        ]
        .sum()
    )

    if not np.isclose(
        total_stock_weight,
        1.0,
        atol=1e-12,
    ):

        raise RuntimeError(
            f"Pesos individuais não somam 100%. "
            f"Total = {total_stock_weight:.12f}"
        )

    # ------------------------------------------------------------------
    # Validar soma por setor
    # ------------------------------------------------------------------

    sector_weight_check = (
        df
        .groupby(
            "sector"
        )[
            "stock_weight"
        ]
        .sum()
    )

    for sector, expected_weight in SECTOR_WEIGHTS.items():

        actual_weight = float(
            sector_weight_check.get(
                sector,
                np.nan,
            )
        )

        if not np.isclose(
            actual_weight,
            expected_weight,
            atol=1e-12,
        ):

            raise RuntimeError(
                f"{sector}: peso agregado {actual_weight:.6f}. "
                f"Esperado: {expected_weight:.6f}."
            )

    return df


# ======================================================================================
# 5. RESUMO DA ALOCAÇÃO
# ======================================================================================

def build_allocation_summary(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retorna um resumo por setor com:
        número de ações
        peso do setor
        peso por ação
    """

    weighted = apply_portfolio_weights(
        ranking
    )

    rows = []

    for sector in SECTOR_WEIGHTS.keys():

        part = (
            weighted[
                weighted[
                    "sector"
                ]
                ==
                sector
            ]
            .copy()
        )

        rows.append(
            {
                "sector":
                    sector,

                "stocks":
                    int(
                        part[
                            "ticker"
                        ]
                        .nunique()
                    ),

                "sector_weight":
                    float(
                        part[
                            "stock_weight"
                        ]
                        .sum()
                    ),

                "stock_weight":
                    float(
                        part[
                            "stock_weight"
                        ]
                        .iloc[0]
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 6. AUDITORIA DA ALOCAÇÃO
# ======================================================================================

def audit_portfolio_weights(
    ranking: pd.DataFrame,
) -> Dict:
    """
    Auditoria simples para garantir que a alocação continua fiel ao estudo.
    """

    weighted = apply_portfolio_weights(
        ranking
    )

    summary = build_allocation_summary(
        weighted
    )

    expected_stock_weights = {
        "Health Care": 0.05,
        "Industrials": 0.05,
        "Information Technology": 0.10,
    }

    sector_checks = {}

    for sector in SECTOR_WEIGHTS.keys():

        row = (
            summary[
                summary[
                    "sector"
                ]
                ==
                sector
            ]
            .iloc[0]
        )

        sector_checks[
            sector
        ] = {
            "stocks_ok":
                int(
                    row[
                        "stocks"
                    ]
                )
                ==
                EXPECTED_STOCKS_PER_SECTOR[
                    sector
                ],

            "sector_weight_ok":
                np.isclose(
                    float(
                        row[
                            "sector_weight"
                        ]
                    ),
                    SECTOR_WEIGHTS[
                        sector
                    ],
                    atol=1e-12,
                ),

            "stock_weight_ok":
                np.isclose(
                    float(
                        row[
                            "stock_weight"
                        ]
                    ),
                    expected_stock_weights[
                        sector
                    ],
                    atol=1e-12,
                ),
        }

    all_checks = [
        check
        for sector_result in sector_checks.values()
        for check in sector_result.values()
    ]

    return {
        "number_of_stocks":
            int(
                weighted[
                    "ticker"
                ]
                .nunique()
            ),

        "total_weight":
            float(
                weighted[
                    "stock_weight"
                ]
                .sum()
            ),

        "sector_checks":
            sector_checks,

        "allocation_ok":
            bool(
                all(
                    all_checks
                )
                and
                weighted[
                    "ticker"
                ]
                .nunique()
                ==
                EXPECTED_TOTAL_STOCKS
                and
                np.isclose(
                    weighted[
                        "stock_weight"
                    ]
                    .sum(),
                    1.0,
                    atol=1e-12,
                )
            ),
    }


# ======================================================================================
# 7. TESTE DO MÓDULO
# ======================================================================================

if __name__ == "__main__":

    validate_allocation_config()

    print(
        "=" * 90
    )

    print(
        "PORTFOLIO ACOES AMERICANO — ALLOCATION ENGINE"
    )

    print(
        "=" * 90
    )

    print()

    print(
        "Pesos setoriais aprovados:"
    )

    for sector, weight in SECTOR_WEIGHTS.items():

        per_stock = (
            weight
            /
            EXPECTED_STOCKS_PER_SECTOR[
                sector
            ]
        )

        print(
            f"  {sector:<28} "
            f"{weight:>7.2%} "
            f"| por ação: {per_stock:>7.2%}"
        )

    print()

    print(
        "Total:",
        f"{sum(SECTOR_WEIGHTS.values()):.2%}"
    )

    print()

    print(
        "Allocation engine carregado com sucesso."
    )
