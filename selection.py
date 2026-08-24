# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# selection.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Selecionar dinamicamente as 15 ações da carteira:
#
#   • 5 Health Care
#   • 5 Industrials
#   • 5 Information Technology
#
# FATORES DEFINITIVOS DO ESTUDO
# -----------------------------
#
# Health Care
#   -> Financial Strength
#      cash_assets ↑
#      debt_assets ↓
#      debt_equity ↓
#
# Industrials
#   -> Growth
#      revenue_growth ↑
#      eps_growth ↑
#      operating_cash_flow_growth ↑
#
# Information Technology
#   -> Quality
#      roa ↑
#      roe ↑
#      operating_margin ↑
#      net_margin ↑
#
# METODOLOGIA
# -----------
#   1) métricas fundamentais
#   2) winsorização P5-P95 dentro do setor
#   3) percentis dentro do setor
#   4) média dos componentes válidos
#   5) mínimo de componentes:
#          Financial Strength = 2/3
#          Growth             = 2/3
#          Quality            = 3/4
#   6) ranking
#   7) Top 5 por setor
#   8) proteção de fronteira 5º vs 6º
#
# ======================================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    SECTORS,
    SECTOR_TARGETS,
    SELECTION_FACTORS,
    FRONTIER_MIN_RELATIVE_GAP,
    CURRENT_PORTFOLIO_FILE,
)


# ======================================================================================
# 1. CONFIGURAÇÃO DOS FATORES — CONGELADA PELO ESTUDO
# ======================================================================================

FACTOR_DEFINITIONS = {

    "financial_strength": {

        "higher": [
            "cash_assets",
        ],

        "lower": [
            "debt_assets",
            "debt_equity",
        ],

        "minimum_components": 2,
    },

    "growth": {

        "higher": [
            "revenue_growth",
            "eps_growth",
            "operating_cash_flow_growth",
        ],

        "lower": [],

        "minimum_components": 2,
    },

    "quality": {

        "higher": [
            "roa",
            "roe",
            "operating_margin",
            "net_margin",
        ],

        "lower": [],

        "minimum_components": 3,
    },
}


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def safe_divide(
    numerator,
    denominator,
):

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = (
        numerator
        /
        denominator.replace(
            0,
            np.nan,
        )
    )

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def winsorize_series(
    series: pd.Series,
) -> pd.Series:
    """
    Winsorização P5-P95.

    Reproduz a metodologia utilizada no estudo.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if len(valid) < 10:
        return values

    lower = valid.quantile(0.05)
    upper = valid.quantile(0.95)

    return values.clip(
        lower=lower,
        upper=upper,
    )


def percentile_score(
    series: pd.Series,
    lower_is_better: bool = False,
) -> pd.Series:
    """
    Sempre:
        score maior = empresa melhor.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return values.rank(
        pct=True,
        ascending=not lower_is_better,
        method="average",
    )


# ======================================================================================
# 3. NORMALIZAÇÃO DAS COLUNAS FUNDAMENTAIS
# ======================================================================================

def build_derived_metrics(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    df = fundamentals.copy()

    # ------------------------------------------------------------------
    # Normalizar ticker
    # ------------------------------------------------------------------

    if "ticker" in df.columns:

        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(
                ".",
                "-",
                regex=False,
            )
        )

    # ------------------------------------------------------------------
    # Colunas numéricas possíveis
    # ------------------------------------------------------------------

    numeric_columns = [

        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",

        "assets",
        "equity",
        "cash",

        "total_debt",
        "long_term_debt",
        "short_term_debt",

        "diluted_eps",

        # métricas eventualmente já calculadas
        "revenue_growth",
        "eps_growth",
        "operating_cash_flow_growth",

        "revenue_growth_yoy",
        "diluted_eps_growth_yoy",
        "operating_cash_flow_growth_yoy",

        "operating_margin",
        "net_margin",

        "roa",
        "roe",

        "cash_assets",
        "debt_assets",
        "debt_equity",

        # aliases do arquivo anterior
        "cash_to_assets",
        "debt_to_assets",
        "debt_to_equity",
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # ------------------------------------------------------------------
    # Garantir colunas RAW necessárias
    # ------------------------------------------------------------------

    for col in [
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "assets",
        "equity",
        "cash",
        "total_debt",
        "long_term_debt",
        "short_term_debt",
    ]:

        if col not in df.columns:
            df[col] = np.nan

    # ------------------------------------------------------------------
    # TOTAL DEBT
    #
    # Preferência:
    #   1) total_debt fornecido pela base
    #   2) short_term_debt + long_term_debt
    #   3) long_term_debt
    # ------------------------------------------------------------------

    calculated_total_debt = (
        df["short_term_debt"].fillna(0)
        +
        df["long_term_debt"].fillna(0)
    )

    no_debt_components = (
        df["short_term_debt"].isna()
        &
        df["long_term_debt"].isna()
    )

    calculated_total_debt.loc[
        no_debt_components
    ] = np.nan

    df["total_debt"] = (
        df["total_debt"]
        .combine_first(
            calculated_total_debt
        )
        .combine_first(
            df["long_term_debt"]
        )
    )

    # ------------------------------------------------------------------
    # MARGENS
    # ------------------------------------------------------------------

    calculated_operating_margin = safe_divide(
        df["operating_income"],
        df["revenue"],
    )

    calculated_net_margin = safe_divide(
        df["net_income"],
        df["revenue"],
    )

    if "operating_margin" not in df.columns:
        df["operating_margin"] = calculated_operating_margin
    else:
        df["operating_margin"] = (
            df["operating_margin"]
            .combine_first(
                calculated_operating_margin
            )
        )

    if "net_margin" not in df.columns:
        df["net_margin"] = calculated_net_margin
    else:
        df["net_margin"] = (
            df["net_margin"]
            .combine_first(
                calculated_net_margin
            )
        )

    # ------------------------------------------------------------------
    # ROA / ROE
    # ------------------------------------------------------------------

    calculated_roa = safe_divide(
        df["net_income"],
        df["assets"],
    )

    calculated_roe = safe_divide(
        df["net_income"],
        df["equity"],
    )

    if "roa" not in df.columns:
        df["roa"] = calculated_roa
    else:
        df["roa"] = (
            df["roa"]
            .combine_first(
                calculated_roa
            )
        )

    if "roe" not in df.columns:
        df["roe"] = calculated_roe
    else:
        df["roe"] = (
            df["roe"]
            .combine_first(
                calculated_roe
            )
        )

    # ------------------------------------------------------------------
    # FINANCIAL STRENGTH
    # ------------------------------------------------------------------

    calculated_cash_assets = safe_divide(
        df["cash"],
        df["assets"],
    )

    calculated_debt_assets = safe_divide(
        df["total_debt"],
        df["assets"],
    )

    calculated_debt_equity = safe_divide(
        df["total_debt"],
        df["equity"],
    )

    # aliases anteriores, se existirem

    if "cash_assets" not in df.columns:
        df["cash_assets"] = np.nan

    if "debt_assets" not in df.columns:
        df["debt_assets"] = np.nan

    if "debt_equity" not in df.columns:
        df["debt_equity"] = np.nan

    if "cash_to_assets" in df.columns:

        df["cash_assets"] = (
            df["cash_assets"]
            .combine_first(
                df["cash_to_assets"]
            )
        )

    if "debt_to_assets" in df.columns:

        df["debt_assets"] = (
            df["debt_assets"]
            .combine_first(
                df["debt_to_assets"]
            )
        )

    if "debt_to_equity" in df.columns:

        df["debt_equity"] = (
            df["debt_equity"]
            .combine_first(
                df["debt_to_equity"]
            )
        )

    df["cash_assets"] = (
        df["cash_assets"]
        .combine_first(
            calculated_cash_assets
        )
    )

    df["debt_assets"] = (
        df["debt_assets"]
        .combine_first(
            calculated_debt_assets
        )
    )

    df["debt_equity"] = (
        df["debt_equity"]
        .combine_first(
            calculated_debt_equity
        )
    )

    # ------------------------------------------------------------------
    # GROWTH — aliases
    # ------------------------------------------------------------------

    if "revenue_growth" not in df.columns:
        df["revenue_growth"] = np.nan

    if "eps_growth" not in df.columns:
        df["eps_growth"] = np.nan

    if "operating_cash_flow_growth" not in df.columns:
        df["operating_cash_flow_growth"] = np.nan

    if "revenue_growth_yoy" in df.columns:

        df["revenue_growth"] = (
            df["revenue_growth"]
            .combine_first(
                df["revenue_growth_yoy"]
            )
        )

    if "diluted_eps_growth_yoy" in df.columns:

        df["eps_growth"] = (
            df["eps_growth"]
            .combine_first(
                df["diluted_eps_growth_yoy"]
            )
        )

    if "operating_cash_flow_growth_yoy" in df.columns:

        df["operating_cash_flow_growth"] = (
            df["operating_cash_flow_growth"]
            .combine_first(
                df[
                    "operating_cash_flow_growth_yoy"
                ]
            )
        )

    # ------------------------------------------------------------------
    # Valores infinitos não são elegíveis
    # ------------------------------------------------------------------

    metric_columns = [

        "cash_assets",
        "debt_assets",
        "debt_equity",

        "revenue_growth",
        "eps_growth",
        "operating_cash_flow_growth",

        "roa",
        "roe",
        "operating_margin",
        "net_margin",
    ]

    for col in metric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        df[col] = df[col].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return df


# ======================================================================================
# 4. SCORE GENÉRICO DE FATOR
# ======================================================================================

def calculate_factor_score(
    sector_df: pd.DataFrame,
    factor_name: str,
) -> pd.DataFrame:

    if factor_name not in FACTOR_DEFINITIONS:

        raise RuntimeError(
            f"Fator desconhecido: {factor_name}"
        )

    df = sector_df.copy()

    definition = FACTOR_DEFINITIONS[
        factor_name
    ]

    higher_metrics = definition[
        "higher"
    ]

    lower_metrics = definition[
        "lower"
    ]

    all_metrics = (
        higher_metrics
        +
        lower_metrics
    )

    # ------------------------------------------------------------------
    # Validar / criar métricas ausentes
    # ------------------------------------------------------------------

    for metric in all_metrics:

        if metric not in df.columns:
            df[metric] = np.nan

    # ------------------------------------------------------------------
    # Winsorização P5-P95
    #
    # A função recebe somente um setor.
    # Portanto a winsorização ocorre dentro do setor.
    # ------------------------------------------------------------------

    winsorized = pd.DataFrame(
        index=df.index
    )

    for metric in all_metrics:

        winsorized[
            metric
        ] = winsorize_series(
            df[metric]
        )

    # ------------------------------------------------------------------
    # Percentis
    # ------------------------------------------------------------------

    components = pd.DataFrame(
        index=df.index
    )

    for metric in higher_metrics:

        components[
            metric
        ] = percentile_score(
            winsorized[metric],
            lower_is_better=False,
        )

    for metric in lower_metrics:

        components[
            metric
        ] = percentile_score(
            winsorized[metric],
            lower_is_better=True,
        )

    # ------------------------------------------------------------------
    # Número de componentes disponíveis
    # ------------------------------------------------------------------

    components_column = (
        f"{factor_name}_components"
    )

    score_column = (
        f"{factor_name}_score"
    )

    df[
        components_column
    ] = (
        components
        .notna()
        .sum(axis=1)
    )

    # ------------------------------------------------------------------
    # SCORE DEFINITIVO
    #
    # Média dos percentis válidos.
    # NÃO mediana.
    # ------------------------------------------------------------------

    df[
        score_column
    ] = (
        components
        .mean(
            axis=1,
            skipna=True,
        )
    )

    # ------------------------------------------------------------------
    # Mínimo de componentes
    # ------------------------------------------------------------------

    minimum_components = int(
        definition[
            "minimum_components"
        ]
    )

    df.loc[
        df[
            components_column
        ]
        <
        minimum_components,
        score_column,
    ] = np.nan

    return df


# ======================================================================================
# 5. WRAPPERS DOS TRÊS FATORES
# ======================================================================================

def calculate_financial_strength_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    return calculate_factor_score(
        sector_df,
        "financial_strength",
    )


def calculate_growth_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    return calculate_factor_score(
        sector_df,
        "growth",
    )


def calculate_quality_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    return calculate_factor_score(
        sector_df,
        "quality",
    )


# ======================================================================================
# 6. CALCULAR SCORE POR SETOR
# ======================================================================================

def score_sector(
    sector_df: pd.DataFrame,
    sector: str,
) -> Tuple[pd.DataFrame, str]:

    factor = SELECTION_FACTORS[
        sector
    ]

    if factor == "financial_strength":

        scored = (
            calculate_financial_strength_score(
                sector_df
            )
        )

        score_column = (
            "financial_strength_score"
        )

    elif factor == "growth":

        scored = (
            calculate_growth_score(
                sector_df
            )
        )

        score_column = (
            "growth_score"
        )

    elif factor == "quality":

        scored = (
            calculate_quality_score(
                sector_df
            )
        )

        score_column = (
            "quality_score"
        )

    else:

        raise RuntimeError(
            f"Fator desconhecido: {factor}"
        )

    return (
        scored,
        score_column,
    )


# ======================================================================================
# 7. CARREGAR CARTEIRA ANTERIOR
# ======================================================================================

def load_previous_portfolio() -> pd.DataFrame:

    path = Path(
        CURRENT_PORTFOLIO_FILE
    )

    if not path.exists():

        return pd.DataFrame()

    try:

        df = pd.read_csv(
            path
        )

        if (
            "ticker"
            not in df.columns
            or
            "sector"
            not in df.columns
        ):

            return pd.DataFrame()

        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
            .str.replace(
                ".",
                "-",
                regex=False,
            )
        )

        return df

    except Exception:

        return pd.DataFrame()


# ======================================================================================
# 8. REGRA DE FRONTEIRA 5º VS 6º
# ======================================================================================

def apply_frontier_rule(
    ranked: pd.DataFrame,
    score_column: str,
    sector: str,
    previous_portfolio: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    target = SECTOR_TARGETS[
        sector
    ]

    ranked = (
        ranked
        .sort_values(
            score_column,
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    if len(ranked) <= target:

        return ranked.head(
            target
        )

    preliminary = (
        ranked
        .head(target)
        .copy()
    )

    if (
        previous_portfolio is None
        or
        previous_portfolio.empty
    ):

        return preliminary

    previous_sector = (
        previous_portfolio[
            previous_portfolio[
                "sector"
            ]
            ==
            sector
        ]
    )

    previous_tickers = set(
        previous_sector[
            "ticker"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if not previous_tickers:

        return preliminary

    rank5 = ranked.iloc[
        target - 1
    ]

    rank6 = ranked.iloc[
        target
    ]

    ticker5 = str(
        rank5["ticker"]
    )

    ticker6 = str(
        rank6["ticker"]
    )

    score5 = float(
        rank5[
            score_column
        ]
    )

    score6 = float(
        rank6[
            score_column
        ]
    )

    # ------------------------------------------------------------------
    # Proteção de fronteira:
    #
    # 5º = novo candidato
    # 6º = incumbente da carteira anterior
    # ------------------------------------------------------------------

    if (
        ticker5
        not in previous_tickers
        and
        ticker6
        in previous_tickers
    ):

        denominator = max(
            abs(score6),
            1e-9,
        )

        relative_gap = (
            score5
            -
            score6
        ) / denominator

        if (
            relative_gap
            <
            FRONTIER_MIN_RELATIVE_GAP
        ):

            preliminary = (
                ranked
                .head(
                    target - 1
                )
                .copy()
            )

            incumbent = (
                ranked[
                    ranked[
                        "ticker"
                    ]
                    ==
                    ticker6
                ]
                .head(1)
            )

            preliminary = pd.concat(
                [
                    preliminary,
                    incumbent,
                ],
                ignore_index=True,
            )

    return preliminary


# ======================================================================================
# 9. SELECIONAR TOP 5 DO SETOR
# ======================================================================================

def select_sector(
    universe: pd.DataFrame,
    sector: str,
    previous_portfolio: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    sector_df = (
        universe[
            universe[
                "sector"
            ]
            ==
            sector
        ]
        .copy()
    )

    if sector_df.empty:

        raise RuntimeError(
            f"Nenhuma empresa disponível em {sector}"
        )

    scored, score_column = (
        score_sector(
            sector_df,
            sector,
        )
    )

    scored = (
        scored[
            scored[
                score_column
            ]
            .notna()
        ]
        .copy()
    )

    target = SECTOR_TARGETS[
        sector
    ]

    if len(scored) < target:

        raise RuntimeError(
            f"{sector}: apenas {len(scored)} "
            f"empresas elegíveis para "
            f"{target} posições."
        )

    scored = (
        scored
        .sort_values(
            score_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    scored[
        "raw_rank"
    ] = np.arange(
        1,
        len(scored) + 1,
    )

    selected = apply_frontier_rule(
        ranked=scored,
        score_column=score_column,
        sector=sector,
        previous_portfolio=previous_portfolio,
    )

    selected = (
        selected
        .copy()
        .reset_index(drop=True)
    )

    selected[
        "selection_factor"
    ] = (
        SELECTION_FACTORS[
            sector
        ]
    )

    selected[
        "selection_score"
    ] = (
        selected[
            score_column
        ]
    )

    selected[
        "selection_rank"
    ] = np.arange(
        1,
        len(selected) + 1,
    )

    selected[
        "selected"
    ] = True

    return selected


# ======================================================================================
# 10. SELEÇÃO FINAL DAS 15 AÇÕES
# ======================================================================================

def select_portfolio(
    universe: pd.DataFrame,
    use_previous_portfolio: bool = True,
) -> pd.DataFrame:

    required_columns = {
        "ticker",
        "sector",
    }

    if not required_columns.issubset(
        universe.columns
    ):

        raise ValueError(
            "Universo precisa conter ticker e sector."
        )

    universe = (
        build_derived_metrics(
            universe
        )
    )

    previous = (
        load_previous_portfolio()
        if use_previous_portfolio
        else pd.DataFrame()
    )

    selected_parts = []

    for sector in SECTORS:

        selected_sector = (
            select_sector(
                universe=universe,
                sector=sector,
                previous_portfolio=previous,
            )
        )

        selected_parts.append(
            selected_sector
        )

    portfolio = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # AUDITORIA — exatamente 15 tickers
    # ------------------------------------------------------------------

    if (
        portfolio[
            "ticker"
        ]
        .nunique()
        !=
        15
    ):

        raise RuntimeError(
            "A carteira final não possui exatamente 15 tickers."
        )

    # ------------------------------------------------------------------
    # AUDITORIA — exatamente 5 por setor
    # ------------------------------------------------------------------

    sector_counts = (
        portfolio
        .groupby(
            "sector"
        )[
            "ticker"
        ]
        .nunique()
    )

    for sector in SECTORS:

        expected = (
            SECTOR_TARGETS[
                sector
            ]
        )

        actual = int(
            sector_counts.get(
                sector,
                0,
            )
        )

        if actual != expected:

            raise RuntimeError(
                f"{sector}: {actual} ações. "
                f"Esperado: {expected}."
            )

    portfolio = (
        portfolio
        .sort_values(
            [
                "sector",
                "selection_rank",
            ]
        )
        .reset_index(drop=True)
    )

    return portfolio


# ======================================================================================
# 11. AUDITORIA DA FRONTEIRA
# ======================================================================================

def build_frontier_audit(
    universe: pd.DataFrame,
) -> pd.DataFrame:

    universe = (
        build_derived_metrics(
            universe
        )
    )

    rows = []

    for sector in SECTORS:

        sector_df = (
            universe[
                universe[
                    "sector"
                ]
                ==
                sector
            ]
            .copy()
        )

        scored, score_column = (
            score_sector(
                sector_df,
                sector,
            )
        )

        scored = (
            scored[
                scored[
                    score_column
                ]
                .notna()
            ]
            .sort_values(
                score_column,
                ascending=False,
            )
            .reset_index(drop=True)
        )

        if len(scored) < 6:
            continue

        rank5 = scored.iloc[4]
        rank6 = scored.iloc[5]

        score5 = float(
            rank5[
                score_column
            ]
        )

        score6 = float(
            rank6[
                score_column
            ]
        )

        relative_gap = (
            (
                score5
                -
                score6
            )
            /
            max(
                abs(score6),
                1e-9,
            )
        )

        rows.append(
            {
                "sector":
                    sector,

                "factor":
                    SELECTION_FACTORS[
                        sector
                    ],

                "rank5_ticker":
                    rank5[
                        "ticker"
                    ],

                "rank5_score":
                    score5,

                "rank6_ticker":
                    rank6[
                        "ticker"
                    ],

                "rank6_score":
                    score6,

                "relative_gap":
                    relative_gap,

                "frontier_material":
                    (
                        relative_gap
                        >=
                        FRONTIER_MIN_RELATIVE_GAP
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 12. TESTE DO MÓDULO
# ======================================================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO ACOES AMERICANO — SELECTION ENGINE"
    )

    print(
        "=" * 100
    )

    print(
        "\nMetodologia:"
    )

    print(
        "  Winsorização P5-P95"
    )

    print(
        "  Percentis dentro do setor"
    )

    print(
        "  Score = média dos componentes"
    )

    print(
        "\nFatores congelados:"
    )

    for sector in SECTORS:

        print(
            f"  {sector:<28} -> "
            f"{SELECTION_FACTORS[sector]}"
        )

    print(
        "\nEstrutura:"
    )

    print(
        "  3 setores fixos"
    )

    print(
        "  5 ações por setor"
    )

    print(
        "  15 ações totais"
    )

    print(
        "  tickers dinâmicos"
    )

    print(
        "\nSelection engine carregado com sucesso."
    )
