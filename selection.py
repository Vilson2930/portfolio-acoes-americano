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
# usando os fatores definidos no estudo:
#
#   Health Care              -> Financial Strength
#   Industrials              -> Growth
#   Information Technology   -> Quality
#
# IMPORTANTE
# ----------
# Os setores e a estrutura 5/5/5 são FIXOS.
# Os tickers são DINÂMICOS.
#
# A seleção utiliza score por setor e aplica proteção de fronteira
# 5º vs 6º para evitar troca excessiva.
#
# ======================================================================================


from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

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
# 1. HELPERS
# ======================================================================================

def safe_divide(
    numerator,
    denominator,
):
    """
    Divisão segura.
    """

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

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


def percentile_score(
    series: pd.Series,
    ascending: bool = True,
) -> pd.Series:
    """
    Converte uma métrica em score percentual 0-1.

    ascending=True:
        valor maior = score maior

    ascending=False:
        valor menor = score maior
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    return values.rank(
        pct=True,
        ascending=ascending,
        method="average",
    )


# ======================================================================================
# 2. PREPARAR MÉTRICAS DERIVADAS
# ======================================================================================

def build_derived_metrics(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:

    df = fundamentals.copy()


    numeric_columns = [
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "capex",
        "assets",
        "equity",
        "cash",
        "long_term_debt",
        "diluted_eps",
        "diluted_shares",
    ]


    for col in numeric_columns:

        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


    # ------------------------------------------------------------------
    # Free Cash Flow
    # ------------------------------------------------------------------

    df["free_cash_flow"] = (
        df["operating_cash_flow"]
        -
        df["capex"].abs()
    )


    # ------------------------------------------------------------------
    # Margens
    # ------------------------------------------------------------------

    df["net_margin"] = safe_divide(
        df["net_income"],
        df["revenue"],
    )


    df["operating_margin"] = safe_divide(
        df["operating_income"],
        df["revenue"],
    )


    df["ocf_margin"] = safe_divide(
        df["operating_cash_flow"],
        df["revenue"],
    )


    df["fcf_margin"] = safe_divide(
        df["free_cash_flow"],
        df["revenue"],
    )


    # ------------------------------------------------------------------
    # Retornos sobre capital
    # ------------------------------------------------------------------

    df["roe"] = safe_divide(
        df["net_income"],
        df["equity"],
    )


    df["roa"] = safe_divide(
        df["net_income"],
        df["assets"],
    )


    # ------------------------------------------------------------------
    # Dívida
    # ------------------------------------------------------------------

    df["debt_to_assets"] = safe_divide(
        df["long_term_debt"],
        df["assets"],
    )


    df["debt_to_equity"] = safe_divide(
        df["long_term_debt"],
        df["equity"],
    )


    # ------------------------------------------------------------------
    # Caixa relativo
    # ------------------------------------------------------------------

    df["cash_to_assets"] = safe_divide(
        df["cash"],
        df["assets"],
    )


    return df


# ======================================================================================
# 3. GROWTH HISTÓRICO
# ======================================================================================
#
# A seleção diária precisa de crescimento.
#
# Para isso, o ideal é receber uma base com pelo menos dois snapshots
# por empresa:
#
#   snapshot atual
#   snapshot anterior comparável
#
# Caso as colunas growth_* já existam, usamos diretamente.
#
# ======================================================================================

def ensure_growth_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()


    growth_columns = [
        "revenue_growth_yoy",
        "net_income_growth_yoy",
        "operating_cash_flow_growth_yoy",
        "diluted_eps_growth_yoy",
    ]


    for col in growth_columns:

        if col not in result.columns:
            result[col] = np.nan


    return result


# ======================================================================================
# 4. FINANCIAL STRENGTH SCORE — HEALTH CARE
# ======================================================================================
#
# Estrutura:
#
#   ROE
#   ROA
#   Margem operacional
#   Margem líquida
#   OCF margin
#   FCF margin
#   Caixa / ativos
#   Dívida / ativos (menor é melhor)
#
# Score final = mediana dos componentes válidos.
#
# ======================================================================================

def calculate_financial_strength_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    df = sector_df.copy()


    components = pd.DataFrame(
        index=df.index
    )


    components["roe"] = percentile_score(
        df["roe"],
        ascending=True,
    )


    components["roa"] = percentile_score(
        df["roa"],
        ascending=True,
    )


    components["operating_margin"] = percentile_score(
        df["operating_margin"],
        ascending=True,
    )


    components["net_margin"] = percentile_score(
        df["net_margin"],
        ascending=True,
    )


    components["ocf_margin"] = percentile_score(
        df["ocf_margin"],
        ascending=True,
    )


    components["fcf_margin"] = percentile_score(
        df["fcf_margin"],
        ascending=True,
    )


    components["cash_to_assets"] = percentile_score(
        df["cash_to_assets"],
        ascending=True,
    )


    components["debt_to_assets"] = percentile_score(
        df["debt_to_assets"],
        ascending=False,
    )


    df["financial_strength_components"] = (
        components
        .notna()
        .sum(axis=1)
    )


    df["financial_strength_score"] = (
        components
        .median(
            axis=1,
            skipna=True,
        )
    )


    df.loc[
        df["financial_strength_components"] < 4,
        "financial_strength_score"
    ] = np.nan


    return df


# ======================================================================================
# 5. GROWTH SCORE — INDUSTRIALS
# ======================================================================================
#
# Componentes:
#
#   Revenue Growth
#   Net Income Growth
#   Operating Cash Flow Growth
#   EPS Growth
#   Margem operacional
#
# ======================================================================================

def calculate_growth_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    df = sector_df.copy()


    df = ensure_growth_metrics(
        df
    )


    components = pd.DataFrame(
        index=df.index
    )


    components["revenue_growth"] = percentile_score(
        df["revenue_growth_yoy"],
        ascending=True,
    )


    components["net_income_growth"] = percentile_score(
        df["net_income_growth_yoy"],
        ascending=True,
    )


    components["ocf_growth"] = percentile_score(
        df["operating_cash_flow_growth_yoy"],
        ascending=True,
    )


    components["eps_growth"] = percentile_score(
        df["diluted_eps_growth_yoy"],
        ascending=True,
    )


    components["operating_margin"] = percentile_score(
        df["operating_margin"],
        ascending=True,
    )


    df["growth_components"] = (
        components
        .notna()
        .sum(axis=1)
    )


    df["growth_score"] = (
        components
        .median(
            axis=1,
            skipna=True,
        )
    )


    df.loc[
        df["growth_components"] < 3,
        "growth_score"
    ] = np.nan


    return df


# ======================================================================================
# 6. QUALITY SCORE — INFORMATION TECHNOLOGY
# ======================================================================================
#
# Componentes:
#
#   ROE
#   ROA
#   Margem operacional
#   Margem líquida
#   OCF margin
#   FCF margin
#   Dívida / ativos (menor melhor)
#
# ======================================================================================

def calculate_quality_score(
    sector_df: pd.DataFrame,
) -> pd.DataFrame:

    df = sector_df.copy()


    components = pd.DataFrame(
        index=df.index
    )


    components["roe"] = percentile_score(
        df["roe"],
        ascending=True,
    )


    components["roa"] = percentile_score(
        df["roa"],
        ascending=True,
    )


    components["operating_margin"] = percentile_score(
        df["operating_margin"],
        ascending=True,
    )


    components["net_margin"] = percentile_score(
        df["net_margin"],
        ascending=True,
    )


    components["ocf_margin"] = percentile_score(
        df["ocf_margin"],
        ascending=True,
    )


    components["fcf_margin"] = percentile_score(
        df["fcf_margin"],
        ascending=True,
    )


    components["debt_to_assets"] = percentile_score(
        df["debt_to_assets"],
        ascending=False,
    )


    df["quality_components"] = (
        components
        .notna()
        .sum(axis=1)
    )


    df["quality_score"] = (
        components
        .median(
            axis=1,
            skipna=True,
        )
    )


    df.loc[
        df["quality_components"] < 4,
        "quality_score"
    ] = np.nan


    return df


# ======================================================================================
# 7. CALCULAR SCORE POR SETOR
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
# 8. CARREGAR CARTEIRA ANTERIOR
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
        )


        return df


    except Exception:

        return pd.DataFrame()


# ======================================================================================
# 9. REGRA DE FRONTEIRA 5º VS 6º
# ======================================================================================
#
# Se houver carteira anterior:
#
#   • a 6ª colocada não substitui automaticamente a 5ª;
#   • só troca se a vantagem relativa for >= FRONTIER_MIN_RELATIVE_GAP.
#
# Isso reduz turnover.
#
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
    # Só existe questão de fronteira se:
    #
    #   • 5º é novo
    #   • 6º fazia parte da carteira anterior
    #
    # ------------------------------------------------------------------

    if (
        ticker5
        not in
        previous_tickers

        and

        ticker6
        in
        previous_tickers
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

            # manter incumbente
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
# 10. SELECIONAR TOP 5 DO SETOR
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


    scored = scored[
        scored[
            score_column
        ]
        .notna()
    ].copy()


    if (
        len(scored)
        <
        SECTOR_TARGETS[
            sector
        ]
    ):

        raise RuntimeError(
            f"{sector}: apenas {len(scored)} "
            f"empresas elegíveis para "
            f"{SECTOR_TARGETS[sector]} posições."
        )


    scored = (
        scored
        .sort_values(
            score_column,
            ascending=False,
        )
        .reset_index(drop=True)
    )


    scored["raw_rank"] = (
        np.arange(
            1,
            len(scored) + 1,
        )
    )


    selected = apply_frontier_rule(
        ranked=scored,
        score_column=score_column,
        sector=sector,
        previous_portfolio=previous_portfolio,
    )


    selected = selected.copy()


    selected["selection_factor"] = (
        SELECTION_FACTORS[
            sector
        ]
    )


    selected["selection_score"] = (
        selected[
            score_column
        ]
    )


    selected["selection_rank"] = (
        np.arange(
            1,
            len(selected) + 1,
        )
    )


    selected["selected"] = True


    return selected


# ======================================================================================
# 11. SELEÇÃO FINAL 15 AÇÕES
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


    # ------------------------------------------------------------------
    # Métricas derivadas
    # ------------------------------------------------------------------

    universe = build_derived_metrics(
        universe
    )


    universe = ensure_growth_metrics(
        universe
    )


    # ------------------------------------------------------------------
    # Carteira anterior
    # ------------------------------------------------------------------

    previous = (
        load_previous_portfolio()
        if use_previous_portfolio
        else pd.DataFrame()
    )


    selected_parts = []


    for sector in SECTORS:

        selected_sector = select_sector(

            universe=universe,

            sector=sector,

            previous_portfolio=previous,
        )


        selected_parts.append(
            selected_sector
        )


    portfolio = pd.concat(
        selected_parts,
        ignore_index=True,
    )


    # ------------------------------------------------------------------
    # Auditoria estrutural
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


    # ------------------------------------------------------------------
    # Ordem final
    # ------------------------------------------------------------------

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
# 12. AUDITORIA DA FRONTEIRA
# ======================================================================================

def build_frontier_audit(
    universe: pd.DataFrame,
) -> pd.DataFrame:

    universe = build_derived_metrics(
        universe
    )


    universe = ensure_growth_metrics(
        universe
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
# 13. TESTE DO MÓDULO
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
        "  tickers dinâmicos"
    )

    print(
        "\nSelection engine carregado com sucesso."
    )
