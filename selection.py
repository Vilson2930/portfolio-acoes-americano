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
#   -> Financial Strength
#      cash_assets ↑
#      debt_assets ↓
#      debt_equity ↓
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
#   6) ranking
#   7) Top 5 por setor
#   8) proteção de fronteira 5º vs 6º
#
# ======================================================================================

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd

from config import (
    SECTORS,
    SECTOR_TARGETS,
    SELECTION_FACTORS as CONFIG_SELECTION_FACTORS,
    CURRENT_PORTFOLIO_FILE,
)


# ======================================================================================
# REGRA VENCEDORA DO ESTUDO — CÉLULAS 15, 16 E 17
# ======================================================================================
#
# A seleção operacional NÃO depende mais de uma configuração divergente em config.py.
# Esta regra é congelada aqui porque foi a estratégia setorial historicamente validada:
#
#   Health Care              -> Financial Strength
#   Industrials              -> Growth
#   Information Technology   -> Financial Strength
#
# Quality permanece implementado apenas para compatibilidade/auditoria, mas NÃO é fator
# vencedor de Information Technology.
# ======================================================================================

STUDY_SELECTION_FACTORS = {
    "Health Care": "financial_strength",
    "Industrials": "growth",
    "Information Technology": "financial_strength",
}

# Nome usado pelo restante deste módulo.
SELECTION_FACTORS = STUDY_SELECTION_FACTORS.copy()


def validate_selection_factor_alignment() -> None:
    """
    Detecta drift entre config.py e a regra vencedora do estudo.

    Não permite que uma alteração futura em config.py mude silenciosamente a
    metodologia validada das Células 15/16/17.
    """

    config_map = dict(CONFIG_SELECTION_FACTORS)

    if config_map != STUDY_SELECTION_FACTORS:
        print(
            "ATENÇÃO: config.py diverge da regra vencedora do estudo. "
            "selection.py usará STUDY_SELECTION_FACTORS = "
            f"{STUDY_SELECTION_FACTORS}. Config recebido = {config_map}."
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
# 8. FRONTEIRA FINAL — CÉLULA 31
# ======================================================================================
#
# A Célula 31 NÃO usa uma simples regra de persistência da carteira anterior.
#
# Regra fiel:
#   • Top 1–4 de cada setor permanecem.
#   • Só 5º x 6º podem ser comparados.
#   • O 6º só pode substituir o 5º se houver "near tie":
#         gap absoluto <= 0.01 OU gap relativo <= 1%.
#   • Mesmo com near tie, a troca exige >= 3 melhorias de risco.
#   • A troca é bloqueada se houver deterioração relevante.
#
# IMPORTANTE:
# Os fatores usados aqui continuam sendo os vencedores já validados:
#   Health Care              -> Financial Strength
#   Industrials              -> Growth
#   Information Technology   -> Financial Strength
#
# ======================================================================================

MAX_ABSOLUTE_FACTOR_GAP = 0.01
MAX_RELATIVE_FACTOR_GAP = 0.01

MIN_RISK_IMPROVEMENTS = 3

VOL_IMPROVEMENT = 0.0025
SHARPE_IMPROVEMENT = 0.02
DRAWDOWN_IMPROVEMENT = 0.005
CORRELATION_IMPROVEMENT = 0.01
RISK_CONTRIBUTION_IMPROVEMENT = 0.01

MAX_VOL_DETERIORATION = 0.005
MAX_SHARPE_DETERIORATION = 0.02
MAX_DRAWDOWN_DETERIORATION = 0.01

MIN_COMMON_DAYS = 120
BOUNDARY_RISK_START = pd.Timestamp("2024-01-01")
ANNUALIZATION_DAILY = 252


def _rank_all_sectors(
    universe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ranking fundamental completo dos três setores.
    Não aplica nenhuma regra de fronteira.
    """

    base = build_derived_metrics(universe)

    parts = []

    for sector in SECTORS:

        sector_df = (
            base[
                base["sector"] == sector
            ]
            .copy()
        )

        scored, score_column = score_sector(
            sector_df,
            sector,
        )

        scored = (
            scored[
                scored[score_column].notna()
            ]
            .sort_values(
                [score_column, "ticker"],
                ascending=[False, True],
            )
            .reset_index(drop=True)
        )

        target = int(
            SECTOR_TARGETS[sector]
        )

        if len(scored) < target + 1:
            raise RuntimeError(
                f"{sector}: são necessárias pelo menos "
                f"{target + 1} empresas elegíveis para auditar 5º vs 6º."
            )

        scored["sector_rank"] = np.arange(
            1,
            len(scored) + 1,
        )

        scored["selection_factor"] = (
            SELECTION_FACTORS[sector]
        )

        scored["selection_score"] = (
            scored[score_column]
        )

        parts.append(scored)

    return pd.concat(
        parts,
        ignore_index=True,
    )


def _build_raw_top5(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Carteira fundamental pura antes do teste de risco da Célula 31.
    """

    parts = []

    for sector in SECTORS:

        target = int(
            SECTOR_TARGETS[sector]
        )

        part = (
            ranking[
                ranking["sector"] == sector
            ]
            .sort_values("sector_rank")
            .head(target)
            .copy()
        )

        parts.append(part)

    portfolio = pd.concat(
        parts,
        ignore_index=True,
    )

    portfolio["selection_rank"] = (
        portfolio["sector_rank"]
        .astype(int)
    )

    portfolio["selected"] = True

    return portfolio


def _normalize_prices(
    prices: pd.DataFrame,
) -> pd.DataFrame:

    if prices is None or prices.empty:
        return pd.DataFrame()

    close = prices.copy()

    close.index = pd.to_datetime(
        close.index
    )

    close.columns = [
        str(c)
        .strip()
        .upper()
        .replace(".", "-")
        for c in close.columns
    ]

    close = close.loc[
        close.index >= BOUNDARY_RISK_START
    ]

    return close.sort_index()


def _portfolio_risk_metrics(
    ticker_list,
    daily_returns: pd.DataFrame,
):

    missing = [
        ticker
        for ticker in ticker_list
        if ticker not in daily_returns.columns
    ]

    if missing:
        return None

    common = (
        daily_returns[
            ticker_list
        ]
        .dropna()
    )

    if len(common) < MIN_COMMON_DAYS:
        return None

    n = len(ticker_list)

    weights = np.repeat(
        1 / n,
        n,
    )

    portfolio_return = (
        common
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
    )

    daily_std = portfolio_return.std(
        ddof=1
    )

    portfolio_vol = (
        daily_std
        *
        np.sqrt(
            ANNUALIZATION_DAILY
        )
    )

    sharpe = (
        portfolio_return.mean()
        /
        daily_std
        *
        np.sqrt(
            ANNUALIZATION_DAILY
        )
        if daily_std > 0
        else np.nan
    )

    wealth = (
        1
        +
        portfolio_return
    ).cumprod()

    drawdown = (
        wealth
        /
        wealth.cummax()
        -
        1
    )

    max_drawdown = drawdown.min()

    corr = common.corr()

    corr_values = []

    for a, b in combinations(
        ticker_list,
        2,
    ):

        value = corr.loc[
            a,
            b,
        ]

        if pd.notna(value):
            corr_values.append(value)

    mean_correlation = (
        np.mean(corr_values)
        if corr_values
        else np.nan
    )

    covariance = (
        common.cov().values
        *
        ANNUALIZATION_DAILY
    )

    variance = (
        weights
        @
        covariance
        @
        weights
    )

    if (
        not np.isfinite(variance)
        or
        variance <= 0
    ):
        return None

    vol = np.sqrt(variance)

    marginal = (
        covariance
        @
        weights
        /
        vol
    )

    component = (
        weights
        *
        marginal
    )

    risk_contribution = (
        component
        /
        vol
    )

    max_risk_contribution = np.max(
        risk_contribution
    )

    return {
        "observations":
            len(common),

        "start_date":
            common.index.min(),

        "end_date":
            common.index.max(),

        "volatility":
            portfolio_vol,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "mean_correlation":
            mean_correlation,

        "max_risk_contribution":
            max_risk_contribution,
    }


def _boundary_rows(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for sector in SECTORS:

        sector_ranking = (
            ranking[
                ranking["sector"] == sector
            ]
            .sort_values("sector_rank")
        )

        rank5 = (
            sector_ranking[
                sector_ranking["sector_rank"] == 5
            ]
            .iloc[0]
        )

        rank6 = (
            sector_ranking[
                sector_ranking["sector_rank"] == 6
            ]
            .iloc[0]
        )

        score5 = float(
            rank5["selection_score"]
        )

        score6 = float(
            rank6["selection_score"]
        )

        absolute_gap = (
            score5
            -
            score6
        )

        relative_gap = (
            absolute_gap
            /
            abs(score5)
            if score5 != 0
            else np.nan
        )

        near_tie = bool(
            (
                absolute_gap
                <=
                MAX_ABSOLUTE_FACTOR_GAP
            )
            or
            (
                pd.notna(relative_gap)
                and
                relative_gap
                <=
                MAX_RELATIVE_FACTOR_GAP
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
                    rank5["ticker"],

                "rank5_score":
                    score5,

                "rank6_ticker":
                    rank6["ticker"],

                "rank6_score":
                    score6,

                "absolute_gap":
                    absolute_gap,

                "relative_gap":
                    relative_gap,

                "near_tie":
                    near_tie,
            }
        )

    return pd.DataFrame(rows)


def get_boundary_test_tickers(
    universe: pd.DataFrame,
) -> list[str]:
    """
    Retorna os 15 Top-5 + os três candidatos de 6º lugar.
    Útil para o main.py baixar preços ANTES de fechar a carteira.
    """

    ranking = _rank_all_sectors(
        universe
    )

    raw_portfolio = _build_raw_top5(
        ranking
    )

    boundary = _boundary_rows(
        ranking
    )

    return sorted(
        set(
            raw_portfolio[
                "ticker"
            ]
            .tolist()
            +
            boundary[
                "rank6_ticker"
            ]
            .tolist()
        )
    )


def _run_boundary_test(
    ranking: pd.DataFrame,
    raw_portfolio: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduz a lógica decisória da Célula 31.
    Cada candidato 6º é comparado contra a MESMA carteira Top-5 original.
    """

    boundary = _boundary_rows(
        ranking
    )

    close = _normalize_prices(
        prices
    )

    if close.empty:

        boundary["test_valid"] = False
        boundary["decision"] = (
            "KEEP — PREÇOS AUSENTES"
        )

        return boundary

    returns = (
        close
        .pct_change(
            fill_method=None
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    current_tickers = (
        raw_portfolio[
            "ticker"
        ]
        .tolist()
    )

    rows = []

    for row in boundary.itertuples(
        index=False
    ):

        incumbent = row.rank5_ticker
        challenger = row.rank6_ticker

        comparison_tickers = sorted(
            set(
                current_tickers
                +
                [challenger]
            )
        )

        missing = [
            ticker
            for ticker in comparison_tickers
            if ticker not in returns.columns
        ]

        if missing:

            rows.append(
                {
                    **row._asdict(),
                    "test_valid":
                        False,
                    "decision":
                        "KEEP — PREÇOS INSUFICIENTES",
                }
            )

            continue

        common_dates = (
            returns[
                comparison_tickers
            ]
            .dropna()
            .index
        )

        if len(common_dates) < MIN_COMMON_DAYS:

            rows.append(
                {
                    **row._asdict(),
                    "test_valid":
                        False,
                    "observations":
                        len(common_dates),
                    "decision":
                        "KEEP — HISTÓRICO INSUFICIENTE",
                }
            )

            continue

        comparison_returns = (
            returns.loc[
                common_dates
            ]
        )

        original_metrics = (
            _portfolio_risk_metrics(
                current_tickers,
                comparison_returns,
            )
        )

        candidate_portfolio = [
            challenger
            if ticker == incumbent
            else ticker
            for ticker in current_tickers
        ]

        candidate_metrics = (
            _portfolio_risk_metrics(
                candidate_portfolio,
                comparison_returns,
            )
        )

        if (
            original_metrics is None
            or
            candidate_metrics is None
        ):

            rows.append(
                {
                    **row._asdict(),
                    "test_valid":
                        False,
                    "decision":
                        "KEEP — TESTE INVÁLIDO",
                }
            )

            continue

        delta_volatility = (
            candidate_metrics[
                "volatility"
            ]
            -
            original_metrics[
                "volatility"
            ]
        )

        delta_sharpe = (
            candidate_metrics[
                "sharpe"
            ]
            -
            original_metrics[
                "sharpe"
            ]
        )

        delta_drawdown = (
            candidate_metrics[
                "max_drawdown"
            ]
            -
            original_metrics[
                "max_drawdown"
            ]
        )

        delta_mean_corr = (
            candidate_metrics[
                "mean_correlation"
            ]
            -
            original_metrics[
                "mean_correlation"
            ]
        )

        delta_max_risk = (
            candidate_metrics[
                "max_risk_contribution"
            ]
            -
            original_metrics[
                "max_risk_contribution"
            ]
        )

        improve_vol = (
            delta_volatility
            <=
            -VOL_IMPROVEMENT
        )

        improve_sharpe = (
            delta_sharpe
            >=
            SHARPE_IMPROVEMENT
        )

        improve_drawdown = (
            delta_drawdown
            >=
            DRAWDOWN_IMPROVEMENT
        )

        improve_corr = (
            delta_mean_corr
            <=
            -CORRELATION_IMPROVEMENT
        )

        improve_risk_concentration = (
            delta_max_risk
            <=
            -RISK_CONTRIBUTION_IMPROVEMENT
        )

        improvement_count = int(
            sum(
                [
                    improve_vol,
                    improve_sharpe,
                    improve_drawdown,
                    improve_corr,
                    improve_risk_concentration,
                ]
            )
        )

        severe_deterioration = bool(
            (
                delta_volatility
                >
                MAX_VOL_DETERIORATION
            )
            or
            (
                delta_sharpe
                <
                -MAX_SHARPE_DETERIORATION
            )
            or
            (
                delta_drawdown
                <
                -MAX_DRAWDOWN_DETERIORATION
            )
        )

        if not row.near_tie:

            decision = (
                "KEEP — VANTAGEM FUNDAMENTAL DO 5º"
            )

        elif severe_deterioration:

            decision = (
                "KEEP — TROCA PIORA A CARTEIRA"
            )

        elif (
            improvement_count
            >=
            MIN_RISK_IMPROVEMENTS
        ):

            decision = (
                "SWAP — 6º MELHORA RISCO COM SCORE EQUIVALENTE"
            )

        else:

            decision = (
                "KEEP — EVIDÊNCIA INSUFICIENTE PARA TROCA"
            )

        rows.append(
            {
                **row._asdict(),

                "observations":
                    original_metrics[
                        "observations"
                    ],

                "baseline_vol":
                    original_metrics[
                        "volatility"
                    ],

                "candidate_vol":
                    candidate_metrics[
                        "volatility"
                    ],

                "delta_vol":
                    delta_volatility,

                "baseline_sharpe":
                    original_metrics[
                        "sharpe"
                    ],

                "candidate_sharpe":
                    candidate_metrics[
                        "sharpe"
                    ],

                "delta_sharpe":
                    delta_sharpe,

                "baseline_max_dd":
                    original_metrics[
                        "max_drawdown"
                    ],

                "candidate_max_dd":
                    candidate_metrics[
                        "max_drawdown"
                    ],

                "delta_max_dd":
                    delta_drawdown,

                "baseline_mean_corr":
                    original_metrics[
                        "mean_correlation"
                    ],

                "candidate_mean_corr":
                    candidate_metrics[
                        "mean_correlation"
                    ],

                "delta_mean_corr":
                    delta_mean_corr,

                "baseline_max_risk":
                    original_metrics[
                        "max_risk_contribution"
                    ],

                "candidate_max_risk":
                    candidate_metrics[
                        "max_risk_contribution"
                    ],

                "delta_max_risk":
                    delta_max_risk,

                "risk_improvements":
                    improvement_count,

                "severe_deterioration":
                    severe_deterioration,

                "test_valid":
                    True,

                "decision":
                    decision,
            }
        )

    return pd.DataFrame(rows)


def _apply_boundary_decisions(
    ranking: pd.DataFrame,
    raw_portfolio: pd.DataFrame,
    boundary_test: pd.DataFrame,
) -> pd.DataFrame:

    final_portfolio = (
        raw_portfolio
        .copy()
    )

    if boundary_test.empty:
        return final_portfolio

    for row in boundary_test.itertuples(
        index=False
    ):

        decision = str(
            getattr(
                row,
                "decision",
                "",
            )
        )

        if not decision.startswith(
            "SWAP"
        ):
            continue

        sector = row.sector
        incumbent = row.rank5_ticker
        challenger = row.rank6_ticker

        candidate = (
            ranking[
                (
                    ranking["sector"]
                    ==
                    sector
                )
                &
                (
                    ranking["ticker"]
                    ==
                    challenger
                )
            ]
            .head(1)
            .copy()
        )

        if candidate.empty:
            continue

        final_portfolio = (
            final_portfolio[
                final_portfolio[
                    "ticker"
                ]
                !=
                incumbent
            ]
            .copy()
        )

        final_portfolio = pd.concat(
            [
                final_portfolio,
                candidate,
            ],
            ignore_index=True,
        )

    final_portfolio = (
        final_portfolio
        .sort_values(
            [
                "sector",
                "selection_score",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    final_portfolio[
        "selection_rank"
    ] = (
        final_portfolio
        .groupby("sector")
        .cumcount()
        +
        1
    )

    final_portfolio[
        "selected"
    ] = True

    return final_portfolio


# ======================================================================================
# 9. SELEÇÃO FINAL DAS 15 AÇÕES
# ======================================================================================

def select_portfolio(
    universe: pd.DataFrame,
    use_previous_portfolio: bool = True,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Seleção final fiel ao estudo.

    `use_previous_portfolio` é mantido na assinatura por compatibilidade,
    mas NÃO controla mais a fronteira. A Célula 31 não usa persistência
    automática do incumbente; usa o teste 5º x 6º baseado em risco.

    Se `prices` for None:
        retorna o Top 5 fundamental puro (pré-fronteira).

    Se `prices` for informado:
        executa o teste completo da Célula 31.
    """

    validate_selection_factor_alignment()

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

    ranking = _rank_all_sectors(
        universe
    )

    raw_portfolio = _build_raw_top5(
        ranking
    )

    if prices is None:

        portfolio = raw_portfolio

    else:

        boundary_test = (
            _run_boundary_test(
                ranking=ranking,
                raw_portfolio=raw_portfolio,
                prices=prices,
            )
        )

        portfolio = (
            _apply_boundary_decisions(
                ranking=ranking,
                raw_portfolio=raw_portfolio,
                boundary_test=boundary_test,
            )
        )

    if (
        portfolio["ticker"]
        .nunique()
        !=
        15
    ):

        raise RuntimeError(
            "A carteira final não possui exatamente 15 tickers."
        )

    sector_counts = (
        portfolio
        .groupby("sector")[
            "ticker"
        ]
        .nunique()
    )

    for sector in SECTORS:

        expected = int(
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

    return (
        portfolio
        .sort_values(
            [
                "sector",
                "selection_rank",
            ]
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 10. AUDITORIA DA FRONTEIRA
# ======================================================================================

def build_frontier_audit(
    universe: pd.DataFrame,
    prices: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    ranking = _rank_all_sectors(
        universe
    )

    raw_portfolio = _build_raw_top5(
        ranking
    )

    if prices is None:
        return _boundary_rows(
            ranking
        )

    return _run_boundary_test(
        ranking=ranking,
        raw_portfolio=raw_portfolio,
        prices=prices,
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
        "\nFatores vencedores congelados (Células 15/16/17):"
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
