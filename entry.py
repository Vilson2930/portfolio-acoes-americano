# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# entry.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Classificar o momento de entrada das 15 ações JÁ SELECIONADAS.
#
# Este módulo NÃO escolhe empresas e NÃO altera a carteira.
#
# ARQUITETURA VALIDADA HISTORICAMENTE
# -----------------------------------
#
# Health Care
#   10% Valuation
#   80% Desconto
#   10% Fundamentos
#
# Industrials
#   20% Desconto
#   80% Fundamentos
#   Regra CONDICIONAL
#   Nunca recebe ENTRADA FORTE
#
# Information Technology
#   Momentum 6M + 12M
#
# CLASSIFICAÇÃO FINAL
# -------------------
#
#   >= 75º percentil  -> ENTRADA FORTE
#   >= 50º percentil  -> ENTRADA
#   >= 25º percentil  -> AGUARDAR
#   <  25º percentil  -> NÃO COMPRAR AGORA
#
# Industrials:
#   ENTRADA FORTE é rebaixada para ENTRADA.
#
# ======================================================================================

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from config import SECTORS


# ======================================================================================
# 1. CONSTANTES
# ======================================================================================

HEALTH_CARE = "Health Care"
INDUSTRIALS = "Industrials"
TECHNOLOGY = "Information Technology"


ENTRY_STRONG_PERCENTILE = 0.75
ENTRY_PERCENTILE = 0.50
WAIT_PERCENTILE = 0.25


ENTRY_METHODS = {

    HEALTH_CARE:
        "10% Valuation + 80% Desconto + 10% Fundamentos",

    INDUSTRIALS:
        "20% Desconto + 80% Fundamentos",

    TECHNOLOGY:
        "Momentum 6M + 12M",
}


SECTOR_VALIDATION_STATUS = {

    HEALTH_CARE:
        "APROVADO",

    INDUSTRIALS:
        "CONDICIONAL",

    TECHNOLOGY:
        "REGRA ALTERNATIVA APROVADA",
}


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def safe_numeric(
    series: pd.Series,
) -> pd.Series:

    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    numerator = safe_numeric(
        numerator
    )

    denominator = safe_numeric(
        denominator
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


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Percentil transversal.

    Sempre:
        valor maior = melhor sinal.
    """

    values = safe_numeric(
        series
    )

    return values.rank(
        pct=True,
        ascending=higher_is_better,
        method="average",
    )


def historical_percentile(
    current_value,
    history: pd.Series,
    lower_is_better: bool = True,
) -> float:
    """
    Percentil do valor atual contra o próprio histórico.

    Retorno sempre padronizado:
        1.0 = condição mais favorável
        0.0 = condição menos favorável
    """

    current_value = pd.to_numeric(
        pd.Series(
            [current_value]
        ),
        errors="coerce",
    ).iloc[0]

    hist = (
        pd.to_numeric(
            history,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if (
        pd.isna(current_value)
        or
        hist.empty
    ):
        return np.nan

    percentile = float(
        (
            hist
            <=
            current_value
        ).mean()
    )

    if lower_is_better:
        percentile = 1.0 - percentile

    return percentile


def mean_valid(
    values: Iterable,
    minimum: int = 1,
) -> float:

    values = pd.to_numeric(
        pd.Series(
            list(values)
        ),
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = values.dropna()

    if len(valid) < minimum:
        return np.nan

    return float(
        valid.mean()
    )


# ======================================================================================
# 3. VALUATION SCORE
# ======================================================================================

VALUATION_METRICS = [
    "pe",
    "pb",
    "ps",
    "p_ocf",
    "p_fcf",
]


def calculate_valuation_score(
    current_row: pd.Series,
    historical_valuation: pd.DataFrame,
) -> float:
    """
    Mede quão barata a empresa está contra o PRÓPRIO histórico.

    Múltiplos:
        PE
        PB
        PS
        P/OCF
        P/FCF

    Múltiplo menor = melhor valuation.
    """

    ticker = str(
        current_row[
            "ticker"
        ]
    ).upper()

    history = historical_valuation[
        historical_valuation[
            "ticker"
        ].astype(str).str.upper()
        ==
        ticker
    ].copy()

    if history.empty:
        return np.nan

    components = []

    for metric in VALUATION_METRICS:

        if (
            metric not in current_row.index
            or
            metric not in history.columns
        ):
            continue

        score = historical_percentile(
            current_value=current_row[
                metric
            ],
            history=history[
                metric
            ],
            lower_is_better=True,
        )

        components.append(
            score
        )

    return mean_valid(
        components,
        minimum=2,
    )


# ======================================================================================
# 4. DESCONTO DE PREÇO
# ======================================================================================

def calculate_discount_metrics(
    prices: pd.Series,
) -> Dict[str, float]:
    """
    Componentes de desconto utilizados no estudo:

        • drawdown 52 semanas
        • drawdown 3 anos
        • distância da média de 200 dias
        • posição na faixa de 3 anos

    Todos são convertidos para:
        maior score = maior desconto.
    """

    prices = (
        safe_numeric(
            prices
        )
        .dropna()
        .sort_index()
    )

    if prices.empty:

        return {
            "discount_52w":
                np.nan,

            "discount_3y":
                np.nan,

            "discount_ma200":
                np.nan,

            "range_position_3y":
                np.nan,
        }

    current = float(
        prices.iloc[-1]
    )

    # ------------------------------------------------------------------
    # 52 semanas
    # ------------------------------------------------------------------

    window_52w = (
        prices
        .tail(252)
    )

    high_52w = (
        window_52w.max()
        if not window_52w.empty
        else np.nan
    )

    discount_52w = (
        1.0
        -
        current / high_52w
        if (
            pd.notna(high_52w)
            and
            high_52w > 0
        )
        else np.nan
    )

    # ------------------------------------------------------------------
    # 3 anos
    # ------------------------------------------------------------------

    window_3y = (
        prices
        .tail(756)
    )

    high_3y = (
        window_3y.max()
        if not window_3y.empty
        else np.nan
    )

    low_3y = (
        window_3y.min()
        if not window_3y.empty
        else np.nan
    )

    discount_3y = (
        1.0
        -
        current / high_3y
        if (
            pd.notna(high_3y)
            and
            high_3y > 0
        )
        else np.nan
    )

    # ------------------------------------------------------------------
    # Média 200 dias
    #
    # Positivo quando preço está abaixo da média.
    # ------------------------------------------------------------------

    ma200 = (
        prices
        .tail(200)
        .mean()
        if len(prices) >= 200
        else np.nan
    )

    discount_ma200 = (
        1.0
        -
        current / ma200
        if (
            pd.notna(ma200)
            and
            ma200 > 0
        )
        else np.nan
    )

    # ------------------------------------------------------------------
    # Posição na faixa de 3 anos
    #
    # 0 = mínima
    # 1 = máxima
    #
    # Para score de desconto:
    #     1 - posição
    # ------------------------------------------------------------------

    if (
        pd.notna(high_3y)
        and
        pd.notna(low_3y)
        and
        high_3y > low_3y
    ):

        position = (
            current
            -
            low_3y
        ) / (
            high_3y
            -
            low_3y
        )

        range_discount = (
            1.0
            -
            position
        )

    else:

        range_discount = np.nan

    return {
        "discount_52w":
            discount_52w,

        "discount_3y":
            discount_3y,

        "discount_ma200":
            discount_ma200,

        "range_position_3y":
            range_discount,
    }


def build_discount_table(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for ticker in (
        portfolio[
            "ticker"
        ]
        .astype(str)
        .str.upper()
    ):

        if ticker not in prices.columns:

            metrics = {
                "discount_52w":
                    np.nan,

                "discount_3y":
                    np.nan,

                "discount_ma200":
                    np.nan,

                "range_position_3y":
                    np.nan,
            }

        else:

            metrics = (
                calculate_discount_metrics(
                    prices[
                        ticker
                    ]
                )
            )

        rows.append(
            {
                "ticker":
                    ticker,

                **metrics,
            }
        )

    table = pd.DataFrame(
        rows
    )

    discount_columns = [
        "discount_52w",
        "discount_3y",
        "discount_ma200",
        "range_position_3y",
    ]

    table[
        "discount_score"
    ] = (
        table[
            discount_columns
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    table[
        "discount_components"
    ] = (
        table[
            discount_columns
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    table.loc[
        table[
            "discount_components"
        ]
        <
        2,
        "discount_score",
    ] = np.nan

    return table


# ======================================================================================
# 5. FUNDAMENTAL PRESERVATION
# ======================================================================================

FUNDAMENTAL_COMPONENTS = [
    "revenue_growth",
    "eps_growth",
    "operating_cash_flow_growth",
    "operating_margin",
    "net_margin",
]


def calculate_fundamental_preservation(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte os fundamentos atuais em score transversal.

    Componentes preservados do estudo:
        crescimento de receita
        crescimento de EPS
        crescimento de OCF
        margem operacional
        margem líquida
    """

    df = portfolio.copy()

    available_components = []

    for metric in FUNDAMENTAL_COMPONENTS:

        if metric not in df.columns:
            df[metric] = np.nan

        component_name = (
            f"{metric}_entry_pct"
        )

        df[
            component_name
        ] = percentile_score(
            df[
                metric
            ],
            higher_is_better=True,
        )

        available_components.append(
            component_name
        )

    df[
        "fundamental_score"
    ] = (
        df[
            available_components
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    df[
        "fundamental_components"
    ] = (
        df[
            available_components
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    df.loc[
        df[
            "fundamental_components"
        ]
        <
        2,
        "fundamental_score",
    ] = np.nan

    return df


# ======================================================================================
# 6. MOMENTUM TECHNOLOGY
# ======================================================================================

def calculate_momentum(
    prices: pd.Series,
    months: int,
) -> float:

    prices = (
        safe_numeric(
            prices
        )
        .dropna()
        .sort_index()
    )

    trading_days = (
        21
        *
        months
    )

    if len(prices) <= trading_days:

        return np.nan

    current = float(
        prices.iloc[-1]
    )

    previous = float(
        prices.iloc[
            -trading_days - 1
        ]
    )

    if (
        previous == 0
        or
        not np.isfinite(previous)
    ):
        return np.nan

    return (
        current
        /
        previous
        -
        1.0
    )


def build_technology_momentum(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    technology = portfolio[
        portfolio[
            "sector"
        ]
        ==
        TECHNOLOGY
    ].copy()

    rows = []

    for ticker in (
        technology[
            "ticker"
        ]
        .astype(str)
        .str.upper()
    ):

        if ticker not in prices.columns:

            momentum_6m = np.nan
            momentum_12m = np.nan

        else:

            series = prices[
                ticker
            ]

            momentum_6m = (
                calculate_momentum(
                    series,
                    6,
                )
            )

            momentum_12m = (
                calculate_momentum(
                    series,
                    12,
                )
            )

        rows.append(
            {
                "ticker":
                    ticker,

                "momentum_6m":
                    momentum_6m,

                "momentum_12m":
                    momentum_12m,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result[
        "momentum_6m_pct"
    ] = percentile_score(
        result[
            "momentum_6m"
        ],
        higher_is_better=True,
    )

    result[
        "momentum_12m_pct"
    ] = percentile_score(
        result[
            "momentum_12m"
        ],
        higher_is_better=True,
    )

    # Regra vencedora:
    # combinação 6M + 12M

    result[
        "momentum_score"
    ] = (
        result[
            [
                "momentum_6m_pct",
                "momentum_12m_pct",
            ]
        ]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    return result


# ======================================================================================
# 7. STATUS DESCRITIVOS
# ======================================================================================

def valuation_status(
    percentile: float,
) -> str:

    if pd.isna(percentile):
        return "N/D"

    if percentile >= 0.80:
        return "MUITO BARATA"

    if percentile >= 0.60:
        return "BARATA"

    if percentile >= 0.40:
        return "NEUTRA"

    if percentile >= 0.20:
        return "CARA"

    return "MUITO CARA"


def discount_status(
    percentile: float,
) -> str:

    if pd.isna(percentile):
        return "N/D"

    if percentile >= 0.80:
        return "MUITO ALTO"

    if percentile >= 0.60:
        return "ALTO"

    if percentile >= 0.40:
        return "MÉDIO"

    if percentile >= 0.20:
        return "BAIXO"

    return "SEM DESCONTO"


def fundamental_status(
    percentile: float,
) -> str:

    if pd.isna(percentile):
        return "N/D"

    if percentile >= 0.80:
        return "MUITO FORTES"

    if percentile >= 0.60:
        return "FORTES"

    if percentile >= 0.40:
        return "PRESERVADOS"

    if percentile >= 0.20:
        return "ENFRAQUECENDO"

    return "FRACOS"


# ======================================================================================
# 8. CLASSIFICAÇÃO FINAL
# ======================================================================================

def classify_entry(
    signal_percentile: float,
    sector: str,
) -> str:

    if pd.isna(
        signal_percentile
    ):

        # Na Célula 41:
        # ausência de informação suficiente -> AGUARDAR

        return "AGUARDAR"

    if (
        signal_percentile
        >=
        ENTRY_STRONG_PERCENTILE
    ):

        if sector == INDUSTRIALS:

            # Regra condicional:
            # Industrials nunca recebe ENTRADA FORTE.

            return "ENTRADA"

        return "ENTRADA FORTE"

    if (
        signal_percentile
        >=
        ENTRY_PERCENTILE
    ):

        return "ENTRADA"

    if (
        signal_percentile
        >=
        WAIT_PERCENTILE
    ):

        return "AGUARDAR"

    return "NÃO COMPRAR AGORA"


# ======================================================================================
# 9. MOTOR FINAL
# ======================================================================================

def classify_portfolio_entries(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    historical_valuation: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Recebe as 15 ações JÁ selecionadas.

    Retorna:
        score
        percentil
        classificação
        prioridade de compra
    """

    required = {
        "ticker",
        "sector",
    }

    if not required.issubset(
        portfolio.columns
    ):

        raise ValueError(
            "Portfolio precisa conter ticker e sector."
        )

    df = portfolio.copy()

    df[
        "ticker"
    ] = (
        df[
            "ticker"
        ]
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
    # Garantir estrutura 15 / 3 / 5-5-5
    # ------------------------------------------------------------------

    if (
        df[
            "ticker"
        ]
        .nunique()
        !=
        15
    ):

        raise RuntimeError(
            "entry.py recebeu carteira diferente de 15 ações."
        )

    sector_counts = (
        df
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
                sector_counts.get(
                    sector,
                    0,
                )
            )
            !=
            5
        ):

            raise RuntimeError(
                f"{sector}: estrutura diferente de 5 ações."
            )

    # ------------------------------------------------------------------
    # FUNDAMENTOS
    # ------------------------------------------------------------------

    df = (
        calculate_fundamental_preservation(
            df
        )
    )

    # ------------------------------------------------------------------
    # DESCONTO
    # ------------------------------------------------------------------

    discount = (
        build_discount_table(
            portfolio=df,
            prices=prices,
        )
    )

    df = df.merge(
        discount,
        on="ticker",
        how="left",
    )

    # Percentil do desconto dentro do setor

    df[
        "discount_percentile"
    ] = (
        df
        .groupby(
            "sector",
            group_keys=False,
        )[
            "discount_score"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # ------------------------------------------------------------------
    # FUNDAMENTAL PERCENTILE POR SETOR
    # ------------------------------------------------------------------

    df[
        "fundamental_percentile"
    ] = (
        df
        .groupby(
            "sector",
            group_keys=False,
        )[
            "fundamental_score"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # ------------------------------------------------------------------
    # VALUATION
    # ------------------------------------------------------------------

    df[
        "valuation_score"
    ] = np.nan

    if (
        historical_valuation is not None
        and
        not historical_valuation.empty
    ):

        valuation_values = []

        for _, row in df.iterrows():

            valuation_values.append(
                calculate_valuation_score(
                    current_row=row,
                    historical_valuation=historical_valuation,
                )
            )

        df[
            "valuation_score"
        ] = valuation_values

    df[
        "valuation_percentile"
    ] = (
        df
        .groupby(
            "sector",
            group_keys=False,
        )[
            "valuation_score"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # ------------------------------------------------------------------
    # TECHNOLOGY MOMENTUM
    # ------------------------------------------------------------------

    momentum = (
        build_technology_momentum(
            portfolio=df,
            prices=prices,
        )
    )

    if not momentum.empty:

        df = df.merge(
            momentum,
            on="ticker",
            how="left",
        )

    else:

        df[
            "momentum_6m"
        ] = np.nan

        df[
            "momentum_12m"
        ] = np.nan

        df[
            "momentum_score"
        ] = np.nan

    # ------------------------------------------------------------------
    # FINAL SIGNAL SCORE
    # ------------------------------------------------------------------

    df[
        "final_signal_score"
    ] = np.nan

    # Health Care
    #
    # 10% valuation
    # 80% desconto
    # 10% fundamentos

    mask = (
        df[
            "sector"
        ]
        ==
        HEALTH_CARE
    )

    health_components = pd.DataFrame(
        {
            "valuation":
                df.loc[
                    mask,
                    "valuation_percentile",
                ],

            "discount":
                df.loc[
                    mask,
                    "discount_percentile",
                ],

            "fundamental":
                df.loc[
                    mask,
                    "fundamental_percentile",
                ],
        }
    )

    health_weights = pd.Series(
        {
            "valuation":
                0.10,

            "discount":
                0.80,

            "fundamental":
                0.10,
        }
    )

    available_weight = (
        health_components
        .notna()
        .mul(
            health_weights,
            axis=1,
        )
        .sum(
            axis=1
        )
    )

    weighted_sum = (
        health_components
        .mul(
            health_weights,
            axis=1,
        )
        .sum(
            axis=1,
            skipna=True,
        )
    )

    health_score = (
        weighted_sum
        /
        available_weight.replace(
            0,
            np.nan,
        )
    )

    df.loc[
        mask,
        "final_signal_score",
    ] = health_score

    # Industrials
    #
    # 20% desconto
    # 80% fundamentos

    mask = (
        df[
            "sector"
        ]
        ==
        INDUSTRIALS
    )

    industrial_components = pd.DataFrame(
        {
            "discount":
                df.loc[
                    mask,
                    "discount_percentile",
                ],

            "fundamental":
                df.loc[
                    mask,
                    "fundamental_percentile",
                ],
        }
    )

    industrial_weights = pd.Series(
        {
            "discount":
                0.20,

            "fundamental":
                0.80,
        }
    )

    available_weight = (
        industrial_components
        .notna()
        .mul(
            industrial_weights,
            axis=1,
        )
        .sum(
            axis=1
        )
    )

    weighted_sum = (
        industrial_components
        .mul(
            industrial_weights,
            axis=1,
        )
        .sum(
            axis=1,
            skipna=True,
        )
    )

    industrial_score = (
        weighted_sum
        /
        available_weight.replace(
            0,
            np.nan,
        )
    )

    df.loc[
        mask,
        "final_signal_score",
    ] = industrial_score

    # Technology
    #
    # Momentum 6M + 12M

    mask = (
        df[
            "sector"
        ]
        ==
        TECHNOLOGY
    )

    df.loc[
        mask,
        "final_signal_score",
    ] = df.loc[
        mask,
        "momentum_score",
    ]

    # ------------------------------------------------------------------
    # SIGNAL PERCENTILE
    #
    # Percentil transversal do score dentro do setor.
    # ------------------------------------------------------------------

    df[
        "signal_percentile"
    ] = (
        df
        .groupby(
            "sector",
            group_keys=False,
        )[
            "final_signal_score"
        ]
        .rank(
            pct=True,
            method="average",
        )
    )

    # ------------------------------------------------------------------
    # STATUS DESCRITIVOS
    # ------------------------------------------------------------------

    df[
        "valuation_status"
    ] = df[
        "valuation_percentile"
    ].apply(
        valuation_status
    )

    df[
        "discount_status"
    ] = df[
        "discount_percentile"
    ].apply(
        discount_status
    )

    df[
        "fundamental_status"
    ] = df[
        "fundamental_percentile"
    ].apply(
        fundamental_status
    )

    # ------------------------------------------------------------------
    # METODOLOGIA / STATUS SETORIAL
    # ------------------------------------------------------------------

    df[
        "timing_method"
    ] = df[
        "sector"
    ].map(
        ENTRY_METHODS
    )

    df[
        "sector_validation_status"
    ] = df[
        "sector"
    ].map(
        SECTOR_VALIDATION_STATUS
    )

    # ------------------------------------------------------------------
    # SINAL FINAL
    # ------------------------------------------------------------------

    df[
        "entry_signal"
    ] = [
        classify_entry(
            signal_percentile=pct,
            sector=sector,
        )
        for pct, sector in zip(
            df[
                "signal_percentile"
            ],
            df[
                "sector"
            ],
        )
    ]

    # ------------------------------------------------------------------
    # PRIORIDADE DENTRO DO SETOR
    # ------------------------------------------------------------------

    df[
        "buy_priority_sector"
    ] = (
        df
        .groupby(
            "sector"
        )[
            "final_signal_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
    )

    # ------------------------------------------------------------------
    # STATUS DA SELEÇÃO
    # ------------------------------------------------------------------

    df[
        "selection_status"
    ] = "APROVADA"

    # ------------------------------------------------------------------
    # ORDEM FINAL
    # ------------------------------------------------------------------

    signal_order = {

        "ENTRADA FORTE":
            1,

        "ENTRADA":
            2,

        "AGUARDAR":
            3,

        "NÃO COMPRAR AGORA":
            4,
    }

    df[
        "_signal_order"
    ] = df[
        "entry_signal"
    ].map(
        signal_order
    )

    df = (
        df
        .sort_values(
            [
                "_signal_order",
                "signal_percentile",
                "final_signal_score",
            ],
            ascending=[
                True,
                False,
                False,
            ],
            na_position="last",
        )
        .drop(
            columns=[
                "_signal_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ======================================================================================
# 10. RESUMO EXECUTIVO
# ======================================================================================

def build_entry_summary(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    columns = [

        "sector",
        "buy_priority_sector",
        "ticker",
        "selection_status",
        "valuation_status",
        "discount_status",
        "fundamental_status",
        "timing_method",
        "sector_validation_status",
        "final_signal_score",
        "signal_percentile",
        "entry_signal",
    ]

    available = [
        column
        for column in columns
        if column in ranking.columns
    ]

    return ranking[
        available
    ].copy()


# ======================================================================================
# 11. AUDITORIA
# ======================================================================================

def audit_entry_ranking(
    ranking: pd.DataFrame,
) -> Dict:

    signals = (
        ranking[
            "entry_signal"
        ]
        .value_counts()
        .to_dict()
    )

    sector_counts = (
        ranking
        .groupby(
            "sector"
        )[
            "ticker"
        ]
        .nunique()
        .to_dict()
    )

    industrial_strong = int(
        (
            (
                ranking[
                    "sector"
                ]
                ==
                INDUSTRIALS
            )
            &
            (
                ranking[
                    "entry_signal"
                ]
                ==
                "ENTRADA FORTE"
            )
        ).sum()
    )

    return {

        "number_of_stocks":
            int(
                ranking[
                    "ticker"
                ].nunique()
            ),

        "number_of_sectors":
            int(
                ranking[
                    "sector"
                ].nunique()
            ),

        "sector_counts":
            sector_counts,

        "entry_strong":
            int(
                signals.get(
                    "ENTRADA FORTE",
                    0,
                )
            ),

        "entry":
            int(
                signals.get(
                    "ENTRADA",
                    0,
                )
            ),

        "wait":
            int(
                signals.get(
                    "AGUARDAR",
                    0,
                )
            ),

        "do_not_buy":
            int(
                signals.get(
                    "NÃO COMPRAR AGORA",
                    0,
                )
            ),

        "industrial_strong_violation":
            industrial_strong,

        "structure_ok":
            (
                ranking[
                    "ticker"
                ].nunique()
                ==
                15
                and
                all(
                    sector_counts.get(
                        sector,
                        0,
                    )
                    ==
                    5
                    for sector in SECTORS
                )
                and
                industrial_strong
                ==
                0
            ),
    }


# ======================================================================================
# 12. TESTE DO MÓDULO
# ======================================================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO ACOES AMERICANO — ENTRY ENGINE"
    )

    print(
        "=" * 100
    )

    print(
        "\nArquitetura validada:"
    )

    print(
        "  Health Care"
        " -> 10% Valuation + "
        "80% Desconto + "
        "10% Fundamentos"
    )

    print(
        "  Industrials"
        " -> 20% Desconto + "
        "80% Fundamentos"
        " — CONDICIONAL"
    )

    print(
        "  Information Technology"
        " -> Momentum 6M + 12M"
    )

    print(
        "\nClassificações:"
    )

    print(
        "  >= 75% -> ENTRADA FORTE"
    )

    print(
        "  >= 50% -> ENTRADA"
    )

    print(
        "  >= 25% -> AGUARDAR"
    )

    print(
        "  <  25% -> NÃO COMPRAR AGORA"
    )

    print(
        "\nRestrição:"
    )

    print(
        "  Industrials nunca recebe "
        "ENTRADA FORTE."
    )

    print(
        "\nEntry engine carregado com sucesso."
    )
