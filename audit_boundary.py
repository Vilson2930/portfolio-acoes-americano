# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_boundary.py
# ======================================================================================
#
# OBJETIVO
# --------
# Auditar a regra de fronteira 5º x 6º da CÉLULA 31 contra a implementação atual
# do GitHub em selection.py.
#
# O teste verifica:
#   1) mesmos 5º e 6º colocados por setor;
#   2) mesmos gaps absoluto e relativo;
#   3) mesma classificação near_tie;
#   4) mesmas métricas de risco;
#   5) mesmo número de melhorias;
#   6) mesma detecção de deterioração relevante;
#   7) mesma decisão final KEEP / SWAP;
#   8) mesma carteira final após a fronteira.
#
# REGRA CONGELADA DA CÉLULA 31
# ----------------------------
# near_tie:
#   gap absoluto <= 0.01
#   OU
#   gap relativo <= 1%
#
# troca:
#   >= 3 melhorias de risco
#   E sem deterioração relevante
#
# melhorias:
#   volatilidade             <= -0.0025
#   Sharpe                   >= +0.02
#   drawdown                 >= +0.005
#   correlação média         <= -0.01
#   maior contribuição risco <= -0.01
#
# deterioração relevante:
#   volatilidade             > +0.005
#   OU Sharpe                < -0.02
#   OU drawdown              < -0.01
#
# janela de risco:
#   desde 2024-01-01
#   mínimo 120 dias comuns
#
# ======================================================================================

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config import SECTORS, SECTOR_TARGETS

from data import (
    build_base_universe,
    enrich_sectors,
    filter_target_sectors,
    download_fundamentals,
    prepare_selection_snapshot,
    download_prices,
)

from selection import (
    build_derived_metrics,
    score_sector,
    build_frontier_audit,
    select_portfolio,
    get_boundary_test_tickers,
)


# ======================================================================================
# 1. CONFIGURAÇÃO DA REFERÊNCIA — CÉLULA 31
# ======================================================================================

REFERENCE_DATE = pd.Timestamp("2026-08-24")

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
RISK_START = pd.Timestamp("2024-01-01")
ANNUALIZATION_DAILY = 252

NUMERIC_TOLERANCE = 1e-10

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMPARISON = OUTPUT_DIR / "audit_boundary_comparison.csv"
OUTPUT_REFERENCE = OUTPUT_DIR / "audit_boundary_reference.csv"
OUTPUT_GITHUB = OUTPUT_DIR / "audit_boundary_github.csv"
OUTPUT_PORTFOLIO = OUTPUT_DIR / "audit_boundary_portfolio_comparison.csv"


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def header(title: str):
    print("\n" + "=" * 150)
    print(title)
    print("=" * 150)


def normalize_ticker(value):
    return (
        str(value)
        .strip()
        .upper()
        .replace(".", "-")
    )


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:

    close = prices.copy()

    close.index = pd.to_datetime(
        close.index
    )

    close.columns = [
        normalize_ticker(c)
        for c in close.columns
    ]

    return (
        close
        .sort_index()
        .loc[
            lambda df:
                df.index >= RISK_START
        ]
    )


# ======================================================================================
# 3. RANKING FUNDAMENTAL COMPLETO
# ======================================================================================

def build_reference_ranking(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    base = build_derived_metrics(
        snapshot
    )

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
                scored[
                    score_column
                ]
                .notna()
            ]
            .sort_values(
                [
                    score_column,
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        scored["sector_rank"] = np.arange(
            1,
            len(scored) + 1,
        )

        scored["selection_score"] = (
            scored[
                score_column
            ]
        )

        parts.append(
            scored
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def build_raw_portfolio(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    parts = []

    for sector in SECTORS:

        target = int(
            SECTOR_TARGETS[
                sector
            ]
        )

        part = (
            ranking[
                ranking["sector"] == sector
            ]
            .sort_values(
                "sector_rank"
            )
            .head(target)
            .copy()
        )

        parts.append(part)

    portfolio = pd.concat(
        parts,
        ignore_index=True,
    )

    portfolio["selection_rank"] = (
        portfolio[
            "sector_rank"
        ]
        .astype(int)
    )

    return portfolio


# ======================================================================================
# 4. FRONTEIRA FUNDAMENTAL — REFERÊNCIA
# ======================================================================================

def build_reference_boundary(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for sector in SECTORS:

        sector_ranking = (
            ranking[
                ranking["sector"] == sector
            ]
            .sort_values(
                "sector_rank"
            )
        )

        row5 = (
            sector_ranking[
                sector_ranking[
                    "sector_rank"
                ]
                ==
                5
            ]
            .iloc[0]
        )

        row6 = (
            sector_ranking[
                sector_ranking[
                    "sector_rank"
                ]
                ==
                6
            ]
            .iloc[0]
        )

        score5 = float(
            row5[
                "selection_score"
            ]
        )

        score6 = float(
            row6[
                "selection_score"
            ]
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
                pd.notna(
                    relative_gap
                )
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

                "rank5_ticker":
                    normalize_ticker(
                        row5["ticker"]
                    ),

                "rank5_score":
                    score5,

                "rank6_ticker":
                    normalize_ticker(
                        row6["ticker"]
                    ),

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

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 5. MÉTRICAS DE RISCO — REFERÊNCIA CÉLULA 31
# ======================================================================================

def portfolio_risk_metrics(
    ticker_list,
    daily_returns: pd.DataFrame,
):

    common = (
        daily_returns[
            ticker_list
        ]
        .dropna()
    )

    if len(common) < MIN_COMMON_DAYS:
        return None

    n = len(
        ticker_list
    )

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

    daily_std = (
        portfolio_return
        .std(ddof=1)
    )

    volatility = (
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

    max_drawdown = (
        drawdown.min()
    )

    corr = (
        common.corr()
    )

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
            corr_values.append(
                value
            )

    mean_correlation = (
        np.mean(
            corr_values
        )
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
        not np.isfinite(
            variance
        )
        or
        variance <= 0
    ):
        return None

    vol = np.sqrt(
        variance
    )

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

    max_risk_contribution = float(
        np.max(
            risk_contribution
        )
    )

    return {
        "observations":
            len(common),

        "volatility":
            float(volatility),

        "sharpe":
            float(sharpe),

        "max_drawdown":
            float(max_drawdown),

        "mean_correlation":
            float(mean_correlation),

        "max_risk_contribution":
            max_risk_contribution,
    }


# ======================================================================================
# 6. TESTE DA FRONTEIRA — REFERÊNCIA
# ======================================================================================

def run_reference_boundary_test(
    ranking: pd.DataFrame,
    raw_portfolio: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:

    boundary = (
        build_reference_boundary(
            ranking
        )
    )

    close = (
        normalize_prices(
            prices
        )
    )

    returns = (
        close
        .pct_change(
            fill_method=None
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )

    current_tickers = (
        raw_portfolio[
            "ticker"
        ]
        .map(
            normalize_ticker
        )
        .tolist()
    )

    rows = []

    for row in boundary.itertuples(
        index=False
    ):

        incumbent = (
            row.rank5_ticker
        )

        challenger = (
            row.rank6_ticker
        )

        comparison_tickers = sorted(
            set(
                current_tickers
                +
                [
                    challenger
                ]
            )
        )

        missing = [
            ticker
            for ticker
            in comparison_tickers
            if ticker
            not in returns.columns
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

        if len(
            common_dates
        ) < MIN_COMMON_DAYS:

            rows.append(
                {
                    **row._asdict(),

                    "observations":
                        len(
                            common_dates
                        ),

                    "test_valid":
                        False,

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
            portfolio_risk_metrics(
                current_tickers,
                comparison_returns,
            )
        )

        candidate_portfolio = [
            challenger
            if ticker
            ==
            incumbent
            else ticker
            for ticker
            in current_tickers
        ]

        candidate_metrics = (
            portfolio_risk_metrics(
                candidate_portfolio,
                comparison_returns,
            )
        )

        if (
            original_metrics
            is None
            or
            candidate_metrics
            is None
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

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 7. CARTEIRA FINAL DE REFERÊNCIA
# ======================================================================================

def apply_reference_decisions(
    ranking: pd.DataFrame,
    raw_portfolio: pd.DataFrame,
    boundary_test: pd.DataFrame,
) -> pd.DataFrame:

    final = (
        raw_portfolio
        .copy()
    )

    for row in boundary_test.itertuples(
        index=False
    ):

        if not str(
            row.decision
        ).startswith(
            "SWAP"
        ):
            continue

        final = (
            final[
                final["ticker"]
                !=
                row.rank5_ticker
            ]
            .copy()
        )

        challenger = (
            ranking[
                (
                    ranking["sector"]
                    ==
                    row.sector
                )
                &
                (
                    ranking["ticker"]
                    ==
                    row.rank6_ticker
                )
            ]
            .head(1)
            .copy()
        )

        final = pd.concat(
            [
                final,
                challenger,
            ],
            ignore_index=True,
        )

    final = (
        final
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

    final["selection_rank"] = (
        final
        .groupby(
            "sector"
        )
        .cumcount()
        +
        1
    )

    return final


# ======================================================================================
# 8. COMPARAÇÃO
# ======================================================================================

NUMERIC_COLUMNS = [
    "rank5_score",
    "rank6_score",
    "absolute_gap",
    "relative_gap",
    "observations",
    "baseline_vol",
    "candidate_vol",
    "delta_vol",
    "baseline_sharpe",
    "candidate_sharpe",
    "delta_sharpe",
    "baseline_max_dd",
    "candidate_max_dd",
    "delta_max_dd",
    "baseline_mean_corr",
    "candidate_mean_corr",
    "delta_mean_corr",
    "baseline_max_risk",
    "candidate_max_risk",
    "delta_max_risk",
    "risk_improvements",
]


def compare_boundary(
    reference: pd.DataFrame,
    github: pd.DataFrame,
) -> pd.DataFrame:

    ref = reference.copy()
    gh = github.copy()

    ref = ref.add_prefix(
        "reference_"
    )

    gh = gh.add_prefix(
        "github_"
    )

    comparison = (
        ref.merge(
            gh,
            left_on="reference_sector",
            right_on="github_sector",
            how="outer",
        )
    )

    comparison["sector"] = (
        comparison[
            "reference_sector"
        ]
        .combine_first(
            comparison[
                "github_sector"
            ]
        )
    )

    comparison[
        "tickers_ok"
    ] = (
        (
            comparison[
                "reference_rank5_ticker"
            ]
            ==
            comparison[
                "github_rank5_ticker"
            ]
        )
        &
        (
            comparison[
                "reference_rank6_ticker"
            ]
            ==
            comparison[
                "github_rank6_ticker"
            ]
        )
    )

    comparison[
        "near_tie_ok"
    ] = (
        comparison[
            "reference_near_tie"
        ]
        ==
        comparison[
            "github_near_tie"
        ]
    )

    comparison[
        "decision_ok"
    ] = (
        comparison[
            "reference_decision"
        ]
        ==
        comparison[
            "github_decision"
        ]
    )

    comparison[
        "test_valid_ok"
    ] = (
        comparison[
            "reference_test_valid"
        ]
        ==
        comparison[
            "github_test_valid"
        ]
    )

    comparison[
        "severe_deterioration_ok"
    ] = (
        comparison[
            "reference_severe_deterioration"
        ]
        ==
        comparison[
            "github_severe_deterioration"
        ]
    )

    numeric_ok_columns = []

    for column in NUMERIC_COLUMNS:

        ref_col = (
            f"reference_{column}"
        )

        gh_col = (
            f"github_{column}"
        )

        ok_col = (
            f"{column}_ok"
        )

        numeric_ok_columns.append(
            ok_col
        )

        left = pd.to_numeric(
            comparison.get(
                ref_col
            ),
            errors="coerce",
        )

        right = pd.to_numeric(
            comparison.get(
                gh_col
            ),
            errors="coerce",
        )

        both_nan = (
            left.isna()
            &
            right.isna()
        )

        diff = (
            left
            -
            right
        ).abs()

        comparison[
            ok_col
        ] = (
            both_nan
            |
            (
                diff
                <=
                NUMERIC_TOLERANCE
            )
        )

    comparison[
        "numeric_ok"
    ] = (
        comparison[
            numeric_ok_columns
        ]
        .all(axis=1)
    )

    comparison[
        "row_ok"
    ] = (
        comparison[
            [
                "tickers_ok",
                "near_tie_ok",
                "decision_ok",
                "test_valid_ok",
                "severe_deterioration_ok",
                "numeric_ok",
            ]
        ]
        .all(axis=1)
    )

    return (
        comparison
        .sort_values(
            "sector"
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 9. EXECUÇÃO
# ======================================================================================

def run_audit():

    header(
        "AUDITORIA DA FRONTEIRA — CÉLULA 31 x GITHUB"
    )

    print(
        f"\nData de referência             : "
        f"{REFERENCE_DATE.date()}"
    )

    print(
        "Near tie                      : "
        "gap absoluto <= 0.01 OU relativo <= 1%"
    )

    print(
        "Melhorias mínimas para troca  : "
        f"{MIN_RISK_IMPROVEMENTS}"
    )

    print(
        f"Janela de risco               : "
        f"{RISK_START.date()} em diante"
    )

    print(
        f"Dias comuns mínimos           : "
        f"{MIN_COMMON_DAYS}"
    )

    # ------------------------------------------------------------------
    # UNIVERSO
    # ------------------------------------------------------------------

    header(
        "1. UNIVERSO E FUNDAMENTOS"
    )

    universe = (
        build_base_universe()
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

    fundamentals, errors = (
        download_fundamentals(
            universe=universe,
            use_cache=True,
        )
    )

    print(
        f"\nEmpresas nos 3 setores        : "
        f"{len(universe):,}"
    )

    print(
        f"Observações fundamentais      : "
        f"{len(fundamentals):,}"
    )

    if isinstance(
        errors,
        pd.DataFrame,
    ):

        print(
            f"Empresas com erro             : "
            f"{len(errors):,}"
        )

    # ------------------------------------------------------------------
    # SNAPSHOT
    # ------------------------------------------------------------------

    header(
        "2. SNAPSHOT"
    )

    snapshot = (
        prepare_selection_snapshot(
            universe=universe,
            fundamentals=fundamentals,
            as_of_date=REFERENCE_DATE,
        )
    )

    print(
        f"\nEmpresas no snapshot          : "
        f"{len(snapshot):,}"
    )

    # ------------------------------------------------------------------
    # RANKING + TICKERS
    # ------------------------------------------------------------------

    ranking = (
        build_reference_ranking(
            snapshot
        )
    )

    raw_portfolio = (
        build_raw_portfolio(
            ranking
        )
    )

    reference_boundary_pre = (
        build_reference_boundary(
            ranking
        )
    )

    header(
        "3. FRONTEIRA FUNDAMENTAL"
    )

    print(
        reference_boundary_pre
        .round(6)
        .to_string(
            index=False
        )
    )

    test_tickers = (
        get_boundary_test_tickers(
            snapshot
        )
    )

    print(
        f"\nTickers necessários            : "
        f"{len(test_tickers)}"
    )

    print(
        ", ".join(
            test_tickers
        )
    )

    # ------------------------------------------------------------------
    # PREÇOS
    # ------------------------------------------------------------------

    header(
        "4. PREÇOS"
    )

    prices = (
        download_prices(
            tickers=test_tickers,
            start="2013-01-01",
        )
    )

    print(
        f"\nTickers com preços            : "
        f"{len(prices.columns)}"
    )

    print(
        f"Primeira data                 : "
        f"{prices.index.min().date()}"
    )

    print(
        f"Última data                   : "
        f"{prices.index.max().date()}"
    )

    # ------------------------------------------------------------------
    # REFERÊNCIA
    # ------------------------------------------------------------------

    header(
        "5. RESULTADO DE REFERÊNCIA — CÉLULA 31"
    )

    reference = (
        run_reference_boundary_test(
            ranking=ranking,
            raw_portfolio=raw_portfolio,
            prices=prices,
        )
    )

    print(
        reference
        .round(6)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # GITHUB
    # ------------------------------------------------------------------

    header(
        "6. RESULTADO DO GITHUB"
    )

    github = (
        build_frontier_audit(
            universe=snapshot,
            prices=prices,
        )
    )

    print(
        github
        .round(6)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # COMPARAÇÃO
    # ------------------------------------------------------------------

    header(
        "7. COMPARAÇÃO NUMÉRICA"
    )

    comparison = (
        compare_boundary(
            reference,
            github,
        )
    )

    display_columns = [
        "sector",
        "tickers_ok",
        "near_tie_ok",
        "numeric_ok",
        "severe_deterioration_ok",
        "test_valid_ok",
        "decision_ok",
        "row_ok",
    ]

    print(
        comparison[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # CARTEIRA FINAL
    # ------------------------------------------------------------------

    header(
        "8. CARTEIRA FINAL — REFERÊNCIA x GITHUB"
    )

    reference_portfolio = (
        apply_reference_decisions(
            ranking=ranking,
            raw_portfolio=raw_portfolio,
            boundary_test=reference,
        )
    )

    github_portfolio = (
        select_portfolio(
            universe=snapshot,
            use_previous_portfolio=False,
            prices=prices,
        )
    )

    portfolio_rows = []

    portfolio_ok = True

    for sector in SECTORS:

        ref = (
            reference_portfolio[
                reference_portfolio[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "selection_rank"
            )[
                "ticker"
            ]
            .map(
                normalize_ticker
            )
            .tolist()
        )

        gh = (
            github_portfolio[
                github_portfolio[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "selection_rank"
            )[
                "ticker"
            ]
            .map(
                normalize_ticker
            )
            .tolist()
        )

        same = (
            ref
            ==
            gh
        )

        portfolio_ok = (
            portfolio_ok
            and
            same
        )

        portfolio_rows.append(
            {
                "sector":
                    sector,

                "reference":
                    ", ".join(
                        ref
                    ),

                "github":
                    ", ".join(
                        gh
                    ),

                "same":
                    same,
            }
        )

    portfolio_comparison = (
        pd.DataFrame(
            portfolio_rows
        )
    )

    print(
        portfolio_comparison
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # DIAGNÓSTICO FINAL
    # ------------------------------------------------------------------

    header(
        "9. DIAGNÓSTICO FINAL"
    )

    rows_ok = int(
        comparison[
            "row_ok"
        ]
        .sum()
    )

    total_rows = len(
        comparison
    )

    decisions_ok = int(
        comparison[
            "decision_ok"
        ]
        .sum()
    )

    numeric_ok = int(
        comparison[
            "numeric_ok"
        ]
        .sum()
    )

    print(
        f"\nFronteiras 100% aprovadas      : "
        f"{rows_ok}/{total_rows}"
    )

    print(
        f"Decisões idênticas             : "
        f"{decisions_ok}/{total_rows}"
    )

    print(
        f"Métricas numéricas idênticas   : "
        f"{numeric_ok}/{total_rows}"
    )

    print(
        f"Carteira final idêntica        : "
        f"{portfolio_ok}"
    )

    approved = (
        rows_ok
        ==
        total_rows
        and
        portfolio_ok
    )

    if approved:

        status = (
            "AUDITORIA APROVADA — "
            "GITHUB REPRODUZ A FRONTEIRA DA CÉLULA 31."
        )

    else:

        status = (
            "AUDITORIA NÃO APROVADA — "
            "EXISTEM DIVERGÊNCIAS NA FRONTEIRA."
        )

    print(
        f"\nSTATUS: {status}"
    )

    # ------------------------------------------------------------------
    # ARQUIVOS
    # ------------------------------------------------------------------

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    reference.to_csv(
        OUTPUT_REFERENCE,
        index=False,
    )

    github.to_csv(
        OUTPUT_GITHUB,
        index=False,
    )

    portfolio_comparison.to_csv(
        OUTPUT_PORTFOLIO,
        index=False,
    )

    header(
        "10. ARQUIVOS"
    )

    print(
        f"\nComparação fronteira : "
        f"{OUTPUT_COMPARISON}"
    )

    print(
        f"Referência            : "
        f"{OUTPUT_REFERENCE}"
    )

    print(
        f"GitHub                : "
        f"{OUTPUT_GITHUB}"
    )

    print(
        f"Comparação carteira   : "
        f"{OUTPUT_PORTFOLIO}"
    )

    if not approved:
        raise SystemExit(1)


# ======================================================================================
# 10. EXECUÇÃO DIRETA
# ======================================================================================

if __name__ == "__main__":

    run_audit()
