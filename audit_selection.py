# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_selection.py
# ======================================================================================
#
# OBJETIVO
# --------
# Auditar a SELEÇÃO FUNDAMENTAL atual do GitHub contra a metodologia congelada
# da CÉLULA 28 e a auditoria de fronteira da CÉLULA 29.
#
# Este arquivo NÃO altera:
#   • data.py
#   • selection.py
#   • entry.py
#   • config.py
#   • pesos
#   • thresholds
#   • carteira persistida
#
# Ele apenas compara:
#
#   CÉLULA 28 (referência)
#       Health Care              -> Financial Strength
#       Industrials              -> Growth
#       Information Technology   -> Quality
#
#   CÉLULA 28 — componentes:
#
#       Financial Strength
#           higher: cash_assets
#           lower : debt_assets, debt_equity
#           mínimo: 2 componentes
#
#       Growth
#           higher: revenue_growth, eps_growth,
#                   operating_cash_flow_growth
#           mínimo: 2 componentes
#
#       Quality
#           higher: roa, roe, operating_margin, net_margin
#           mínimo: 3 componentes
#
#   CÉLULA 28 — transformação:
#       • winsorização P5/P95 dentro do setor;
#       • percentil dentro do setor;
#       • menor dívida = melhor;
#       • score do fator = MÉDIA dos percentis válidos;
#       • ranking decrescente;
#       • Top 5 por setor.
#
# Também compara com a implementação atual de selection.py.
#
# ======================================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import SECTORS

from data import (
    build_base_universe,
    enrich_sectors,
    filter_target_sectors,
    download_fundamentals,
    prepare_selection_snapshot,
)

from selection import (
    build_derived_metrics,
    ensure_growth_metrics,
    score_sector,
    select_portfolio,
    build_frontier_audit,
)


# ======================================================================================
# 1. CONFIGURAÇÕES
# ======================================================================================

REFERENCE_DATE = pd.Timestamp("2026-08-24")

TARGET_SECTORS = [
    "Health Care",
    "Industrials",
    "Information Technology",
]

N_PER_SECTOR = 5

MIN_FACTOR_COMPONENTS = {
    "financial_strength": 2,
    "growth": 2,
    "quality": 3,
}

SECTOR_FACTOR = {
    "Health Care": "financial_strength_score",
    "Industrials": "growth_score",
    "Information Technology": "quality_score",
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_COMPARISON = (
    OUTPUT_DIR
    / "audit_selection_comparison.csv"
)

OUTPUT_REFERENCE_RANKING = (
    OUTPUT_DIR
    / "audit_selection_cell28_ranking.csv"
)

OUTPUT_GITHUB_RANKING = (
    OUTPUT_DIR
    / "audit_selection_github_ranking.csv"
)

OUTPUT_FRONTIER = (
    OUTPUT_DIR
    / "audit_selection_frontier.csv"
)


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def header(
    title: str,
):
    print(
        "\n"
        +
        "=" * 145
    )

    print(
        title
    )

    print(
        "=" * 145
    )


def normalize_ticker(
    value,
):
    return (
        str(value)
        .strip()
        .upper()
        .replace(
            ".",
            "-",
        )
    )


def winsorize_cell28(
    series: pd.Series,
) -> pd.Series:
    """
    Regra exata da Célula 28.

    • coerção numérica;
    • se menos de 10 válidos, não winsoriza;
    • caso contrário, clip P5/P95.
    """

    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = (
        series
        .dropna()
    )

    if len(valid) < 10:
        return series

    lower = valid.quantile(
        0.05
    )

    upper = valid.quantile(
        0.95
    )

    return series.clip(
        lower=lower,
        upper=upper,
    )


def prepare_cell28_aliases(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Traduz os nomes atuais do GitHub para os nomes usados na Célula 28.

    A tradução NÃO muda nenhuma fórmula econômica.
    """

    df = (
        build_derived_metrics(
            snapshot
        )
    )

    df = (
        ensure_growth_metrics(
            df
        )
    )

    # --------------------------------------------------------------
    # aliases da Célula 28
    # --------------------------------------------------------------

    aliases = {
        "cash_assets":
            "cash_to_assets",

        "debt_assets":
            "debt_to_assets",

        "debt_equity":
            "debt_to_equity",

        "revenue_growth":
            "revenue_growth_yoy",

        "eps_growth":
            "diluted_eps_growth_yoy",

        "operating_cash_flow_growth":
            "operating_cash_flow_growth_yoy",
    }

    for target, source in aliases.items():

        if source in df.columns:

            df[
                target
            ] = pd.to_numeric(
                df[
                    source
                ],
                errors="coerce",
            )

        else:

            df[
                target
            ] = np.nan

    return df


# ======================================================================================
# 3. REPRODUÇÃO INDEPENDENTE DA CÉLULA 28
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
    },

    "growth": {

        "higher": [
            "revenue_growth",
            "eps_growth",
            "operating_cash_flow_growth",
        ],

        "lower": [],
    },

    "quality": {

        "higher": [
            "roa",
            "roe",
            "operating_margin",
            "net_margin",
        ],

        "lower": [],
    },
}


ALL_METRICS = sorted(
    {
        metric

        for definition
        in FACTOR_DEFINITIONS.values()

        for direction
        in [
            "higher",
            "lower",
        ]

        for metric
        in definition[
            direction
        ]
    }
)


LOWER_IS_BETTER = {
    "debt_assets",
    "debt_equity",
}


def reproduce_cell28(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Implementação independente da Célula 28.

    IMPORTANTE:
    esta função NÃO chama score_sector() para calcular os scores de referência.
    Assim, ela consegue detectar divergência entre a metodologia original e
    selection.py.
    """

    current = (
        prepare_cell28_aliases(
            snapshot
        )
    )

    current[
        "ticker"
    ] = (
        current[
            "ticker"
        ]
        .map(
            normalize_ticker
        )
    )

    current = (
        current[
            current[
                "sector"
            ]
            .isin(
                TARGET_SECTORS
            )
        ]
        .copy()
    )

    # ------------------------------------------------------------------
    # Winsorização P5/P95 dentro do setor
    # ------------------------------------------------------------------

    for metric in ALL_METRICS:

        if metric not in current.columns:

            current[
                metric
            ] = np.nan

        current[
            f"{metric}_winsor"
        ] = (
            current
            .groupby(
                "sector"
            )[
                metric
            ]
            .transform(
                winsorize_cell28
            )
        )

    # ------------------------------------------------------------------
    # Percentis dentro do setor
    # ------------------------------------------------------------------

    for metric in ALL_METRICS:

        winsor_col = (
            f"{metric}_winsor"
        )

        pct_col = (
            f"{metric}_pct"
        )

        lower_is_better = (
            metric
            in
            LOWER_IS_BETTER
        )

        current[
            pct_col
        ] = (
            current
            .groupby(
                "sector"
            )[
                winsor_col
            ]
            .rank(
                pct=True,
                ascending=(
                    False
                    if lower_is_better
                    else True
                ),
                method="average",
            )
        )

    # ------------------------------------------------------------------
    # Três fatores
    # ------------------------------------------------------------------

    for (
        factor_name,
        definition,
    ) in FACTOR_DEFINITIONS.items():

        components = (
            definition[
                "higher"
            ]
            +
            definition[
                "lower"
            ]
        )

        pct_columns = [
            f"{metric}_pct"
            for metric
            in components
        ]

        current[
            f"{factor_name}_components_cell28"
        ] = (
            current[
                pct_columns
            ]
            .notna()
            .sum(
                axis=1
            )
        )

        # CÉLULA 28 = MÉDIA, não mediana.
        current[
            f"{factor_name}_score_cell28"
        ] = (
            current[
                pct_columns
            ]
            .mean(
                axis=1,
                skipna=True,
            )
        )

        current.loc[
            (
                current[
                    f"{factor_name}_components_cell28"
                ]
                <
                MIN_FACTOR_COMPONENTS[
                    factor_name
                ]
            ),
            f"{factor_name}_score_cell28",
        ] = np.nan

    current[
        "factor_used_cell28"
    ] = (
        current[
            "sector"
        ]
        .map(
            {
                "Health Care":
                    "financial_strength_score_cell28",

                "Industrials":
                    "growth_score_cell28",

                "Information Technology":
                    "quality_score_cell28",
            }
        )
    )

    def selected_score(
        row,
    ):

        factor_col = (
            row[
                "factor_used_cell28"
            ]
        )

        if (
            factor_col
            not in
            row.index
        ):

            return np.nan

        return row[
            factor_col
        ]

    current[
        "selected_score_cell28"
    ] = (
        current
        .apply(
            selected_score,
            axis=1,
        )
    )

    current[
        "selection_eligible_cell28"
    ] = (
        current[
            "selected_score_cell28"
        ]
        .notna()
    )

    # Se o snapshot atual já contém ranking_eligible,
    # a Célula 28 também exigia esse filtro.
    if (
        "ranking_eligible"
        in
        current.columns
    ):

        current[
            "selection_eligible_cell28"
        ] = (
            current[
                "selection_eligible_cell28"
            ]
            &
            current[
                "ranking_eligible"
            ]
            .fillna(
                False
            )
            .astype(
                bool
            )
        )

    eligible = (
        current[
            current[
                "selection_eligible_cell28"
            ]
        ]
        .copy()
    )

    eligible = (
        eligible
        .sort_values(
            [
                "sector",
                "selected_score_cell28",
                "ticker",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # Célula 28 usava rank(method="first").
    eligible[
        "sector_rank_cell28"
    ] = (
        eligible
        .groupby(
            "sector"
        )[
            "selected_score_cell28"
        ]
        .rank(
            ascending=False,
            method="first",
        )
        .astype(
            int
        )
    )

    return eligible


# ======================================================================================
# 4. RANKING DA IMPLEMENTAÇÃO ATUAL DO GITHUB
# ======================================================================================

def build_github_full_ranking(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Abre score_sector() do selection.py para todo o universo,
    antes da regra de fronteira.
    """

    base = (
        build_derived_metrics(
            snapshot
        )
    )

    base = (
        ensure_growth_metrics(
            base
        )
    )

    parts = []

    for sector in TARGET_SECTORS:

        sector_df = (
            base[
                base[
                    "sector"
                ]
                ==
                sector
            ]
            .copy()
        )

        scored, score_col = (
            score_sector(
                sector_df,
                sector,
            )
        )

        scored[
            "github_score_column"
        ] = score_col

        scored[
            "selected_score_github"
        ] = (
            scored[
                score_col
            ]
        )

        scored = (
            scored[
                scored[
                    "selected_score_github"
                ]
                .notna()
            ]
            .sort_values(
                [
                    "selected_score_github",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        scored[
            "sector_rank_github"
        ] = (
            np.arange(
                1,
                len(
                    scored
                )
                +
                1,
            )
        )

        parts.append(
            scored
        )

    return (
        pd.concat(
            parts,
            ignore_index=True,
        )
    )


# ======================================================================================
# 5. COMPARAÇÃO
# ======================================================================================

def build_comparison(
    reference: pd.DataFrame,
    github: pd.DataFrame,
) -> pd.DataFrame:

    ref_cols = [
        "sector",
        "ticker",
        "factor_used_cell28",
        "selected_score_cell28",
        "sector_rank_cell28",
    ]

    gh_cols = [
        "sector",
        "ticker",
        "github_score_column",
        "selected_score_github",
        "sector_rank_github",
    ]

    comparison = (
        reference[
            ref_cols
        ]
        .merge(
            github[
                gh_cols
            ],
            on=[
                "sector",
                "ticker",
            ],
            how="outer",
        )
    )

    comparison[
        "score_diff"
    ] = (
        comparison[
            "selected_score_github"
        ]
        -
        comparison[
            "selected_score_cell28"
        ]
    )

    comparison[
        "rank_diff"
    ] = (
        comparison[
            "sector_rank_github"
        ]
        -
        comparison[
            "sector_rank_cell28"
        ]
    )

    comparison[
        "top5_cell28"
    ] = (
        comparison[
            "sector_rank_cell28"
        ]
        <=
        N_PER_SECTOR
    )

    comparison[
        "top5_github_raw"
    ] = (
        comparison[
            "sector_rank_github"
        ]
        <=
        N_PER_SECTOR
    )

    comparison[
        "same_top5_status"
    ] = (
        comparison[
            "top5_cell28"
        ]
        ==
        comparison[
            "top5_github_raw"
        ]
    )

    comparison = (
        comparison
        .sort_values(
            [
                "sector",
                "sector_rank_cell28",
                "sector_rank_github",
                "ticker",
            ],
            na_position="last",
        )
        .reset_index(
            drop=True
        )
    )

    return comparison


# ======================================================================================
# 6. EXECUÇÃO
# ======================================================================================

def run_audit():

    header(
        "AUDITORIA DA SELEÇÃO — CÉLULA 28 x GITHUB"
    )

    print(
        f"\nData de referência              : "
        f"{REFERENCE_DATE.date()}"
    )

    print(
        "Referência metodológica         : "
        "Célula 28"
    )

    print(
        "Auditoria de fronteira          : "
        "Célula 29 / GitHub"
    )

    # ------------------------------------------------------------------
    # UNIVERSO
    # ------------------------------------------------------------------

    header(
        "1. UNIVERSO"
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

    print(
        f"\nEmpresas nos 3 setores          : "
        f"{len(universe):,}"
    )

    counts = (
        universe[
            "sector"
        ]
        .value_counts()
    )

    for sector in TARGET_SECTORS:

        print(
            f"{sector:<31}: "
            f"{int(counts.get(sector, 0))}"
        )

    # ------------------------------------------------------------------
    # FUNDAMENTOS
    # ------------------------------------------------------------------

    header(
        "2. FUNDAMENTOS SEC"
    )

    fundamentals, errors = (
        download_fundamentals(
            universe=universe,
            use_cache=True,
        )
    )

    print(
        f"\nObservações fundamentais        : "
        f"{len(fundamentals):,}"
    )

    if isinstance(
        errors,
        pd.DataFrame,
    ):

        print(
            f"Empresas com erro               : "
            f"{len(errors):,}"
        )

    # ------------------------------------------------------------------
    # SNAPSHOT
    # ------------------------------------------------------------------

    header(
        "3. SNAPSHOT FUNDAMENTAL"
    )

    snapshot = (
        prepare_selection_snapshot(
            universe=universe,
            fundamentals=fundamentals,
            as_of_date=REFERENCE_DATE,
        )
    )

    snapshot[
        "ticker"
    ] = (
        snapshot[
            "ticker"
        ]
        .map(
            normalize_ticker
        )
    )

    print(
        f"\nEmpresas no snapshot            : "
        f"{len(snapshot):,}"
    )

    # ------------------------------------------------------------------
    # CÉLULA 28
    # ------------------------------------------------------------------

    header(
        "4. REPRODUÇÃO INDEPENDENTE DA CÉLULA 28"
    )

    cell28 = (
        reproduce_cell28(
            snapshot
        )
    )

    for sector in TARGET_SECTORS:

        temp = (
            cell28[
                cell28[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "sector_rank_cell28"
            )
            .head(
                10
            )
        )

        print(
            "\n"
            +
            sector.upper()
        )

        print(
            temp[
                [
                    "sector_rank_cell28",
                    "ticker",
                    "factor_used_cell28",
                    "selected_score_cell28",
                ]
            ]
            .round(
                6
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # GITHUB RAW
    # ------------------------------------------------------------------

    header(
        "5. RANKING BRUTO DO selection.py"
    )

    github = (
        build_github_full_ranking(
            snapshot
        )
    )

    for sector in TARGET_SECTORS:

        temp = (
            github[
                github[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "sector_rank_github"
            )
            .head(
                10
            )
        )

        print(
            "\n"
            +
            sector.upper()
        )

        print(
            temp[
                [
                    "sector_rank_github",
                    "ticker",
                    "github_score_column",
                    "selected_score_github",
                ]
            ]
            .round(
                6
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # COMPARAÇÃO
    # ------------------------------------------------------------------

    header(
        "6. COMPARAÇÃO — CÉLULA 28 x GITHUB"
    )

    comparison = (
        build_comparison(
            cell28,
            github,
        )
    )

    boundary = (
        comparison[
            (
                comparison[
                    "sector_rank_cell28"
                ]
                <=
                7
            )
            |
            (
                comparison[
                    "sector_rank_github"
                ]
                <=
                7
            )
        ]
        .copy()
    )

    print(
        boundary[
            [
                "sector",
                "ticker",
                "selected_score_cell28",
                "selected_score_github",
                "score_diff",
                "sector_rank_cell28",
                "sector_rank_github",
                "rank_diff",
                "top5_cell28",
                "top5_github_raw",
                "same_top5_status",
            ]
        ]
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # TOP 5 PURO DA CÉLULA 28
    # ------------------------------------------------------------------

    header(
        "7. TOP 5 — CÉLULA 28"
    )

    cell28_top5 = (
        cell28[
            cell28[
                "sector_rank_cell28"
            ]
            <=
            N_PER_SECTOR
        ]
        .copy()
    )

    for sector in TARGET_SECTORS:

        tickers = (
            cell28_top5[
                cell28_top5[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "sector_rank_cell28"
            )[
                "ticker"
            ]
            .tolist()
        )

        print(
            f"{sector:<31}: "
            +
            ", ".join(
                tickers
            )
        )

    # ------------------------------------------------------------------
    # TOP 5 DO GITHUB SEM FRONTEIRA
    # ------------------------------------------------------------------

    header(
        "8. TOP 5 — GITHUB BRUTO / SEM FRONTEIRA"
    )

    github_top5 = (
        github[
            github[
                "sector_rank_github"
            ]
            <=
            N_PER_SECTOR
        ]
        .copy()
    )

    for sector in TARGET_SECTORS:

        tickers = (
            github_top5[
                github_top5[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "sector_rank_github"
            )[
                "ticker"
            ]
            .tolist()
        )

        print(
            f"{sector:<31}: "
            +
            ", ".join(
                tickers
            )
        )

    # ------------------------------------------------------------------
    # CARTEIRA OPERACIONAL COM FRONTEIRA
    # ------------------------------------------------------------------

    header(
        "9. CARTEIRA OPERACIONAL DO GITHUB — COM REGRA DE FRONTEIRA"
    )

    operational = (
        select_portfolio(
            universe=snapshot,
            use_previous_portfolio=True,
        )
    )

    for sector in TARGET_SECTORS:

        temp = (
            operational[
                operational[
                    "sector"
                ]
                ==
                sector
            ]
            .sort_values(
                "selection_rank"
            )
        )

        print(
            f"{sector:<31}: "
            +
            ", ".join(
                temp[
                    "ticker"
                ]
                .astype(
                    str
                )
                .tolist()
            )
        )

    # ------------------------------------------------------------------
    # FRONTEIRA
    # ------------------------------------------------------------------

    header(
        "10. AUDITORIA DA FRONTEIRA"
    )

    frontier = (
        build_frontier_audit(
            snapshot
        )
    )

    if frontier.empty:

        print(
            "\nSem auditoria de fronteira disponível."
        )

    else:

        print(
            frontier.to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------
    # DIAGNÓSTICO
    # ------------------------------------------------------------------

    header(
        "11. DIAGNÓSTICO FINAL"
    )

    top5_match_by_sector = {}

    for sector in TARGET_SECTORS:

        ref = set(
            cell28_top5.loc[
                cell28_top5[
                    "sector"
                ]
                ==
                sector,
                "ticker",
            ]
        )

        gh = set(
            github_top5.loc[
                github_top5[
                    "sector"
                ]
                ==
                sector,
                "ticker",
            ]
        )

        top5_match_by_sector[
            sector
        ] = (
            ref
            ==
            gh
        )

        print(
            f"{sector:<31}: "
            f"{'TOP 5 IDÊNTICO' if ref == gh else 'DIVERGENTE'}"
        )

    all_top5_match = all(
        top5_match_by_sector.values()
    )

    # Quantas empresas que aparecem em pelo menos um Top 5
    # estão com o mesmo status.
    top5_union = (
        comparison[
            comparison[
                "top5_cell28"
            ]
            |
            comparison[
                "top5_github_raw"
            ]
        ]
    )

    same_status = int(
        top5_union[
            "same_top5_status"
        ]
        .sum()
    )

    total_status = len(
        top5_union
    )

    print(
        f"\nStatus Top 5 coincidentes        : "
        f"{same_status}/{total_status}"
    )

    print(
        f"Top 5 idêntico nos 3 setores    : "
        f"{all_top5_match}"
    )

    # Diferenças de score entre os mesmos tickers.
    common_scores = (
        comparison[
            comparison[
                "selected_score_cell28"
            ]
            .notna()
            &
            comparison[
                "selected_score_github"
            ]
            .notna()
        ]
    )

    if not common_scores.empty:

        mean_abs_score_diff = float(
            common_scores[
                "score_diff"
            ]
            .abs()
            .mean()
        )

        max_abs_score_diff = float(
            common_scores[
                "score_diff"
            ]
            .abs()
            .max()
        )

    else:

        mean_abs_score_diff = np.nan
        max_abs_score_diff = np.nan

    print(
        f"Diferença média absoluta score  : "
        f"{mean_abs_score_diff:.6f}"
    )

    print(
        f"Diferença máxima absoluta score  : "
        f"{max_abs_score_diff:.6f}"
    )

    if all_top5_match:

        status = (
            "APROVADA NO TOP 5 — "
            "VERIFICAR TAMBÉM FIDELIDADE NUMÉRICA DOS SCORES"
        )

    else:

        status = (
            "DIVERGÊNCIA DETECTADA — "
            "selection.py NÃO REPRODUZ INTEGRALMENTE A CÉLULA 28"
        )

    print(
        f"\nSTATUS: {status}"
    )

    # ------------------------------------------------------------------
    # SALVAR
    # ------------------------------------------------------------------

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    cell28.to_csv(
        OUTPUT_REFERENCE_RANKING,
        index=False,
    )

    github.to_csv(
        OUTPUT_GITHUB_RANKING,
        index=False,
    )

    frontier.to_csv(
        OUTPUT_FRONTIER,
        index=False,
    )

    header(
        "12. ARQUIVOS"
    )

    print(
        f"\nComparação      : "
        f"{OUTPUT_COMPARISON}"
    )

    print(
        f"Ranking Célula28: "
        f"{OUTPUT_REFERENCE_RANKING}"
    )

    print(
        f"Ranking GitHub  : "
        f"{OUTPUT_GITHUB_RANKING}"
    )

    print(
        f"Fronteira       : "
        f"{OUTPUT_FRONTIER}"
    )


# ======================================================================================
# 7. EXECUÇÃO
# ======================================================================================

if __name__ == "__main__":

    run_audit()
