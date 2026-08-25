# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_selection.py
# ======================================================================================
#
# OBJETIVO
# --------
# Auditar a seleção fundamental atual do GitHub contra a regra vencedora
# confirmada nas Células 15 e 16 do estudo.
#
# REGRA VENCEDORA
# ----------------
# Health Care              -> Financial Strength
# Industrials              -> Growth
# Information Technology   -> Financial Strength
#
# Financial Strength:
#   cash_assets ↑
#   debt_assets ↓
#   debt_equity ↓
#
# Growth:
#   revenue_growth ↑
#   eps_growth ↑
#   operating_cash_flow_growth ↑
#
# Metodologia:
#   • winsorização P5-P95 dentro do setor
#   • percentis dentro do setor
#   • score = MÉDIA dos componentes válidos
#   • mínimo 2 componentes
#   • ranking decrescente
#   • Top 5 por setor
#
# Este arquivo NÃO altera o motor.
# ======================================================================================

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    SECTORS,
    SELECTION_FACTORS,
    SECTOR_TARGETS,
)

from data import (
    build_base_universe,
    enrich_sectors,
    filter_target_sectors,
    download_fundamentals,
    prepare_selection_snapshot,
)

from selection import (
    build_derived_metrics,
    score_sector,
)


# ======================================================================================
# 1. CONFIGURAÇÃO
# ======================================================================================

REFERENCE_DATE = pd.Timestamp("2026-08-24")

EXPECTED_FACTORS = {
    "Health Care": "financial_strength",
    "Industrials": "growth",
    "Information Technology": "financial_strength",
}

FACTOR_COMPONENTS = {
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
}

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_COMPARISON = OUTPUT_DIR / "audit_selection_comparison.csv"
OUTPUT_REFERENCE = OUTPUT_DIR / "audit_selection_reference.csv"
OUTPUT_GITHUB = OUTPUT_DIR / "audit_selection_github.csv"


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def header(title: str):
    print("\n" + "=" * 145)
    print(title)
    print("=" * 145)


def normalize_ticker(value):
    return (
        str(value)
        .strip()
        .upper()
        .replace(".", "-")
    )


def winsorize_reference(series: pd.Series) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if len(valid) < 10:
        return values

    p05 = valid.quantile(0.05)
    p95 = valid.quantile(0.95)

    return values.clip(
        lower=p05,
        upper=p95,
    )


# ======================================================================================
# 3. REPRODUÇÃO INDEPENDENTE DA REGRA DO ESTUDO
# ======================================================================================

def build_reference_ranking(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    base = build_derived_metrics(
        snapshot
    )

    base["ticker"] = (
        base["ticker"]
        .map(normalize_ticker)
    )

    parts = []

    for sector in SECTORS:

        factor = EXPECTED_FACTORS[
            sector
        ]

        definition = FACTOR_COMPONENTS[
            factor
        ]

        sector_df = (
            base[
                base["sector"] == sector
            ]
            .copy()
        )

        metrics = (
            definition["higher"]
            +
            definition["lower"]
        )

        components = pd.DataFrame(
            index=sector_df.index
        )

        for metric in metrics:

            if metric not in sector_df.columns:
                sector_df[metric] = np.nan

            winsor = winsorize_reference(
                sector_df[metric]
            )

            lower_is_better = (
                metric
                in
                definition["lower"]
            )

            components[metric] = (
                winsor.rank(
                    pct=True,
                    ascending=not lower_is_better,
                    method="average",
                )
            )

        sector_df[
            "reference_components"
        ] = (
            components
            .notna()
            .sum(axis=1)
        )

        sector_df[
            "reference_score"
        ] = (
            components
            .mean(
                axis=1,
                skipna=True,
            )
        )

        sector_df.loc[
            sector_df["reference_components"]
            <
            definition["minimum_components"],
            "reference_score",
        ] = np.nan

        sector_df[
            "reference_factor"
        ] = factor

        sector_df = (
            sector_df[
                sector_df[
                    "reference_score"
                ]
                .notna()
            ]
            .sort_values(
                [
                    "reference_score",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        sector_df[
            "reference_rank"
        ] = np.arange(
            1,
            len(sector_df) + 1,
        )

        parts.append(
            sector_df
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ======================================================================================
# 4. RANKING ATUAL DO GITHUB
# ======================================================================================

def build_github_ranking(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    base = build_derived_metrics(
        snapshot
    )

    base["ticker"] = (
        base["ticker"]
        .map(normalize_ticker)
    )

    parts = []

    for sector in SECTORS:

        sector_df = (
            base[
                base["sector"] == sector
            ]
            .copy()
        )

        scored, score_column = (
            score_sector(
                sector_df,
                sector,
            )
        )

        scored[
            "github_factor"
        ] = SELECTION_FACTORS[
            sector
        ]

        scored[
            "github_score"
        ] = scored[
            score_column
        ]

        scored = (
            scored[
                scored[
                    "github_score"
                ]
                .notna()
            ]
            .sort_values(
                [
                    "github_score",
                    "ticker",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        scored[
            "github_rank"
        ] = np.arange(
            1,
            len(scored) + 1,
        )

        parts.append(
            scored
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ======================================================================================
# 5. COMPARAÇÃO
# ======================================================================================

def compare_rankings(
    reference: pd.DataFrame,
    github: pd.DataFrame,
) -> pd.DataFrame:

    ref = reference[
        [
            "sector",
            "ticker",
            "reference_factor",
            "reference_score",
            "reference_rank",
        ]
    ].copy()

    gh = github[
        [
            "sector",
            "ticker",
            "github_factor",
            "github_score",
            "github_rank",
        ]
    ].copy()

    comparison = (
        ref.merge(
            gh,
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
            "github_score"
        ]
        -
        comparison[
            "reference_score"
        ]
    )

    comparison[
        "rank_diff"
    ] = (
        comparison[
            "github_rank"
        ]
        -
        comparison[
            "reference_rank"
        ]
    )

    comparison[
        "top5_reference"
    ] = (
        comparison[
            "reference_rank"
        ]
        <=
        5
    )

    comparison[
        "top5_github"
    ] = (
        comparison[
            "github_rank"
        ]
        <=
        5
    )

    comparison[
        "top5_same"
    ] = (
        comparison[
            "top5_reference"
        ]
        ==
        comparison[
            "top5_github"
        ]
    )

    comparison[
        "factor_same"
    ] = (
        comparison[
            "reference_factor"
        ]
        ==
        comparison[
            "github_factor"
        ]
    )

    return (
        comparison
        .sort_values(
            [
                "sector",
                "reference_rank",
                "github_rank",
                "ticker",
            ],
            na_position="last",
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 6. EXECUÇÃO
# ======================================================================================

def run_audit():

    header(
        "AUDITORIA DA SELEÇÃO — ESTUDO x GITHUB"
    )

    print(
        f"\nData de referência              : "
        f"{REFERENCE_DATE.date()}"
    )

    print(
        "Regra esperada                 : "
        "FS / GROWTH / FS"
    )

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    header(
        "1. CONFIGURAÇÃO"
    )

    config_ok = True

    for sector in SECTORS:

        actual = SELECTION_FACTORS.get(
            sector
        )

        expected = EXPECTED_FACTORS[
            sector
        ]

        ok = (
            actual
            ==
            expected
        )

        config_ok = (
            config_ok
            and
            ok
        )

        print(
            f"{sector:<31}: "
            f"{actual:<20} "
            f"{'OK' if ok else 'DIVERGENTE'}"
        )

    # ------------------------------------------------------------------
    # UNIVERSO
    # ------------------------------------------------------------------

    header(
        "2. UNIVERSO"
    )

    universe = build_base_universe()
    universe = enrich_sectors(
        universe
    )
    universe = filter_target_sectors(
        universe
    )

    print(
        f"\nEmpresas nos 3 setores          : "
        f"{len(universe):,}"
    )

    counts = (
        universe["sector"]
        .value_counts()
    )

    for sector in SECTORS:

        print(
            f"{sector:<31}: "
            f"{int(counts.get(sector, 0))}"
        )

    # ------------------------------------------------------------------
    # FUNDAMENTOS
    # ------------------------------------------------------------------

    header(
        "3. FUNDAMENTOS SEC"
    )

    fundamentals, errors = (
        download_fundamentals(
            universe=universe,
            use_cache=True,
        )
    )

    print(
        f"\nObservações fundamentais       : "
        f"{len(fundamentals):,}"
    )

    if isinstance(
        errors,
        pd.DataFrame,
    ):
        print(
            f"Empresas com erro              : "
            f"{len(errors):,}"
        )

    # ------------------------------------------------------------------
    # SNAPSHOT
    # ------------------------------------------------------------------

    header(
        "4. SNAPSHOT"
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
        f"\nEmpresas no snapshot           : "
        f"{len(snapshot):,}"
    )

    # ------------------------------------------------------------------
    # RANKINGS
    # ------------------------------------------------------------------

    header(
        "5. RANKING DE REFERÊNCIA DO ESTUDO"
    )

    reference = (
        build_reference_ranking(
            snapshot
        )
    )

    for sector in SECTORS:

        temp = (
            reference[
                reference[
                    "sector"
                ]
                ==
                sector
            ]
            .head(10)
        )

        print(
            f"\n{sector.upper()}"
        )

        print(
            temp[
                [
                    "reference_rank",
                    "ticker",
                    "reference_factor",
                    "reference_score",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    header(
        "6. RANKING ATUAL DO GITHUB"
    )

    github = (
        build_github_ranking(
            snapshot
        )
    )

    for sector in SECTORS:

        temp = (
            github[
                github[
                    "sector"
                ]
                ==
                sector
            ]
            .head(10)
        )

        print(
            f"\n{sector.upper()}"
        )

        print(
            temp[
                [
                    "github_rank",
                    "ticker",
                    "github_factor",
                    "github_score",
                ]
            ]
            .round(6)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # COMPARAÇÃO
    # ------------------------------------------------------------------

    header(
        "7. COMPARAÇÃO NUMÉRICA"
    )

    comparison = compare_rankings(
        reference,
        github,
    )

    boundary = (
        comparison[
            (
                comparison[
                    "reference_rank"
                ]
                <=
                7
            )
            |
            (
                comparison[
                    "github_rank"
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
                "reference_score",
                "github_score",
                "score_diff",
                "reference_rank",
                "github_rank",
                "rank_diff",
                "top5_reference",
                "top5_github",
                "top5_same",
                "factor_same",
            ]
        ]
        .round(6)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # TOP 5
    # ------------------------------------------------------------------

    header(
        "8. TOP 5 POR SETOR"
    )

    all_top5_ok = True

    for sector in SECTORS:

        ref_top5 = (
            reference[
                (
                    reference["sector"]
                    ==
                    sector
                )
                &
                (
                    reference["reference_rank"]
                    <=
                    5
                )
            ]
            .sort_values(
                "reference_rank"
            )["ticker"]
            .tolist()
        )

        gh_top5 = (
            github[
                (
                    github["sector"]
                    ==
                    sector
                )
                &
                (
                    github["github_rank"]
                    <=
                    5
                )
            ]
            .sort_values(
                "github_rank"
            )["ticker"]
            .tolist()
        )

        same = (
            ref_top5
            ==
            gh_top5
        )

        all_top5_ok = (
            all_top5_ok
            and
            same
        )

        print(
            f"\n{sector}"
        )

        print(
            "  Estudo : "
            +
            ", ".join(
                ref_top5
            )
        )

        print(
            "  GitHub : "
            +
            ", ".join(
                gh_top5
            )
        )

        print(
            "  Status : "
            +
            (
                "IDÊNTICO"
                if same
                else
                "DIVERGENTE"
            )
        )

    # ------------------------------------------------------------------
    # DIAGNÓSTICO FINAL
    # ------------------------------------------------------------------

    header(
        "9. DIAGNÓSTICO FINAL"
    )

    common = (
        comparison[
            comparison[
                "reference_score"
            ]
            .notna()
            &
            comparison[
                "github_score"
            ]
            .notna()
        ]
    )

    if not common.empty:

        mean_abs_diff = float(
            common[
                "score_diff"
            ]
            .abs()
            .mean()
        )

        max_abs_diff = float(
            common[
                "score_diff"
            ]
            .abs()
            .max()
        )

    else:

        mean_abs_diff = np.nan
        max_abs_diff = np.nan

    all_factor_ok = bool(
        comparison[
            comparison[
                "reference_factor"
            ]
            .notna()
            &
            comparison[
                "github_factor"
            ]
            .notna()
        ][
            "factor_same"
        ]
        .all()
    )

    print(
        f"\nConfiguração FS/Growth/FS        : "
        f"{config_ok}"
    )

    print(
        f"Fatores iguais                   : "
        f"{all_factor_ok}"
    )

    print(
        f"Top 5 idêntico nos 3 setores     : "
        f"{all_top5_ok}"
    )

    print(
        f"Diferença média absoluta score   : "
        f"{mean_abs_diff:.10f}"
    )

    print(
        f"Diferença máxima absoluta score  : "
        f"{max_abs_diff:.10f}"
    )

    if (
        config_ok
        and
        all_factor_ok
        and
        all_top5_ok
        and
        (
            pd.isna(max_abs_diff)
            or
            max_abs_diff
            <
            1e-10
        )
    ):

        status = (
            "AUDITORIA APROVADA — "
            "SELEÇÃO DO GITHUB REPRODUZ A REGRA VENCEDORA DO ESTUDO."
        )

    elif (
        config_ok
        and
        all_factor_ok
        and
        all_top5_ok
    ):

        status = (
            "TOP 5 APROVADO — "
            "EXISTEM DIFERENÇAS NUMÉRICAS A AUDITAR."
        )

    else:

        status = (
            "AUDITORIA NÃO APROVADA — "
            "EXISTEM DIVERGÊNCIAS NA SELEÇÃO."
        )

    print(
        "\nSTATUS: "
        +
        status
    )

    # ------------------------------------------------------------------
    # SALVAR
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

    header(
        "10. ARQUIVOS"
    )

    print(
        f"\nComparação : "
        f"{OUTPUT_COMPARISON}"
    )

    print(
        f"Referência : "
        f"{OUTPUT_REFERENCE}"
    )

    print(
        f"GitHub     : "
        f"{OUTPUT_GITHUB}"
    )


# ======================================================================================
# 7. EXECUÇÃO DIRETA
# ======================================================================================

if __name__ == "__main__":

    run_audit()
