# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# data.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Centralizar coleta e preparação dos dados:
#
#   1. Universo SEC
#   2. Classificação setorial
#   3. Preços históricos
#   4. Ticker -> CIK
#   5. SEC Company Facts
#   6. Snapshot fundamental point-in-time
#   7. Crescimentos YoY necessários ao selection.py
#   8. Cache local
#
# IMPORTANTE
# ----------
# Este módulo NÃO seleciona ações.
# A seleção Top 5 por setor pertence ao selection.py.
#
# ======================================================================================

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    SECTORS,
    DATA_DIR,
    CACHE_DIR,
    SEC_TICKER_MAP_URL,
    SEC_COMPANY_FACTS_URL,
    SEC_USER_AGENT,
)


# ======================================================================================
# 1. DIRETÓRIOS
# ======================================================================================

DATA_PATH = Path(DATA_DIR)
CACHE_PATH = Path(CACHE_DIR)

DATA_PATH.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================================
# 2. HTTP
# ======================================================================================

DECLARED_SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    SEC_USER_AGENT,
).strip()

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": DECLARED_SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json,text/plain,*/*",
        "Connection": "keep-alive",
    }
)


# ======================================================================================
# 3. HELPERS
# ======================================================================================

def normalize_ticker(ticker: str) -> str:
    return str(ticker).upper().strip().replace(".", "-")


def safe_numeric(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def safe_growth(current, previous):
    current = safe_numeric(current)
    previous = safe_numeric(previous)

    if (
        not np.isfinite(current)
        or not np.isfinite(previous)
        or previous == 0
    ):
        return np.nan

    result = current / previous - 1.0

    return result if np.isfinite(result) else np.nan


# ======================================================================================
# 4. DOWNLOAD JSON
# ======================================================================================

def request_json(
    url: str,
    retries: int = 4,
    sleep_seconds: float = 0.75,
):
    """
    Requisição JSON com backoff e headers adequados para a SEC.
    """

    last_error = None

    for attempt in range(retries):
        try:
            headers = {
                "User-Agent": DECLARED_SEC_USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json,text/plain,*/*",
            }

            response = SESSION.get(
                url,
                headers=headers,
                timeout=30,
            )

            if response.status_code in (403, 429):
                last_error = RuntimeError(
                    f"HTTP {response.status_code} para {url}"
                )
                time.sleep(max(2.0, sleep_seconds * (attempt + 1) * 2))
                continue

            response.raise_for_status()
            return response.json()

        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError(
        f"Falha ao acessar {url}: {last_error}"
    )


# ======================================================================================
# 5. UNIVERSO S&P 500 — BASE OPERACIONAL
# ======================================================================================

SP500_URL = (
    "https://en.wikipedia.org/wiki/"
    "List_of_S%26P_500_companies"
)


def get_sp500_universe(
    force_refresh: bool = False,
) -> pd.DataFrame:

    cache_file = (
        CACHE_PATH
        /
        "sp500_universe.parquet"
    )

    if (
        cache_file.exists()
        and
        not force_refresh
    ):

        try:

            cached = pd.read_parquet(
                cache_file
            )

            required = {
                "ticker",
                "cik",
                "company_name",
                "sector",
            }

            if (
                not cached.empty
                and
                required.issubset(
                    cached.columns
                )
            ):

                return cached

        except Exception:
            pass

    print(
        "Baixando universo atual do S&P 500..."
    )

    # Leitura direta da página.
    # Evita passar response.text bruto ao pandas.
    tables = pd.read_html(
        SP500_URL,
        attrs={
            "id": "constituents"
        },
    )

    if not tables:

        raise RuntimeError(
            "Tabela do S&P 500 não encontrada."
        )

    raw = tables[0].copy()

    required_source = {
        "Symbol",
        "Security",
        "GICS Sector",
        "CIK",
    }

    if not required_source.issubset(
        raw.columns
    ):

        raise RuntimeError(
            "Tabela S&P 500 sem as colunas esperadas."
        )

    universe = pd.DataFrame(
        {
            "ticker":
                raw["Symbol"]
                .map(
                    normalize_ticker
                ),

            "company_name":
                raw["Security"]
                .astype(str)
                .str.strip(),

            "sector":
                raw["GICS Sector"]
                .astype(str)
                .str.strip(),

            "cik":
                pd.to_numeric(
                    raw["CIK"],
                    errors="coerce",
                ),
        }
    )

    universe = universe[
        universe[
            "cik"
        ].notna()
    ].copy()

    universe[
        "cik"
    ] = (
        universe[
            "cik"
        ]
        .astype("int64")
        .astype(str)
        .str.zfill(10)
    )

    universe = (
        universe
        .drop_duplicates(
            subset=[
                "ticker"
            ]
        )
        .sort_values(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )

    if len(universe) < 450:

        raise RuntimeError(
            f"Universo S&P 500 insuficiente: "
            f"{len(universe)} empresas."
        )

    universe.to_parquet(
        cache_file,
        index=False,
    )

    print(
        f"Universo S&P 500 carregado: "
        f"{len(universe)} empresas."
    )

    return universe


# ======================================================================================
# 6. MAPA OFICIAL TICKER -> CIK — FALLBACK / AUDITORIA
# ======================================================================================

def get_sec_ticker_map(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Mantido para auditoria. Não é mais obrigatório
    para construir o universo diário.
    """

    cache_file = CACHE_PATH / "sec_ticker_map.parquet"

    if cache_file.exists() and not force_refresh:
        try:
            cached = pd.read_parquet(cache_file)
            if not cached.empty:
                return cached
        except Exception:
            pass

    data = request_json(SEC_TICKER_MAP_URL)

    rows = []

    for item in data.values():
        ticker = normalize_ticker(item.get("ticker", ""))
        cik = item.get("cik_str")
        title = item.get("title")

        if not ticker or cik is None:
            continue

        rows.append(
            {
                "ticker": ticker,
                "cik": str(int(cik)).zfill(10),
                "company_name": title,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "Mapa ticker/CIK da SEC retornou vazio."
        )

    df = (
        df
        .drop_duplicates(subset=["ticker"])
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    df.to_parquet(cache_file, index=False)

    return df


# ======================================================================================
# 7. UNIVERSO BASE
# ======================================================================================

def build_base_universe() -> pd.DataFrame:
    universe = get_sp500_universe().copy()

    universe = universe[universe["ticker"].notna()]

    universe = universe[
        ~universe["ticker"].str.contains(
            r"[\^/]",
            regex=True,
            na=False,
        )
    ]

    return (
        universe
        .drop_duplicates(subset=["ticker"])
        .reset_index(drop=True)
    )


# ======================================================================================
# 8. SETORES
# ======================================================================================

def get_ticker_sector(
    ticker: str,
) -> Optional[str]:
    """
    Fallback individual. Normalmente não é usado,
    porque o S&P 500 já fornece o GICS sector.
    """

    ticker = normalize_ticker(ticker)

    try:
        info = yf.Ticker(ticker).get_info()
        sector = info.get("sector")
        if sector is None:
            return None
        return str(sector).strip()
    except Exception:
        return None


def load_sector_cache() -> pd.DataFrame:
    cache_file = CACHE_PATH / "ticker_sectors.parquet"

    if not cache_file.exists():
        return pd.DataFrame(columns=["ticker", "sector"])

    try:
        df = pd.read_parquet(cache_file)
        if "ticker" not in df.columns or "sector" not in df.columns:
            return pd.DataFrame(columns=["ticker", "sector"])

        df["ticker"] = df["ticker"].map(normalize_ticker)

        return (
            df
            .drop_duplicates(subset=["ticker"], keep="last")
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame(columns=["ticker", "sector"])


def save_sector_cache(
    sectors: pd.DataFrame,
):
    cache_file = CACHE_PATH / "ticker_sectors.parquet"
    sectors.to_parquet(cache_file, index=False)


def enrich_sectors(
    universe: pd.DataFrame,
    force_refresh: bool = False,
    sleep_seconds: float = 0.05,
) -> pd.DataFrame:
    """
    Usa o setor já existente no universo.
    Só consulta yfinance se algum ticker vier sem setor.
    """

    df = universe.copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)

    if "sector" not in df.columns:
        df["sector"] = np.nan

    missing_mask = (
        df["sector"].isna()
        |
        df["sector"].astype(str).str.strip().isin(
            ["", "nan", "None"]
        )
    )

    if not missing_mask.any():
        return df

    cache = load_sector_cache()

    cache_map = {}
    if not cache.empty:
        cache_map = dict(zip(cache["ticker"], cache["sector"]))

    updated_rows = []

    missing_indices = df.index[missing_mask].tolist()
    total = len(missing_indices)

    for number, idx in enumerate(missing_indices, start=1):
        ticker = df.at[idx, "ticker"]

        if (
            not force_refresh
            and ticker in cache_map
            and pd.notna(cache_map[ticker])
        ):
            sector = cache_map[ticker]
        else:
            sector = get_ticker_sector(ticker)
            time.sleep(sleep_seconds)

        df.at[idx, "sector"] = sector

        updated_rows.append(
            {"ticker": ticker, "sector": sector}
        )

        if number % 25 == 0 or number == total:
            print(
                f"Setores faltantes: {number:,}/{total:,}"
            )

    if updated_rows:
        updated = pd.DataFrame(updated_rows)

        combined_cache = pd.concat(
            [cache, updated],
            ignore_index=True,
        )

        combined_cache = (
            combined_cache
            .drop_duplicates(subset=["ticker"], keep="last")
            .reset_index(drop=True)
        )

        save_sector_cache(combined_cache)

    return df


def filter_target_sectors(
    universe: pd.DataFrame,
) -> pd.DataFrame:

    if "sector" not in universe.columns:
        raise ValueError(
            "A coluna 'sector' não existe no universo."
        )

    df = universe[
        universe["sector"].isin(SECTORS)
    ].copy()

    return (
        df
        .drop_duplicates(subset=["ticker"])
        .reset_index(drop=True)
    )


# ======================================================================================
# 8. PREÇOS HISTÓRICOS
# ======================================================================================

def download_prices(
    tickers: Iterable[str],
    start: str = "2014-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:

    tickers = sorted(
        {
            normalize_ticker(t)
            for t in tickers
            if str(t).strip()
        }
    )

    if not tickers:

        raise ValueError(
            "Nenhum ticker recebido para download."
        )

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )

    if raw.empty:

        raise RuntimeError(
            "Download de preços retornou vazio."
        )

    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):

        if (
            "Close"
            not in
            raw.columns.get_level_values(
                0
            )
        ):

            raise RuntimeError(
                "Coluna Close não encontrada."
            )

        close = (
            raw[
                "Close"
            ]
            .copy()
        )

    else:

        if "Close" not in raw.columns:

            raise RuntimeError(
                "Coluna Close não encontrada."
            )

        close = (
            raw[
                [
                    "Close"
                ]
            ]
            .copy()
        )

        if len(tickers) == 1:

            close.columns = (
                tickers
            )

    if isinstance(
        close,
        pd.Series,
    ):

        close = (
            close.to_frame(
                name=tickers[0]
            )
        )

    close.columns = [
        normalize_ticker(c)
        for c in close.columns
    ]

    close.index = pd.to_datetime(
        close.index
    )

    try:

        close.index = (
            close.index
            .tz_localize(
                None
            )
        )

    except Exception:
        pass

    return (
        close
        .sort_index()
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
    )


# ======================================================================================
# 9. SEC COMPANY FACTS
# ======================================================================================

def get_company_facts(
    cik: str,
    use_cache: bool = True,
) -> Dict:

    cik = str(cik).zfill(10)

    cache_file = (
        CACHE_PATH
        /
        f"companyfacts_{cik}.json"
    )

    if use_cache and cache_file.exists():
        try:
            with open(
                cache_file,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)
        except Exception:
            pass

    url = (
        f"{SEC_COMPANY_FACTS_URL}"
        f"CIK{cik}.json"
    )

    last_error = None

    for attempt in range(4):
        try:
            response = SESSION.get(
                url,
                headers={
                    "User-Agent": DECLARED_SEC_USER_AGENT,
                    "Accept-Encoding": "gzip, deflate",
                    "Accept": "application/json,text/plain,*/*",
                },
                timeout=30,
            )

            if response.status_code in (403, 429):
                last_error = RuntimeError(
                    f"HTTP {response.status_code}"
                )
                time.sleep(2.0 * (attempt + 1))
                continue

            response.raise_for_status()
            data = response.json()

            with open(
                cache_file,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(data, file)

            time.sleep(0.20)
            return data

        except Exception as exc:
            last_error = exc
            time.sleep(1.0 * (attempt + 1))

    raise RuntimeError(
        f"SEC Company Facts indisponível "
        f"para CIK {cik}: {last_error}"
    )


# ======================================================================================
# 10. CONCEITOS SEC
# ======================================================================================

SEC_CONCEPTS = {

    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ],

    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],

    "operating_income": [
        "OperatingIncomeLoss",
    ],

    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
    ],

    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForAdditionsToPropertyPlantAndEquipment",
    ],

    "assets": [
        "Assets",
    ],

    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],

    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],

    # ------------------------------------------------------------------
    # Dívida
    # ------------------------------------------------------------------

    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],

    "short_term_debt": [
        "ShortTermBorrowings",
        "ShortTermDebtCurrent",
        "LongTermDebtCurrent",
        "CurrentPortionOfLongTermDebt",
    ],

    # ------------------------------------------------------------------
    # EPS
    # ------------------------------------------------------------------

    "diluted_eps": [
        "EarningsPerShareDiluted",
    ],

    "diluted_shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
}


# ======================================================================================
# 11. LOCALIZAR CONCEITO SEC
# ======================================================================================

def find_sec_concept(
    company_facts: Dict,
    concept_names: Iterable[str],
):

    facts = (
        company_facts
        .get(
            "facts",
            {}
        )
        .get(
            "us-gaap",
            {}
        )
    )

    for concept in concept_names:

        if concept in facts:

            return (
                concept,
                facts[
                    concept
                ],
            )

    return (
        None,
        None,
    )


# ======================================================================================
# 12. EXTRAIR OBSERVAÇÕES
# ======================================================================================

def extract_concept_observations(
    company_facts: Dict,
    concept_names: Iterable[str],
    ticker: str,
    metric_name: str,
) -> pd.DataFrame:

    concept, fact = (
        find_sec_concept(
            company_facts,
            concept_names,
        )
    )

    if fact is None:

        return pd.DataFrame()

    units = fact.get(
        "units",
        {}
    )

    rows = []

    for (
        unit_name,
        observations
    ) in units.items():

        for obs in observations:

            value = safe_numeric(
                obs.get(
                    "val"
                )
            )

            filed = pd.to_datetime(
                obs.get(
                    "filed"
                ),
                errors="coerce",
            )

            end = pd.to_datetime(
                obs.get(
                    "end"
                ),
                errors="coerce",
            )

            start = pd.to_datetime(
                obs.get(
                    "start"
                ),
                errors="coerce",
            )

            if (
                pd.isna(filed)
                or
                pd.isna(end)
                or
                pd.isna(value)
            ):
                continue

            rows.append(
                {
                    "ticker":
                        normalize_ticker(
                            ticker
                        ),

                    "metric":
                        metric_name,

                    "concept":
                        concept,

                    "unit":
                        unit_name,

                    "value":
                        value,

                    "start":
                        start,

                    "end":
                        end,

                    "filed":
                        filed,

                    "form":
                        obs.get(
                            "form"
                        ),

                    "fy":
                        obs.get(
                            "fy"
                        ),

                    "fp":
                        obs.get(
                            "fp"
                        ),

                    "accn":
                        obs.get(
                            "accn"
                        ),
                }
            )

    if not rows:

        return pd.DataFrame()

    df = pd.DataFrame(
        rows
    )

    df = df[
        df[
            "form"
        ].isin(
            [
                "10-K",
                "10-Q",
                "10-K/A",
                "10-Q/A",
            ]
        )
    ].copy()

    # ------------------------------------------------------------------
    # POINT-IN-TIME
    #
    # O mercado somente conhece o dado depois do FILED.
    # ------------------------------------------------------------------

    df[
        "available_date"
    ] = df[
        "filed"
    ]

    df = (
        df
        .sort_values(
            [
                "available_date",
                "end",
            ]
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "metric",
                "end",
                "filed",
                "value",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ======================================================================================
# 13. FUNDAMENTOS DE UMA EMPRESA
# ======================================================================================

def extract_company_fundamentals(
    ticker: str,
    cik: str,
    use_cache: bool = True,
) -> pd.DataFrame:

    company_facts = (
        get_company_facts(
            cik=cik,
            use_cache=use_cache,
        )
    )

    parts = []

    for (
        metric_name,
        concepts
    ) in SEC_CONCEPTS.items():

        temp = (
            extract_concept_observations(
                company_facts=company_facts,
                concept_names=concepts,
                ticker=ticker,
                metric_name=metric_name,
            )
        )

        if not temp.empty:

            parts.append(
                temp
            )

    if not parts:

        return pd.DataFrame()

    return pd.concat(
        parts,
        ignore_index=True,
    )


# ======================================================================================
# 14. FUNDAMENTOS DE VÁRIAS EMPRESAS
# ======================================================================================

def download_fundamentals(
    universe: pd.DataFrame,
    use_cache: bool = True,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    required = {
        "ticker",
        "cik",
    }

    if not required.issubset(
        universe.columns
    ):

        raise ValueError(
            "Universo precisa conter ticker e cik."
        )

    parts = []
    errors = []

    total = len(
        universe
    )

    for number, row in enumerate(
        universe.itertuples(
            index=False
        ),
        start=1,
    ):

        ticker = normalize_ticker(
            row.ticker
        )

        cik = str(
            row.cik
        ).zfill(10)

        print(
            f"[{number:03d}/{total:03d}] "
            f"{ticker:<7}",
            end=" ",
        )

        try:

            temp = (
                extract_company_fundamentals(
                    ticker=ticker,
                    cik=cik,
                    use_cache=use_cache,
                )
            )

            if temp.empty:

                print(
                    "SEM DADOS"
                )

                errors.append(
                    {
                        "ticker":
                            ticker,

                        "reason":
                            "SEM DADOS",
                    }
                )

                continue

            parts.append(
                temp
            )

            print(
                f"OK ({len(temp):,})"
            )

        except Exception as exc:

            print(
                "ERRO"
            )

            errors.append(
                {
                    "ticker":
                        ticker,

                    "reason":
                        str(exc),
                }
            )

    if parts:

        fundamentals = pd.concat(
            parts,
            ignore_index=True,
        )

    else:

        fundamentals = (
            pd.DataFrame()
        )

    errors_df = pd.DataFrame(
        errors
    )

    return (
        fundamentals,
        errors_df,
    )


# ======================================================================================
# 15. ESCOLHER ÚLTIMA OBSERVAÇÃO CONHECIDA
# ======================================================================================

def _latest_metric_observation(
    df: pd.DataFrame,
) -> pd.Series:

    ordered = (
        df
        .sort_values(
            [
                "available_date",
                "end",
                "filed",
            ]
        )
    )

    return ordered.iloc[
        -1
    ]


# ======================================================================================
# 16. CRESCIMENTO YOY POINT-IN-TIME
# ======================================================================================

def calculate_point_in_time_growth(
    metric_history: pd.DataFrame,
    as_of_date,
) -> float:
    """
    Calcula crescimento YoY usando SOMENTE informações
    disponíveis até as_of_date.

    Procedimento:
        1. filtra available_date <= as_of_date
        2. pega a observação mais recente
        3. procura observação comparável aproximadamente
           um ano antes
        4. calcula current / previous - 1

    A janela de comparação aceita 300–430 dias.
    """

    if metric_history.empty:
        return np.nan

    df = metric_history.copy()

    df[
        "available_date"
    ] = pd.to_datetime(
        df[
            "available_date"
        ],
        errors="coerce",
    )

    df[
        "end"
    ] = pd.to_datetime(
        df[
            "end"
        ],
        errors="coerce",
    )

    as_of_date = pd.Timestamp(
        as_of_date
    )

    df = df[
        df[
            "available_date"
        ]
        <=
        as_of_date
    ].copy()

    df = df[
        df[
            "end"
        ].notna()
        &
        df[
            "value"
        ].notna()
    ].copy()

    if len(df) < 2:
        return np.nan

    current = (
        _latest_metric_observation(
            df
        )
    )

    current_end = pd.Timestamp(
        current[
            "end"
        ]
    )

    # ------------------------------------------------------------------
    # Observações anteriores com aproximadamente 1 ano de diferença
    # ------------------------------------------------------------------

    candidates = df[
        df[
            "end"
        ]
        <
        current_end
    ].copy()

    if candidates.empty:
        return np.nan

    candidates[
        "days_difference"
    ] = (
        current_end
        -
        candidates[
            "end"
        ]
    ).dt.days

    candidates = candidates[
        candidates[
            "days_difference"
        ].between(
            300,
            430,
        )
    ].copy()

    if candidates.empty:
        return np.nan

    candidates[
        "distance_to_year"
    ] = (
        candidates[
            "days_difference"
        ]
        -
        365
    ).abs()

    candidates = (
        candidates
        .sort_values(
            [
                "distance_to_year",
                "available_date",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    previous = candidates.iloc[
        0
    ]

    return safe_growth(
        current[
            "value"
        ],
        previous[
            "value"
        ],
    )


# ======================================================================================
# 17. SNAPSHOT FUNDAMENTAL POINT-IN-TIME
# ======================================================================================

def latest_fundamental_snapshot(
    fundamentals: pd.DataFrame,
    as_of_date=None,
) -> pd.DataFrame:

    if fundamentals.empty:

        return pd.DataFrame()

    df = fundamentals.copy()

    df[
        "available_date"
    ] = pd.to_datetime(
        df[
            "available_date"
        ],
        errors="coerce",
    )

    df[
        "filed"
    ] = pd.to_datetime(
        df[
            "filed"
        ],
        errors="coerce",
    )

    df[
        "end"
    ] = pd.to_datetime(
        df[
            "end"
        ],
        errors="coerce",
    )

    if as_of_date is None:

        as_of_date = (
            pd.Timestamp.today()
            .normalize()
        )

    else:

        as_of_date = pd.Timestamp(
            as_of_date
        )

    # ------------------------------------------------------------------
    # ANTI-LOOK-AHEAD
    # ------------------------------------------------------------------

    known = df[
        df[
            "available_date"
        ]
        <=
        as_of_date
    ].copy()

    if known.empty:

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Último fato conhecido por ticker/métrica
    # ------------------------------------------------------------------

    latest = (
        known
        .sort_values(
            [
                "ticker",
                "metric",
                "available_date",
                "end",
            ]
        )
        .groupby(
            [
                "ticker",
                "metric",
            ],
            as_index=False,
        )
        .tail(1)
    )

    wide = (
        latest
        .pivot(
            index="ticker",
            columns="metric",
            values="value",
        )
        .reset_index()
    )

    wide.columns.name = None

    # ------------------------------------------------------------------
    # Crescimentos necessários ao fator GROWTH
    # ------------------------------------------------------------------

    growth_metrics = {
        "revenue":
            "revenue_growth",

        "diluted_eps":
            "eps_growth",

        "operating_cash_flow":
            "operating_cash_flow_growth",
    }

    growth_rows = []

    for ticker in (
        wide[
            "ticker"
        ].unique()
    ):

        ticker_history = known[
            known[
                "ticker"
            ]
            ==
            ticker
        ]

        row = {
            "ticker":
                ticker,
        }

        for (
            raw_metric,
            output_metric
        ) in growth_metrics.items():

            history = ticker_history[
                ticker_history[
                    "metric"
                ]
                ==
                raw_metric
            ]

            row[
                output_metric
            ] = (
                calculate_point_in_time_growth(
                    metric_history=history,
                    as_of_date=as_of_date,
                )
            )

        growth_rows.append(
            row
        )

    growth_df = pd.DataFrame(
        growth_rows
    )

    wide = wide.merge(
        growth_df,
        on="ticker",
        how="left",
    )

    # ------------------------------------------------------------------
    # Métricas derivadas úteis ao selection.py
    # ------------------------------------------------------------------

    required_numeric = [
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "assets",
        "equity",
        "cash",
        "long_term_debt",
        "short_term_debt",
        "diluted_eps",
    ]

    for column in required_numeric:

        if column not in wide.columns:

            wide[
                column
            ] = np.nan

        wide[
            column
        ] = pd.to_numeric(
            wide[
                column
            ],
            errors="coerce",
        )

    # Dívida total

    debt_components = (
        wide[
            "long_term_debt"
        ].fillna(0)
        +
        wide[
            "short_term_debt"
        ].fillna(0)
    )

    no_debt_data = (
        wide[
            "long_term_debt"
        ].isna()
        &
        wide[
            "short_term_debt"
        ].isna()
    )

    debt_components.loc[
        no_debt_data
    ] = np.nan

    wide[
        "total_debt"
    ] = debt_components

    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------

    def divide_columns(
        numerator: str,
        denominator: str,
    ) -> pd.Series:

        result = (
            wide[
                numerator
            ]
            /
            wide[
                denominator
            ].replace(
                0,
                np.nan,
            )
        )

        return result.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )

    wide[
        "cash_assets"
    ] = divide_columns(
        "cash",
        "assets",
    )

    wide[
        "debt_assets"
    ] = divide_columns(
        "total_debt",
        "assets",
    )

    wide[
        "debt_equity"
    ] = divide_columns(
        "total_debt",
        "equity",
    )

    wide[
        "roa"
    ] = divide_columns(
        "net_income",
        "assets",
    )

    wide[
        "roe"
    ] = divide_columns(
        "net_income",
        "equity",
    )

    wide[
        "operating_margin"
    ] = divide_columns(
        "operating_income",
        "revenue",
    )

    wide[
        "net_margin"
    ] = divide_columns(
        "net_income",
        "revenue",
    )

    return (
        wide
        .sort_values(
            "ticker"
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================================
# 18. PREPARAR SNAPSHOT PARA SELEÇÃO
# ======================================================================================

def prepare_selection_snapshot(
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    as_of_date=None,
) -> pd.DataFrame:
    """
    Une:
        universo
        + setor
        + snapshot fundamental

    Saída pronta para selection.py.
    """

    required = {
        "ticker",
        "sector",
    }

    if not required.issubset(
        universe.columns
    ):

        raise ValueError(
            "Universo precisa conter ticker e sector."
        )

    snapshot = (
        latest_fundamental_snapshot(
            fundamentals=fundamentals,
            as_of_date=as_of_date,
        )
    )

    if snapshot.empty:

        raise RuntimeError(
            "Snapshot fundamental retornou vazio."
        )

    base_columns = [
        column
        for column in [
            "ticker",
            "cik",
            "company_name",
            "sector",
        ]
        if column in universe.columns
    ]

    base = (
        universe[
            base_columns
        ]
        .copy()
    )

    base[
        "ticker"
    ] = base[
        "ticker"
    ].map(
        normalize_ticker
    )

    result = base.merge(
        snapshot,
        on="ticker",
        how="inner",
    )

    result = result[
        result[
            "sector"
        ].isin(
            SECTORS
        )
    ].copy()

    return (
        result
        .drop_duplicates(
            subset=[
                "ticker"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ======================================================================================
# 19. AUDITORIA DE LOOK-AHEAD
# ======================================================================================

def audit_lookahead(
    fundamentals: pd.DataFrame,
) -> int:

    if fundamentals.empty:

        return 0

    available = pd.to_datetime(
        fundamentals[
            "available_date"
        ],
        errors="coerce",
    )

    filed = pd.to_datetime(
        fundamentals[
            "filed"
        ],
        errors="coerce",
    )

    invalid = (
        available
        <
        filed
    )

    return int(
        invalid
        .fillna(False)
        .sum()
    )


# ======================================================================================
# 20. AUDITORIA DO SNAPSHOT DE SELEÇÃO
# ======================================================================================

def audit_selection_snapshot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:

    metrics = [

        # Health Care
        "cash_assets",
        "debt_assets",
        "debt_equity",

        # Industrials
        "revenue_growth",
        "eps_growth",
        "operating_cash_flow_growth",

        # Technology
        "roa",
        "roe",
        "operating_margin",
        "net_margin",
    ]

    rows = []

    total = len(
        snapshot
    )

    for metric in metrics:

        if metric in snapshot.columns:

            available = int(
                snapshot[
                    metric
                ]
                .notna()
                .sum()
            )

        else:

            available = 0

        rows.append(
            {
                "metric":
                    metric,

                "available":
                    available,

                "total":
                    total,

                "coverage":
                    (
                        available
                        /
                        total
                        if total
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ======================================================================================
# 21. TESTE DO MÓDULO
# ======================================================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO ACOES AMERICANO — DATA LAYER"
    )

    print(
        "=" * 100
    )

    print(
        "\nCarregando mapa oficial da SEC..."
    )

    sec_map = (
        get_sec_ticker_map()
    )

    print(
        f"Tickers SEC: "
        f"{len(sec_map):,}"
    )

    print(
        "\nConstruindo universo base..."
    )

    universe = (
        build_base_universe()
    )

    print(
        f"Empresas no universo base: "
        f"{len(universe):,}"
    )

    print(
        "\nMétricas produzidas para o selection.py:"
    )

    print(
        "  Health Care:"
    )
    print(
        "    cash_assets"
    )
    print(
        "    debt_assets"
    )
    print(
        "    debt_equity"
    )

    print(
        "  Industrials:"
    )
    print(
        "    revenue_growth"
    )
    print(
        "    eps_growth"
    )
    print(
        "    operating_cash_flow_growth"
    )

    print(
        "  Information Technology:"
    )
    print(
        "    roa"
    )
    print(
        "    roe"
    )
    print(
        "    operating_margin"
    )
    print(
        "    net_margin"
    )

    print(
        "\nPoint-in-time:"
    )

    print(
        "  available_date = filed"
    )

    print(
        "  crescimento YoY usa somente dados "
        "conhecidos até a data do snapshot"
    )

    print(
        "\nData layer carregada com sucesso."
    )
