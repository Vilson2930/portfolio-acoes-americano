# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# data.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Centralizar a coleta e preparação dos dados usados pelo sistema:
#
#   1. Universo de ações americanas
#   2. Setores
#   3. Preços históricos
#   4. Mapeamento ticker -> CIK da SEC
#   5. SEC Company Facts
#   6. Cache local
#
# IMPORTANTE
# ----------
# Este arquivo NÃO escolhe ações.
# A seleção Top 5 por setor será feita em selection.py.
#
# ======================================================================================


from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

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

DATA_PATH.mkdir(parents=True, exist_ok=True)
CACHE_PATH.mkdir(parents=True, exist_ok=True)


# ======================================================================================
# 2. SESSÃO HTTP
# ======================================================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }
)


# ======================================================================================
# 3. HELPERS
# ======================================================================================

def normalize_ticker(ticker: str) -> str:
    """
    Padroniza ticker para uso interno.
    """

    return (
        str(ticker)
        .upper()
        .strip()
        .replace(".", "-")
    )


def safe_numeric(value):
    """
    Conversão segura para número.
    """

    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ======================================================================================
# 4. DOWNLOAD SEGURO
# ======================================================================================

def request_json(
    url: str,
    retries: int = 3,
    sleep_seconds: float = 0.25,
):
    """
    Requisição JSON com tentativas automáticas.
    """

    last_error = None

    for attempt in range(retries):

        try:

            response = SESSION.get(
                url,
                timeout=30,
            )

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
# 5. MAPA OFICIAL TICKER -> CIK
# ======================================================================================

def get_sec_ticker_map(
    force_refresh: bool = False,
) -> pd.DataFrame:

    cache_file = (
        CACHE_PATH /
        "sec_ticker_map.parquet"
    )

    if (
        cache_file.exists()
        and
        not force_refresh
    ):

        df = pd.read_parquet(
            cache_file
        )

        if not df.empty:
            return df


    data = request_json(
        SEC_TICKER_MAP_URL
    )


    rows = []

    for item in data.values():

        ticker = normalize_ticker(
            item.get("ticker", "")
        )

        cik = item.get(
            "cik_str"
        )

        title = item.get(
            "title"
        )


        if not ticker or cik is None:
            continue


        rows.append(
            {
                "ticker": ticker,

                "cik":
                    str(int(cik))
                    .zfill(10),

                "company_name":
                    title,
            }
        )


    df = pd.DataFrame(
        rows
    )


    if df.empty:

        raise RuntimeError(
            "Mapa ticker/CIK da SEC retornou vazio."
        )


    df = (
        df
        .drop_duplicates(
            subset=["ticker"]
        )
        .sort_values("ticker")
        .reset_index(drop=True)
    )


    df.to_parquet(
        cache_file,
        index=False,
    )


    return df


# ======================================================================================
# 6. UNIVERSO BASE
# ======================================================================================
#
# O mapa da SEC é usado como universo inicial de empresas registradas.
#
# Depois buscamos a classificação setorial necessária.
#
# Não selecionamos Top 5 aqui.
#
# ======================================================================================

def build_base_universe() -> pd.DataFrame:

    sec_map = get_sec_ticker_map()

    universe = (
        sec_map[
            [
                "ticker",
                "cik",
                "company_name",
            ]
        ]
        .copy()
    )


    universe = universe[
        universe["ticker"].notna()
    ]


    universe = universe[
        ~universe["ticker"].str.contains(
            r"[\^/]",
            regex=True,
            na=False,
        )
    ]


    universe = (
        universe
        .drop_duplicates(
            subset=["ticker"]
        )
        .reset_index(drop=True)
    )


    return universe


# ======================================================================================
# 7. CLASSIFICAÇÃO SETORIAL
# ======================================================================================
#
# yfinance é usado aqui para descobrir o setor atual.
#
# Isso NÃO define a seleção.
# Serve apenas para separar o universo nos três setores congelados.
#
# ======================================================================================

def get_ticker_sector(
    ticker: str,
) -> Optional[str]:

    ticker = normalize_ticker(
        ticker
    )

    try:

        obj = yf.Ticker(
            ticker
        )

        info = obj.get_info()

        sector = info.get(
            "sector"
        )

        if sector is None:
            return None

        return str(
            sector
        ).strip()

    except Exception:

        return None


# ======================================================================================
# 8. FILTRAR UNIVERSO POR SETORES
# ======================================================================================

def filter_target_sectors(
    universe: pd.DataFrame,
) -> pd.DataFrame:

    if "sector" not in universe.columns:

        raise ValueError(
            "A coluna 'sector' não existe no universo."
        )


    df = universe[
        universe["sector"].isin(
            SECTORS
        )
    ].copy()


    return (
        df
        .drop_duplicates(
            subset=["ticker"]
        )
        .reset_index(drop=True)
    )


# ======================================================================================
# 9. PREÇOS HISTÓRICOS
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


    # ------------------------------------------------------------------
    # Extrair fechamento
    # ------------------------------------------------------------------

    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):

        if "Close" not in raw.columns.get_level_values(0):

            raise RuntimeError(
                "Coluna Close não encontrada."
            )

        close = raw["Close"].copy()

    else:

        if "Close" not in raw.columns:

            raise RuntimeError(
                "Coluna Close não encontrada."
            )

        close = raw[["Close"]].copy()

        if len(tickers) == 1:
            close.columns = tickers


    # ------------------------------------------------------------------
    # Padronização
    # ------------------------------------------------------------------

    if isinstance(
        close,
        pd.Series,
    ):

        close = close.to_frame(
            name=tickers[0]
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
            .tz_localize(None)
        )

    except Exception:
        pass


    close = (
        close
        .sort_index()
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


    return close


# ======================================================================================
# 10. SEC COMPANY FACTS
# ======================================================================================

def get_company_facts(
    cik: str,
    use_cache: bool = True,
) -> Dict:

    cik = str(
        cik
    ).zfill(10)


    cache_file = (
        CACHE_PATH /
        f"companyfacts_{cik}.json"
    )


    if (
        use_cache
        and
        cache_file.exists()
    ):

        try:

            with open(
                cache_file,
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(
                    file
                )

        except Exception:
            pass


    url = (
        f"{SEC_COMPANY_FACTS_URL}"
        f"CIK{cik}.json"
    )


    # Company Facts usa data.sec.gov.
    headers = {
        "User-Agent":
            SEC_USER_AGENT,

        "Accept-Encoding":
            "gzip, deflate",
    }


    last_error = None


    for attempt in range(3):

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()


            with open(
                cache_file,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    data,
                    file,
                )


            time.sleep(
                0.12
            )

            return data

        except Exception as exc:

            last_error = exc

            time.sleep(
                0.5 * (attempt + 1)
            )


    raise RuntimeError(
        f"SEC Company Facts indisponível "
        f"para CIK {cik}: {last_error}"
    )


# ======================================================================================
# 11. CONCEITOS FUNDAMENTAIS SEC
# ======================================================================================
#
# Mantemos alternativas porque empresas podem usar conceitos XBRL diferentes.
#
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

    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
    ],

    "diluted_eps": [
        "EarningsPerShareDiluted",
    ],

    "diluted_shares": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ],
}


# ======================================================================================
# 12. LOCALIZAR CONCEITO SEC
# ======================================================================================

def find_sec_concept(
    company_facts: Dict,
    concept_names: Iterable[str],
):

    facts = (
        company_facts
        .get("facts", {})
        .get("us-gaap", {})
    )


    for concept in concept_names:

        if concept in facts:

            return (
                concept,
                facts[concept]
            )


    return None, None


# ======================================================================================
# 13. EXTRAIR OBSERVAÇÕES DE UM CONCEITO
# ======================================================================================

def extract_concept_observations(
    company_facts: Dict,
    concept_names: Iterable[str],
    ticker: str,
    metric_name: str,
) -> pd.DataFrame:

    concept, fact = find_sec_concept(
        company_facts,
        concept_names,
    )


    if fact is None:

        return pd.DataFrame()


    units = fact.get(
        "units",
        {}
    )


    rows = []


    for unit_name, observations in units.items():

        for obs in observations:

            value = safe_numeric(
                obs.get("val")
            )


            filed = pd.to_datetime(
                obs.get("filed"),
                errors="coerce",
            )


            end = pd.to_datetime(
                obs.get("end"),
                errors="coerce",
            )


            start = pd.to_datetime(
                obs.get("start"),
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
                        normalize_ticker(ticker),

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
                        obs.get("form"),

                    "fy":
                        obs.get("fy"),

                    "fp":
                        obs.get("fp"),

                    "accn":
                        obs.get("accn"),
                }
            )


    if not rows:

        return pd.DataFrame()


    df = pd.DataFrame(
        rows
    )


    # Somente demonstrações relevantes
    df = df[
        df["form"].isin(
            [
                "10-K",
                "10-Q",
                "10-K/A",
                "10-Q/A",
            ]
        )
    ].copy()


    # ------------------------------------------------------------------
    # Regra point-in-time
    # ------------------------------------------------------------------
    #
    # O dado só passa a existir para o modelo a partir da data FILED.
    #
    # Nunca usamos a data END como se o mercado já soubesse o resultado.
    #
    # ------------------------------------------------------------------

    df["available_date"] = (
        df["filed"]
    )


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
        .reset_index(drop=True)
    )


    return df


# ======================================================================================
# 14. EXTRAIR FUNDAMENTOS DE UMA EMPRESA
# ======================================================================================

def extract_company_fundamentals(
    ticker: str,
    cik: str,
    use_cache: bool = True,
) -> pd.DataFrame:

    company_facts = get_company_facts(
        cik=cik,
        use_cache=use_cache,
    )


    parts = []


    for metric_name, concepts in SEC_CONCEPTS.items():

        temp = extract_concept_observations(
            company_facts=company_facts,
            concept_names=concepts,
            ticker=ticker,
            metric_name=metric_name,
        )


        if not temp.empty:

            parts.append(
                temp
            )


    if not parts:

        return pd.DataFrame()


    result = pd.concat(
        parts,
        ignore_index=True,
    )


    return result


# ======================================================================================
# 15. EXTRAIR FUNDAMENTOS DE VÁRIAS EMPRESAS
# ======================================================================================

def download_fundamentals(
    universe: pd.DataFrame,
    use_cache: bool = True,
) -> pd.DataFrame:

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
        universe.itertuples(index=False),
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
            end=" "
        )


        try:

            temp = extract_company_fundamentals(
                ticker=ticker,
                cik=cik,
                use_cache=use_cache,
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

        fundamentals = pd.DataFrame()


    errors_df = pd.DataFrame(
        errors
    )


    return fundamentals, errors_df


# ======================================================================================
# 16. SNAPSHOT POINT-IN-TIME
# ======================================================================================

def latest_fundamental_snapshot(
    fundamentals: pd.DataFrame,
    as_of_date=None,
) -> pd.DataFrame:

    if fundamentals.empty:

        return pd.DataFrame()


    df = fundamentals.copy()


    df["available_date"] = pd.to_datetime(
        df["available_date"],
        errors="coerce",
    )


    if as_of_date is None:

        as_of_date = pd.Timestamp.today().normalize()

    else:

        as_of_date = pd.Timestamp(
            as_of_date
        )


    # ------------------------------------------------------------------
    # Anti-look-ahead
    # ------------------------------------------------------------------

    df = df[
        df["available_date"]
        <=
        as_of_date
    ].copy()


    if df.empty:

        return pd.DataFrame()


    # Último fato conhecido para cada métrica.
    latest = (
        df
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


    return wide


# ======================================================================================
# 17. AUDITORIA DE LOOK-AHEAD
# ======================================================================================

def audit_lookahead(
    fundamentals: pd.DataFrame,
) -> int:

    if fundamentals.empty:

        return 0


    invalid = (
        pd.to_datetime(
            fundamentals["available_date"],
            errors="coerce",
        )
        <
        pd.to_datetime(
            fundamentals["filed"],
            errors="coerce",
        )
    )


    return int(
        invalid.fillna(False).sum()
    )


# ======================================================================================
# 18. TESTE DO MÓDULO
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


    sec_map = get_sec_ticker_map()


    print(
        f"Tickers SEC: {len(sec_map):,}"
    )


    print(
        "\nConstruindo universo base..."
    )


    universe = build_base_universe()


    print(
        f"Empresas no universo base: "
        f"{len(universe):,}"
    )


    print(
        "\nData layer carregada com sucesso."
    )


    print(
        "\nATENÇÃO:"
    )

    print(
        "A classificação setorial completa será "
        "preparada antes da seleção."
    )

    print(
        "Nenhuma ação foi selecionada neste módulo."
    )
