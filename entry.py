# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# entry.py — VERSÃO CORRIGIDA / ALINHADA ÀS CÉLULAS 33B, 37 E 41
# ======================================================================================
#
# CORREÇÃO CENTRAL
# ----------------
# O signal_percentile NÃO é o rank das 5 ações atuais.
# Ele é o percentil do final_signal_score atual contra TODO o histórico anterior
# daquele SETOR, exatamente como na Célula 41.
#
# Componentes também são normalizados point-in-time contra o PRÓPRIO histórico
# de cada ação:
#
#   Valuation     -> 5 múltiplos; menor é melhor; mediana; mínimo 2
#   Desconto      -> 4 métricas; menor valor bruto = mais descontado;
#                    mediana; mínimo 2
#   Fundamentos   -> 7 componentes; maior é melhor; mediana; mínimo 3
#
# Arquitetura:
#   Health Care             10% valuation + 80% desconto + 10% fundamentos
#   Industrials              0% valuation + 20% desconto + 80% fundamentos
#   Information Technology   Momentum 6M + 12M
#
# Industrials nunca recebe ENTRADA FORTE.
# ======================================================================================

from __future__ import annotations

from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd

from config import SECTORS

HEALTH_CARE = "Health Care"
INDUSTRIALS = "Industrials"
TECHNOLOGY = "Information Technology"

MIN_HISTORY = 24
MIN_VALUATION_COMPONENTS = 2
MIN_DISCOUNT_COMPONENTS = 2
MIN_FUNDAMENTAL_COMPONENTS = 3

VALUATION_METRICS = ["pe", "pb", "ps", "p_ocf", "p_fcf"]
DISCOUNT_METRICS = [
    "drawdown_52w",
    "drawdown_3y",
    "distance_ma200",
    "price_position_3y",
]
FUNDAMENTAL_COMPONENTS = [
    "revenue_growth_yoy",
    "net_income_growth_yoy",
    "operating_cash_flow_growth_yoy",
    "diluted_eps_growth_yoy",
    "net_margin",
    "ocf_margin",
    "fcf_margin",
]

ENTRY_METHODS = {
    HEALTH_CARE: "10% Valuation + 80% Desconto + 10% Fundamentos",
    INDUSTRIALS: "20% Desconto + 80% Fundamentos",
    TECHNOLOGY: "Momentum 6M + 12M",
}

SECTOR_VALIDATION_STATUS = {
    HEALTH_CARE: "APROVADO",
    INDUSTRIALS: "CONDICIONAL",
    TECHNOLOGY: "REGRA ALTERNATIVA APROVADA",
}


# ======================================================================================
# HELPERS
# ======================================================================================

def _num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def _safe_div(a, b):
    a = _num(a)
    b = _num(b)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return np.nan
    value = a / b
    return value if np.isfinite(value) else np.nan


def _midrank_percentile(current: float, history: Iterable[float]) -> float:
    """Percentil = (n menores + 0.5*n iguais) / n."""
    current = _num(current)
    hist = np.asarray(
        [float(x) for x in history if np.isfinite(_num(x))],
        dtype=float,
    )
    if not np.isfinite(current) or len(hist) < MIN_HISTORY:
        return np.nan
    less = np.sum(hist < current)
    equal = np.sum(hist == current)
    return float((less + 0.5 * equal) / len(hist))


def _expanding_score(
    values: pd.Series,
    higher_is_better: bool,
    min_history: int = MIN_HISTORY,
) -> pd.Series:
    """
    Score point-in-time. O valor atual nunca participa do próprio benchmark.
    """
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan)
    history = []

    for i, current in enumerate(arr):
        if np.isfinite(current) and len(history) >= min_history:
            hist = np.asarray(history, dtype=float)
            less = np.sum(hist < current)
            equal = np.sum(hist == current)
            pct = (less + 0.5 * equal) / len(hist)
            out[i] = pct if higher_is_better else 1.0 - pct

        if np.isfinite(current):
            history.append(float(current))

    return pd.Series(out, index=values.index, dtype=float)


def _weighted_score(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Igual à Célula 41: componentes ausentes têm peso removido do denominador.
    """
    cols = list(weights)
    components = df[cols].apply(pd.to_numeric, errors="coerce")
    w = pd.Series(weights, dtype=float)

    available = components.notna()
    effective_weights = available.mul(w, axis=1)

    numerator = (
        components.fillna(0)
        .mul(effective_weights, axis=1)
        .sum(axis=1)
    )
    denominator = effective_weights.sum(axis=1).replace(0, np.nan)
    return numerator / denominator


def _label(score, names):
    score = _num(score)
    if not np.isfinite(score):
        return "N/D"
    if score >= 0.80:
        return names[0]
    if score >= 0.65:
        return names[1]
    if score >= 0.40:
        return names[2]
    if score >= 0.20:
        return names[3]
    return names[4]


def valuation_status(score):
    return _label(
        score,
        ["MUITO BARATA", "BARATA", "NEUTRA", "CARA", "MUITO CARA"],
    )


def discount_status(score):
    return _label(
        score,
        ["MUITO ALTO", "ALTO", "MÉDIO", "BAIXO", "SEM DESCONTO"],
    )


def fundamental_status(score):
    return _label(
        score,
        ["MUITO FORTES", "FORTES", "PRESERVADOS", "ENFRAQUECENDO", "FRACOS"],
    )


def classify_entry(signal_percentile: float, sector: str) -> str:
    pct = _num(signal_percentile)

    if not np.isfinite(pct):
        return "AGUARDAR"

    if pct >= 0.75:
        signal = "ENTRADA FORTE"
    elif pct >= 0.50:
        signal = "ENTRADA"
    elif pct >= 0.25:
        signal = "AGUARDAR"
    else:
        signal = "NÃO COMPRAR AGORA"

    if sector == INDUSTRIALS and signal == "ENTRADA FORTE":
        signal = "ENTRADA"

    return signal


# ======================================================================================
# PREÇOS — HISTÓRICO DE DESCONTO + MOMENTUM
# ======================================================================================

def _build_daily_price_indicators(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    parts = []

    for ticker in tickers:
        if ticker not in prices.columns:
            continue

        s = pd.to_numeric(prices[ticker], errors="coerce").dropna().sort_index()
        if s.empty:
            continue

        temp = pd.DataFrame({"date": s.index, "ticker": ticker, "price": s.values})

        temp["high_52w"] = temp["price"].rolling(252, min_periods=126).max()
        temp["drawdown_52w"] = temp["price"] / temp["high_52w"] - 1.0

        temp["high_3y"] = temp["price"].rolling(756, min_periods=252).max()
        temp["low_3y"] = temp["price"].rolling(756, min_periods=252).min()
        temp["drawdown_3y"] = temp["price"] / temp["high_3y"] - 1.0

        temp["ma200"] = temp["price"].rolling(200, min_periods=150).mean()
        temp["distance_ma200"] = temp["price"] / temp["ma200"] - 1.0

        range_3y = (temp["high_3y"] - temp["low_3y"]).replace(0, np.nan)
        temp["price_position_3y"] = (
            (temp["price"] - temp["low_3y"]) / range_3y
        )

        temp["momentum_6m"] = temp["price"] / temp["price"].shift(126) - 1.0
        temp["momentum_12m"] = temp["price"] / temp["price"].shift(252) - 1.0

        parts.append(temp)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)


# ======================================================================================
# FUNDAMENTOS POINT-IN-TIME
# ======================================================================================

def _prepare_facts(
    fundamentals_history: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """
    Prepara os fatos SEC exatamente na arquitetura usada pela Célula 33B.

    REGRA DO ESTUDO
    ---------------
    Fluxos:
        revenue
        net_income
        operating_cash_flow
        capex
        diluted_eps
        diluted_shares

    usam somente demonstrações ANUAIS:
        10-K / 10-K/A
        20-F / 20-F/A
        40-F / 40-F/A

    e, quando START existe, duração entre 250 e 450 dias.

    Estoques/instantâneos podem usar:
        10-K / 10-K/A
        10-Q / 10-Q/A
        20-F / 20-F/A
        40-F / 40-F/A

    Isso é importante porque a Célula 37 recebeu a base corrigida da
    Célula 33B. Portanto, misturar trimestre/YTD com anual altera os
    crescimentos, margens, percentis e principalmente Industrials.

    Nenhum peso ou regra de entrada é alterado aqui.
    """

    required = {
        "ticker",
        "metric",
        "value",
        "available_date",
        "end",
    }

    if (
        fundamentals_history is None
        or
        fundamentals_history.empty
    ):
        raise RuntimeError(
            "Histórico fundamental vazio."
        )

    if not required.issubset(
        fundamentals_history.columns
    ):
        raise RuntimeError(
            "Histórico fundamental precisa conter: "
            f"{sorted(required)}"
        )

    facts = fundamentals_history.copy()

    facts["ticker"] = (
        facts["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    facts = facts[
        facts["ticker"].isin(
            tickers
        )
    ].copy()

    facts["available_date"] = pd.to_datetime(
        facts["available_date"],
        errors="coerce",
    )

    facts["end"] = pd.to_datetime(
        facts["end"],
        errors="coerce",
    )

    if "filed" in facts.columns:

        facts["filed"] = pd.to_datetime(
            facts["filed"],
            errors="coerce",
        )

    else:

        # data.py define available_date = filed.
        facts["filed"] = facts[
            "available_date"
        ]

    if "start" in facts.columns:

        facts["start"] = pd.to_datetime(
            facts["start"],
            errors="coerce",
        )

    else:

        facts["start"] = pd.NaT

    facts["value"] = pd.to_numeric(
        facts["value"],
        errors="coerce",
    )

    facts = facts.dropna(
        subset=[
            "ticker",
            "metric",
            "value",
            "available_date",
            "end",
        ]
    ).copy()

    # ------------------------------------------------------------------
    # MESMAS FAMÍLIAS DA CÉLULA 33B
    # ------------------------------------------------------------------

    flow_metrics = {
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capex",
        "diluted_eps",
        "diluted_shares",
    }

    # Célula 33B:
    # fluxos usam formulários anuais;
    # estoques/instantâneos também podem usar trimestrais.
    annual_forms = {
        "10-K",
        "10-K/A",
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
    }

    instant_forms = {
        "10-K",
        "10-K/A",
        "10-Q",
        "10-Q/A",
        "20-F",
        "20-F/A",
        "40-F",
        "40-F/A",
    }

    if "form" in facts.columns:

        facts["form"] = (
            facts["form"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        flow_mask = facts[
            "metric"
        ].isin(
            flow_metrics
        )

        instant_mask = ~flow_mask

        keep_flow = (
            flow_mask
            &
            facts[
                "form"
            ].isin(
                annual_forms
            )
        )

        keep_instant = (
            instant_mask
            &
            facts[
                "form"
            ].isin(
                instant_forms
            )
        )

        facts = facts[
            keep_flow
            |
            keep_instant
        ].copy()

    # ------------------------------------------------------------------
    # DURAÇÃO ANUAL — CÉLULA 33B: 250–450 dias
    # ------------------------------------------------------------------

    flow_mask = facts[
        "metric"
    ].isin(
        flow_metrics
    )

    facts[
        "duration_days"
    ] = (
        facts[
            "end"
        ]
        -
        facts[
            "start"
        ]
    ).dt.days

    # Célula 33B: fluxo anual só é aceito com duração entre 250 e 450 dias.
    # Não há fallback para duração ausente.
    valid_duration = facts[
        "duration_days"
    ].between(
        250,
        450,
    )

    facts = facts[
        (~flow_mask)
        |
        valid_duration
    ].copy()

    # ------------------------------------------------------------------
    # DEDUPLICAÇÃO / ORDENAÇÃO POINT-IN-TIME
    # ------------------------------------------------------------------

    sort_cols = [
        "ticker",
        "metric",
        "end",
        "available_date",
    ]

    if "concept" in facts.columns:
        sort_cols.append(
            "concept"
        )

    facts = (
        facts
        .sort_values(
            sort_cols
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "metric",
                "end",
                "available_date",
                "value",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return facts


def _latest_metric(
    history: pd.DataFrame,
    metric: str,
    snapshot: pd.Timestamp,
):
    """
    Último fato conhecido no snapshot.

    Igual ao princípio da Célula 33B:
        filed/available_date <= snapshot
        period_end/end       <= snapshot

    Entre os fatos elegíveis, o período mais recente prevalece;
    em empate, prevalece o filing mais recente.
    """

    temp = history[
        (
            history[
                "metric"
            ]
            ==
            metric
        )
        &
        (
            history[
                "available_date"
            ]
            <=
            snapshot
        )
        &
        (
            history[
                "end"
            ]
            <=
            snapshot
        )
    ].copy()

    if temp.empty:
        return None

    if "priority" not in temp.columns:
        temp["priority"] = 0

    temp["priority"] = pd.to_numeric(
        temp["priority"],
        errors="coerce",
    ).fillna(999999)

    temp = temp.sort_values(
        [
            "end",
            "available_date",
            "priority",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return temp.iloc[-1]


def _growth_yoy(history: pd.DataFrame, metric: str, snapshot: pd.Timestamp) -> float:
    current = _latest_metric(history, metric, snapshot)
    if current is None:
        return np.nan

    current_end = pd.Timestamp(current["end"])
    candidates = history[
        (history["metric"] == metric)
        & (history["available_date"] <= snapshot)
        & (history["end"] < current_end)
    ].copy()

    if candidates.empty:
        return np.nan

    candidates["days_diff"] = (current_end - candidates["end"]).dt.days
    candidates = candidates[candidates["days_diff"].between(300, 430)].copy()

    # Quando possível, prioriza o mesmo fiscal period.
    if "fp" in candidates.columns and "fp" in current.index:
        same_fp = candidates[candidates["fp"] == current.get("fp")]
        if not same_fp.empty:
            candidates = same_fp

    if candidates.empty:
        return np.nan

    candidates["distance"] = (candidates["days_diff"] - 365).abs()
    previous = candidates.sort_values(
        ["distance", "available_date"],
        ascending=[True, False],
    ).iloc[0]

    return _safe_div(current["value"], previous["value"]) - 1.0


def _snapshot_fundamentals(
    ticker_history: pd.DataFrame,
    snapshot: pd.Timestamp,
) -> Dict[str, float]:
    """
    Snapshot point-in-time dos fatos SEC.

    CÉLULA 33B
    ----------
    A base fundamental é reconstruída com:
        available_date <= snapshot
        end <= snapshot

    Para Market Cap:
        1. shares_outstanding
        2. diluted_shares como fallback

    A função também mantém os period_end usados depois pela lógica
    de crescimento da Célula 37.
    """

    raw = {}
    period_end = {}

    metrics = [
        "revenue",
        "net_income",
        "operating_income",
        "operating_cash_flow",
        "capex",
        "assets",
        "equity",
        "cash",
        "long_term_debt",
        "short_term_debt",
        "diluted_eps",
        "diluted_shares",
        "shares_outstanding",
        "shares",
    ]

    for metric in metrics:
        row = _latest_metric(ticker_history, metric, snapshot)

        if row is None:
            raw[metric] = np.nan
            period_end[metric] = pd.NaT
        else:
            raw[metric] = _num(row["value"])
            period_end[metric] = pd.to_datetime(
                row.get("end", pd.NaT), errors="coerce"
            )

    shares_outstanding = _num(raw.get("shares_outstanding"))
    legacy_shares = _num(raw.get("shares"))

    if (not np.isfinite(shares_outstanding) or shares_outstanding <= 0):
        if np.isfinite(legacy_shares) and legacy_shares > 0:
            shares_outstanding = legacy_shares

    raw["shares_outstanding"] = shares_outstanding

    revenue = raw["revenue"]
    net_income = raw["net_income"]
    ocf = raw["operating_cash_flow"]
    capex = raw["capex"]

    fcf = (
        ocf - capex
        if (np.isfinite(ocf) and np.isfinite(capex))
        else np.nan
    )

    return {
        **raw,
        "revenue_period_end": period_end["revenue"],
        "net_income_period_end": period_end["net_income"],
        "operating_cash_flow_period_end": period_end["operating_cash_flow"],
        "diluted_eps_period_end": period_end["diluted_eps"],
        "free_cash_flow": fcf,
        "net_margin": _safe_div(net_income, revenue),
        "ocf_margin": _safe_div(ocf, revenue),
        "fcf_margin": _safe_div(fcf, revenue),
    }

def _apply_cell37_growth_logic(
    hist: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduz a Célula 37:

      1. para cada ticker e métrica;
      2. cria uma série de períodos fundamentais ÚNICOS;
      3. mantém a primeira data em que cada period_end apareceu;
      4. previous_value = shift(1);
      5. growth = current / previous - 1;
      6. previous_value <= 0 => NaN;
      7. mapeia o growth do período de volta aos snapshots mensais.

    Isso é deliberadamente diferente de procurar "aprox. 1 ano atrás".
    """

    out = hist.copy()

    growth_specs = {
        "revenue":
            "revenue_period_end",

        "net_income":
            "net_income_period_end",

        "operating_cash_flow":
            "operating_cash_flow_period_end",

        "diluted_eps":
            "diluted_eps_period_end",
    }

    for ticker, idx in out.groupby(
        "ticker"
    ).groups.items():

        temp = (
            out.loc[list(idx)]
            .sort_values(
                "snapshot_date"
            )
            .copy()
        )

        for value_col, period_col in growth_specs.items():

            growth_col = (
                f"{value_col}_growth_yoy"
            )

            if (
                value_col not in temp.columns
                or
                period_col not in temp.columns
            ):

                out.loc[
                    temp.index,
                    growth_col,
                ] = np.nan

                continue

            unique_series = (
                temp[
                    [
                        "snapshot_date",
                        value_col,
                        period_col,
                    ]
                ]
                .dropna(
                    subset=[
                        value_col,
                        period_col,
                    ]
                )
                .copy()
            )

            unique_series[
                period_col
            ] = pd.to_datetime(
                unique_series[
                    period_col
                ],
                errors="coerce",
            )

            unique_series = (
                unique_series
                .dropna(
                    subset=[
                        period_col
                    ]
                )
                .sort_values(
                    [
                        period_col,
                        "snapshot_date",
                    ]
                )
                .drop_duplicates(
                    subset=[
                        period_col
                    ],
                    keep="first",
                )
                .sort_values(
                    period_col
                )
            )

            unique_series[
                "previous_value"
            ] = (
                pd.to_numeric(
                    unique_series[
                        value_col
                    ],
                    errors="coerce",
                )
                .shift(1)
            )

            current_values = pd.to_numeric(
                unique_series[
                    value_col
                ],
                errors="coerce",
            )

            unique_series[
                "growth"
            ] = (
                current_values
                /
                unique_series[
                    "previous_value"
                ]
                -
                1.0
            )

            unique_series.loc[
                (
                    unique_series[
                        "previous_value"
                    ]
                    <=
                    0
                ),
                "growth",
            ] = np.nan

            growth_map = (
                unique_series
                .set_index(
                    period_col
                )[
                    "growth"
                ]
                .to_dict()
            )

            mapped = (
                pd.to_datetime(
                    temp[
                        period_col
                    ],
                    errors="coerce",
                )
                .map(
                    growth_map
                )
            )

            out.loc[
                temp.index,
                growth_col,
            ] = mapped.values

    return out

def _valuation_raw(
    price: float,
    f: Dict[str, float],
) -> Dict[str, float]:
    """
    Reproduz os múltiplos da Célula 33B.

    Regras:
        1. Market Cap = preço × shares_outstanding
        2. diluted_shares apenas como fallback
        3. Market Cap válido somente quando:
             MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP
             e shares_ratio entre 0,50 e 2,00
        4. P/B, P/S, P/OCF e P/FCF são invalidados quando
           a base de Market Cap é suspeita.
        5. 0 < múltiplo <= 500
    """

    MIN_MARKET_CAP = 50_000_000
    MAX_MARKET_CAP = 20_000_000_000_000
    MIN_SHARE_RATIO = 0.50
    MAX_SHARE_RATIO = 2.00
    MAX_MULTIPLE = 500.0

    shares_outstanding = _num(
        f.get("shares_outstanding")
    )

    diluted_shares = _num(
        f.get("diluted_shares")
    )

    if (
        np.isfinite(shares_outstanding)
        and shares_outstanding > 0
    ):
        shares_for_market_cap = shares_outstanding
        shares_source = "shares_outstanding"

    elif (
        np.isfinite(diluted_shares)
        and diluted_shares > 0
    ):
        shares_for_market_cap = diluted_shares
        shares_source = "diluted_shares_fallback"

    else:
        shares_for_market_cap = np.nan
        shares_source = "missing"

    market_cap = (
        price * shares_for_market_cap
        if (
            np.isfinite(price)
            and np.isfinite(shares_for_market_cap)
            and shares_for_market_cap > 0
        )
        else np.nan
    )

    # ------------------------------------------------------------------
    # SANIDADE ENTRE SHARES OUTSTANDING E DILUTED SHARES
    # ------------------------------------------------------------------

    if (
        np.isfinite(shares_outstanding)
        and shares_outstanding > 0
        and np.isfinite(diluted_shares)
        and diluted_shares > 0
    ):
        shares_ratio = (
            shares_outstanding
            /
            diluted_shares
        )
    else:
        shares_ratio = np.nan

    shares_ratio_ok = (
        (
            np.isfinite(shares_ratio)
            and MIN_SHARE_RATIO <= shares_ratio <= MAX_SHARE_RATIO
        )
        or
        not np.isfinite(shares_ratio)
    )

    market_cap_range_ok = (
        np.isfinite(market_cap)
        and MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP
    )

    # Exatamente como na Célula 33B:
    # market_cap_valid depende também da sanidade do shares_ratio.
    market_cap_valid = bool(
        market_cap_range_ok
        and shares_ratio_ok
    )

    eps = _num(
        f.get("diluted_eps")
    )

    equity = _num(
        f.get("equity")
    )

    revenue = _num(
        f.get("revenue")
    )

    ocf = _num(
        f.get("operating_cash_flow")
    )

    fcf = _num(
        f.get("free_cash_flow")
    )

    def positive_ratio(
        numerator,
        denominator,
    ):

        numerator = _num(numerator)
        denominator = _num(denominator)

        if (
            not np.isfinite(numerator)
            or not np.isfinite(denominator)
            or numerator <= 0
            or denominator <= 0
        ):
            return np.nan

        value = numerator / denominator

        if (
            not np.isfinite(value)
            or value <= 0
            or value > MAX_MULTIPLE
        ):
            return np.nan

        return float(value)

    # P/E é direto preço / EPS e NÃO depende do market_cap_valid.
    pe = positive_ratio(
        price,
        eps,
    )

    # Market-cap multiples.
    pb = positive_ratio(
        market_cap,
        equity,
    )

    ps = positive_ratio(
        market_cap,
        revenue,
    )

    p_ocf = positive_ratio(
        market_cap,
        ocf,
    )

    p_fcf = positive_ratio(
        market_cap,
        fcf,
    )

    # ------------------------------------------------------------------
    # CÉLULA 33B — INVALIDAR MÚLTIPLOS DE MARKET CAP
    # ------------------------------------------------------------------

    if not market_cap_valid:
        pb = np.nan
        ps = np.nan
        p_ocf = np.nan
        p_fcf = np.nan

    return {
        "pe": pe,
        "pb": pb,
        "ps": ps,
        "p_ocf": p_ocf,
        "p_fcf": p_fcf,

        "shares_for_market_cap":
            shares_for_market_cap,

        "shares_source":
            shares_source,

        "shares_ratio":
            shares_ratio,

        "shares_ratio_ok":
            shares_ratio_ok,

        "market_cap":
            market_cap,

        "market_cap_valid":
            market_cap_valid,
    }

# ======================================================================================
# CONSTRUIR HISTÓRICO MENSAL DO ENTRY SCORE
# ======================================================================================

def build_entry_signal_history(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals_history: pd.DataFrame,
    as_of_date=None,
    start_date: str = "2015-01-31",
) -> pd.DataFrame:

    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()
    else:
        as_of_date = pd.Timestamp(as_of_date).normalize()

    p = portfolio.copy()
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()
    sector_map = p.set_index("ticker")["sector"].to_dict()
    tickers = p["ticker"].tolist()

    # Datas mensais + snapshot atual.
    monthly = list(pd.date_range(start=start_date, end=as_of_date, freq="ME"))
    if not monthly or monthly[-1].normalize() != as_of_date:
        monthly.append(as_of_date)
    snapshot_dates = sorted(pd.DatetimeIndex(monthly).unique())

    daily = _build_daily_price_indicators(prices, tickers)
    if daily.empty:
        raise RuntimeError("Não foi possível construir indicadores de preço.")

    facts = _prepare_facts(fundamentals_history, tickers)

    rows = []

    for ticker in tickers:
        price_hist = daily[daily["ticker"] == ticker].sort_values("date").copy()
        fact_hist = facts[facts["ticker"] == ticker].copy()

        for snapshot in snapshot_dates:
            # A Célula 33B SEMPRE cria a linha ticker x snapshot.
            # Preço é associado depois, para trás, com tolerância de 10 dias.
            pr = None

            if not price_hist.empty:
                price_rows = price_hist[price_hist["date"] <= snapshot]

                if not price_rows.empty:
                    candidate = price_rows.iloc[-1]
                    age_days = (
                        snapshot
                        -
                        pd.Timestamp(candidate["date"])
                    ).days

                    if age_days <= 10:
                        pr = candidate

            market_price = (
                _num(pr["price"])
                if pr is not None
                else np.nan
            )

            f = _snapshot_fundamentals(fact_hist, snapshot)
            valuation = _valuation_raw(market_price, f)

            price_components = {
                k: (
                    _num(pr[k])
                    if pr is not None
                    else np.nan
                )
                for k in DISCOUNT_METRICS
            }

            rows.append(
                {
                    "snapshot_date": snapshot,
                    "sector": sector_map[ticker],
                    "ticker": ticker,
                    "market_price": market_price,
                    **valuation,

                    # Base fundamental da Célula 37
                    "revenue": f.get("revenue", np.nan),
                    "net_income": f.get("net_income", np.nan),
                    "operating_cash_flow": f.get("operating_cash_flow", np.nan),
                    "free_cash_flow": f.get("free_cash_flow", np.nan),
                    "diluted_eps": f.get("diluted_eps", np.nan),
                    "shares_outstanding": f.get("shares_outstanding", np.nan),
                    "shares": f.get("shares", np.nan),
                    "diluted_shares": f.get("diluted_shares", np.nan),

                    "revenue_period_end": f.get("revenue_period_end", pd.NaT),
                    "net_income_period_end": f.get("net_income_period_end", pd.NaT),
                    "operating_cash_flow_period_end": f.get(
                        "operating_cash_flow_period_end", pd.NaT
                    ),
                    "diluted_eps_period_end": f.get(
                        "diluted_eps_period_end", pd.NaT
                    ),

                    "net_margin": f.get("net_margin", np.nan),
                    "ocf_margin": f.get("ocf_margin", np.nan),
                    "fcf_margin": f.get("fcf_margin", np.nan),

                    **price_components,

                    "momentum_6m": (
                        _num(pr["momentum_6m"])
                        if pr is not None
                        else np.nan
                    ),
                    "momentum_12m": (
                        _num(pr["momentum_12m"])
                        if pr is not None
                        else np.nan
                    ),
                }
            )

    hist = pd.DataFrame(rows)
    if hist.empty:
        raise RuntimeError("Histórico mensal de entrada ficou vazio.")

    hist = hist.sort_values(
        [
            "ticker",
            "snapshot_date",
        ]
    ).reset_index(
        drop=True
    )

    # ------------------------------------------------------------------
    # FUNDAMENTOS — CÉLULA 37
    # ------------------------------------------------------------------
    #
    # Os crescimentos são construídos sobre períodos fundamentais
    # únicos consecutivos e só então normalizados contra o próprio
    # histórico da ação.
    # ------------------------------------------------------------------

    hist = _apply_cell37_growth_logic(
        hist
    )

    # ------------------------------------------------------------------
    # VALUATION: próprio histórico; menor múltiplo = melhor; mediana
    # ------------------------------------------------------------------
    for metric in VALUATION_METRICS:
        hist[f"{metric}_relative_score"] = np.nan

    for ticker, idx in hist.groupby("ticker").groups.items():
        idx = list(idx)
        temp = hist.loc[idx].sort_values("snapshot_date")
        for metric in VALUATION_METRICS:
            scores = _expanding_score(temp[metric], higher_is_better=False)
            hist.loc[temp.index, f"{metric}_relative_score"] = scores.values

    val_cols = [f"{m}_relative_score" for m in VALUATION_METRICS]
    hist["valid_relative_metrics"] = hist[val_cols].notna().sum(axis=1)
    hist["relative_valuation_score"] = hist[val_cols].median(axis=1, skipna=True)
    hist.loc[
        hist["valid_relative_metrics"] < MIN_VALUATION_COMPONENTS,
        "relative_valuation_score",
    ] = np.nan

    # ------------------------------------------------------------------
    # DESCONTO: próprio histórico; bruto menor = mais descontado; mediana
    # ------------------------------------------------------------------
    for metric in DISCOUNT_METRICS:
        hist[f"{metric}_discount_score"] = np.nan

    for ticker, idx in hist.groupby("ticker").groups.items():
        temp = hist.loc[list(idx)].sort_values("snapshot_date")
        for metric in DISCOUNT_METRICS:
            scores = _expanding_score(temp[metric], higher_is_better=False)
            hist.loc[temp.index, f"{metric}_discount_score"] = scores.values

    dis_cols = [f"{m}_discount_score" for m in DISCOUNT_METRICS]
    hist["valid_discount_metrics"] = hist[dis_cols].notna().sum(axis=1)
    hist["price_discount_score"] = hist[dis_cols].median(axis=1, skipna=True)
    hist.loc[
        hist["valid_discount_metrics"] < MIN_DISCOUNT_COMPONENTS,
        "price_discount_score",
    ] = np.nan

    # ------------------------------------------------------------------
    # FUNDAMENTOS: próprio histórico; maior = melhor; mediana
    # ------------------------------------------------------------------
    for component in FUNDAMENTAL_COMPONENTS:
        hist[f"{component}_score"] = np.nan

    for ticker, idx in hist.groupby("ticker").groups.items():
        temp = hist.loc[list(idx)].sort_values("snapshot_date")
        for component in FUNDAMENTAL_COMPONENTS:
            scores = _expanding_score(temp[component], higher_is_better=True)
            hist.loc[temp.index, f"{component}_score"] = scores.values

    fund_cols = [f"{c}_score" for c in FUNDAMENTAL_COMPONENTS]
    hist["valid_fundamental_components"] = hist[fund_cols].notna().sum(axis=1)
    hist["fundamental_preservation_score"] = hist[fund_cols].median(axis=1, skipna=True)
    hist.loc[
        hist["valid_fundamental_components"] < MIN_FUNDAMENTAL_COMPONENTS,
        "fundamental_preservation_score",
    ] = np.nan

    # ------------------------------------------------------------------
    # FINAL SIGNAL SCORE
    # ------------------------------------------------------------------
    hist["final_signal_score"] = np.nan
    hist["valid_signal_components"] = 0

    mask = hist["sector"] == HEALTH_CARE
    h = hist.loc[mask].copy()
    h["valuation"] = h["relative_valuation_score"]
    h["discount"] = h["price_discount_score"]
    h["fundamental"] = h["fundamental_preservation_score"]
    hist.loc[mask, "final_signal_score"] = _weighted_score(
        h,
        {"valuation": 0.10, "discount": 0.80, "fundamental": 0.10},
    ).values
    hist.loc[mask, "valid_signal_components"] = h[
        ["valuation", "discount", "fundamental"]
    ].notna().sum(axis=1).values

    mask = hist["sector"] == INDUSTRIALS
    ind = hist.loc[mask].copy()
    ind["valuation"] = ind["relative_valuation_score"]
    ind["discount"] = ind["price_discount_score"]
    ind["fundamental"] = ind["fundamental_preservation_score"]
    hist.loc[mask, "final_signal_score"] = _weighted_score(
        ind,
        {"valuation": 0.00, "discount": 0.20, "fundamental": 0.80},
    ).values
    hist.loc[mask, "valid_signal_components"] = ind[
        ["discount", "fundamental"]
    ].notna().sum(axis=1).values

    # Technology: cross-sectional por snapshot, exatamente como Célula 41
    mask = hist["sector"] == TECHNOLOGY
    tech = hist.loc[mask].copy()
    tech["momentum_6m_score"] = (
        tech.groupby("snapshot_date")["momentum_6m"].rank(pct=True, ascending=True)
    )
    tech["momentum_12m_score"] = (
        tech.groupby("snapshot_date")["momentum_12m"].rank(pct=True, ascending=True)
    )
    tech["tech_score"] = (
        0.50 * tech["momentum_6m_score"]
        + 0.50 * tech["momentum_12m_score"]
    )
    hist.loc[tech.index, "momentum_6m_score"] = tech["momentum_6m_score"]
    hist.loc[tech.index, "momentum_12m_score"] = tech["momentum_12m_score"]
    hist.loc[tech.index, "final_signal_score"] = tech["tech_score"]
    hist.loc[tech.index, "valid_signal_components"] = tech[
        ["momentum_6m_score", "momentum_12m_score"]
    ].notna().sum(axis=1)

    # ------------------------------------------------------------------
    # PERCENTIL HISTÓRICO DO SINAL POR SETOR
    # Célula 41: benchmark contém apenas snapshots ANTERIORES.
    # ------------------------------------------------------------------
    hist["signal_percentile"] = np.nan

    for sector in SECTORS:
        sector_idx = hist["sector"] == sector
        dates = sorted(hist.loc[sector_idx, "snapshot_date"].dropna().unique())
        previous_scores = []

        for snapshot in dates:
            current_idx = hist.index[
                sector_idx & (hist["snapshot_date"] == snapshot)
            ]

            benchmark = [
                x for x in previous_scores if np.isfinite(_num(x))
            ]

            if len(benchmark) >= MIN_HISTORY:
                for idx in current_idx:
                    score = _num(hist.at[idx, "final_signal_score"])
                    hist.at[idx, "signal_percentile"] = _midrank_percentile(
                        score,
                        benchmark,
                    )

            previous_scores.extend(
                hist.loc[current_idx, "final_signal_score"].dropna().tolist()
            )

    return hist.sort_values(["sector", "ticker", "snapshot_date"]).reset_index(drop=True)


# ======================================================================================
# CLASSIFICAÇÃO ATUAL
# ======================================================================================

def classify_portfolio_entries(
    portfolio: pd.DataFrame,
    prices: pd.DataFrame,
    fundamentals_history: pd.DataFrame,
    as_of_date=None,
) -> pd.DataFrame:

    if portfolio["ticker"].nunique() != 15:
        raise RuntimeError("entry.py recebeu carteira diferente de 15 ações.")

    counts = portfolio.groupby("sector")["ticker"].nunique()
    for sector in SECTORS:
        if int(counts.get(sector, 0)) != 5:
            raise RuntimeError(f"{sector}: estrutura diferente de 5 ações.")

    history = build_entry_signal_history(
        portfolio=portfolio,
        prices=prices,
        fundamentals_history=fundamentals_history,
        as_of_date=as_of_date,
    )

    latest_date = history["snapshot_date"].max()
    current = history[history["snapshot_date"] == latest_date].copy()

    # manter metadados de seleção
    meta_cols = [
        c for c in [
            "ticker",
            "selection_factor",
            "selection_score",
            "selection_rank",
        ]
        if c in portfolio.columns
    ]
    current = current.merge(
        portfolio[meta_cols].drop_duplicates("ticker"),
        on="ticker",
        how="left",
    )

    current["selection_status"] = "APROVADA"
    current["valuation_status"] = current["relative_valuation_score"].map(valuation_status)
    current["discount_status"] = current["price_discount_score"].map(discount_status)
    current["fundamental_status"] = current["fundamental_preservation_score"].map(
        fundamental_status
    )
    current["timing_method"] = current["sector"].map(ENTRY_METHODS)
    current["sector_validation_status"] = current["sector"].map(
        SECTOR_VALIDATION_STATUS
    )
    current["entry_signal"] = [
        classify_entry(pct, sector)
        for pct, sector in zip(
            current["signal_percentile"],
            current["sector"],
        )
    ]

    current["buy_priority_sector"] = (
        current.groupby("sector")["final_signal_score"]
        .rank(method="min", ascending=False, na_option="bottom")
    )

    # Salvar histórico para auditoria / futuras comparações
    history.to_parquet("output/entry_signal_history.parquet", index=False)

    signal_order = {
        "ENTRADA FORTE": 1,
        "ENTRADA": 2,
        "AGUARDAR": 3,
        "NÃO COMPRAR AGORA": 4,
    }
    current["_signal_order"] = current["entry_signal"].map(signal_order)

    return (
        current
        .sort_values(
            ["_signal_order", "signal_percentile", "final_signal_score"],
            ascending=[True, False, False],
            na_position="last",
        )
        .drop(columns=["_signal_order"])
        .reset_index(drop=True)
    )


def audit_entry_ranking(ranking: pd.DataFrame) -> Dict:
    signals = ranking["entry_signal"].value_counts().to_dict()
    sector_counts = ranking.groupby("sector")["ticker"].nunique().to_dict()

    industrial_strong = int(
        (
            (ranking["sector"] == INDUSTRIALS)
            & (ranking["entry_signal"] == "ENTRADA FORTE")
        ).sum()
    )

    structure_ok = (
        ranking["ticker"].nunique() == 15
        and all(sector_counts.get(s, 0) == 5 for s in SECTORS)
        and industrial_strong == 0
    )

    return {
        "number_of_stocks": int(ranking["ticker"].nunique()),
        "number_of_sectors": int(ranking["sector"].nunique()),
        "sector_counts": sector_counts,
        "entry_strong": int(signals.get("ENTRADA FORTE", 0)),
        "entry": int(signals.get("ENTRADA", 0)),
        "wait": int(signals.get("AGUARDAR", 0)),
        "do_not_buy": int(signals.get("NÃO COMPRAR AGORA", 0)),
        "industrial_strong_violation": industrial_strong,
        "structure_ok": bool(structure_ok),
    }


# ======================================================================================
# COMPATIBILIDADE COM A ESTRUTURA ANTERIOR
# ======================================================================================
#
# As funções abaixo preservam nomes utilizados na versão antiga do entry.py.
# Elas NÃO alteram a lógica corrigida. Servem para:
#
#   • facilitar auditoria;
#   • evitar quebra de imports antigos;
#   • manter o arquivo operacional mais completo;
#   • permitir comparação com a implementação anterior.
#
# ======================================================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    """
    Converte uma Series para numérico e remove infinitos.
    """
    return (
        pd.to_numeric(series, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """
    Divisão vetorial segura.
    """
    numerator = safe_numeric(numerator)
    denominator = safe_numeric(denominator)

    result = numerator / denominator.replace(0, np.nan)

    return result.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def percentile_score(
    series: pd.Series,
    higher_is_better: bool = True,
) -> pd.Series:
    """
    Percentil transversal simples.

    IMPORTANTE:
    Esta função é mantida apenas para compatibilidade/auditoria.
    Ela NÃO é usada para produzir o signal_percentile final.

    O signal_percentile final continua sendo calculado contra
    o histórico anterior do próprio setor, conforme a Célula 41.
    """

    values = safe_numeric(series)

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
    Percentil de um valor atual contra uma série histórica.

    Utilidade:
        auditoria de valuation e desconto.

    Não substitui o processo point-in-time utilizado no motor principal.
    """

    current_value = _num(current_value)

    hist = (
        pd.to_numeric(history, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    if (
        not np.isfinite(current_value)
        or len(hist) < MIN_HISTORY
    ):
        return np.nan

    less = np.sum(hist < current_value)
    equal = np.sum(hist == current_value)

    percentile = (
        less
        +
        0.5 * equal
    ) / len(hist)

    if lower_is_better:
        percentile = 1.0 - percentile

    return float(percentile)


def mean_valid(
    values: Iterable,
    minimum: int = 1,
) -> float:
    """
    Média dos valores válidos.
    """

    values = pd.to_numeric(
        pd.Series(list(values)),
        errors="coerce",
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    valid = values.dropna()

    if len(valid) < minimum:
        return np.nan

    return float(valid.mean())


def calculate_momentum(
    prices: pd.Series,
    months: int,
) -> float:
    """
    Retorno simples aproximado de N meses.

    Mantido para compatibilidade com a versão antiga.
    O motor principal usa 126 dias úteis para 6M
    e 252 dias úteis para 12M.
    """

    prices = (
        pd.to_numeric(prices, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )

    trading_days = 21 * months

    if len(prices) <= trading_days:
        return np.nan

    current = _num(prices.iloc[-1])
    previous = _num(prices.iloc[-trading_days - 1])

    if (
        not np.isfinite(current)
        or not np.isfinite(previous)
        or previous == 0
    ):
        return np.nan

    return float(
        current / previous - 1.0
    )


# ======================================================================================
# RESUMO EXECUTIVO
# ======================================================================================

def build_entry_summary(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói a tabela resumida usada em relatório/auditoria.
    """

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
        "relative_valuation_score",
        "price_discount_score",
        "fundamental_preservation_score",
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
# AUDITORIA DO HISTÓRICO DE ENTRADA
# ======================================================================================

def audit_entry_history(
    history: pd.DataFrame,
) -> Dict:
    """
    Auditoria metodológica do histórico produzido pelo entry engine.

    Não interfere nos sinais.
    """

    if history is None or history.empty:
        return {
            "history_rows": 0,
            "history_ok": False,
        }

    required = {
        "snapshot_date",
        "ticker",
        "sector",
        "final_signal_score",
        "signal_percentile",
    }

    missing = sorted(
        required - set(history.columns)
    )

    duplicates = int(
        history.duplicated(
            subset=[
                "ticker",
                "snapshot_date",
            ]
        ).sum()
    )

    counts = (
        history.groupby("sector")["ticker"]
        .nunique()
        .to_dict()
    )

    valid_signal = int(
        history["final_signal_score"]
        .notna()
        .sum()
    )

    valid_percentile = int(
        history["signal_percentile"]
        .notna()
        .sum()
    )

    return {
        "history_rows":
            int(len(history)),

        "tickers":
            int(history["ticker"].nunique()),

        "sectors":
            int(history["sector"].nunique()),

        "sector_ticker_counts":
            counts,

        "first_snapshot":
            history["snapshot_date"].min(),

        "last_snapshot":
            history["snapshot_date"].max(),

        "duplicate_ticker_date":
            duplicates,

        "valid_final_signal_score":
            valid_signal,

        "valid_signal_percentile":
            valid_percentile,

        "missing_required_columns":
            missing,

        "history_ok":
            (
                not missing
                and duplicates == 0
                and history["ticker"].nunique() == 15
                and all(
                    counts.get(sector, 0) == 5
                    for sector in SECTORS
                )
            ),
    }


# ======================================================================================
# AUDITORIA DOS COMPONENTES DO SNAPSHOT ATUAL
# ======================================================================================

def audit_entry_components(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mostra, ação por ação, quais componentes do sinal estão disponíveis.

    Serve para identificar rapidamente:
        valuation insuficiente,
        desconto insuficiente,
        fundamentos insuficientes,
        momentum insuficiente.
    """

    df = ranking.copy()

    rows = []

    for _, row in df.iterrows():

        sector = row.get("sector")
        ticker = row.get("ticker")

        valuation_n = int(
            _num(
                row.get(
                    "valid_relative_metrics",
                    0,
                )
            )
            if np.isfinite(
                _num(
                    row.get(
                        "valid_relative_metrics",
                        0,
                    )
                )
            )
            else 0
        )

        discount_n = int(
            _num(
                row.get(
                    "valid_discount_metrics",
                    0,
                )
            )
            if np.isfinite(
                _num(
                    row.get(
                        "valid_discount_metrics",
                        0,
                    )
                )
            )
            else 0
        )

        fundamental_n = int(
            _num(
                row.get(
                    "valid_fundamental_components",
                    0,
                )
            )
            if np.isfinite(
                _num(
                    row.get(
                        "valid_fundamental_components",
                        0,
                    )
                )
            )
            else 0
        )

        signal_n = int(
            _num(
                row.get(
                    "valid_signal_components",
                    0,
                )
            )
            if np.isfinite(
                _num(
                    row.get(
                        "valid_signal_components",
                        0,
                    )
                )
            )
            else 0
        )

        if sector == HEALTH_CARE:
            component_ok = (
                discount_n >= MIN_DISCOUNT_COMPONENTS
                and fundamental_n >= MIN_FUNDAMENTAL_COMPONENTS
                and signal_n >= 1
            )

        elif sector == INDUSTRIALS:
            component_ok = (
                discount_n >= MIN_DISCOUNT_COMPONENTS
                and fundamental_n >= MIN_FUNDAMENTAL_COMPONENTS
                and signal_n >= 1
            )

        elif sector == TECHNOLOGY:
            component_ok = (
                signal_n >= 2
            )

        else:
            component_ok = False

        rows.append(
            {
                "sector":
                    sector,

                "ticker":
                    ticker,

                "valid_valuation_metrics":
                    valuation_n,

                "valid_discount_metrics":
                    discount_n,

                "valid_fundamental_components":
                    fundamental_n,

                "valid_signal_components":
                    signal_n,

                "final_signal_score":
                    row.get(
                        "final_signal_score"
                    ),

                "signal_percentile":
                    row.get(
                        "signal_percentile"
                    ),

                "entry_signal":
                    row.get(
                        "entry_signal"
                    ),

                "component_ok":
                    component_ok,
            }
        )

    return pd.DataFrame(rows)


# ======================================================================================
# AUDITORIA DAS REGRAS CONGELADAS
# ======================================================================================

def audit_frozen_rules() -> Dict:
    """
    Confirma em runtime que as regras centrais continuam congeladas.
    """

    return {
        "health_care_rule":
            {
                "valuation":
                    0.10,

                "discount":
                    0.80,

                "fundamental":
                    0.10,

                "status":
                    "APROVADO",
            },

        "industrials_rule":
            {
                "valuation":
                    0.00,

                "discount":
                    0.20,

                "fundamental":
                    0.80,

                "status":
                    "CONDICIONAL",

                "allow_strong_entry":
                    False,
            },

        "technology_rule":
            {
                "momentum_6m":
                    0.50,

                "momentum_12m":
                    0.50,

                "status":
                    "REGRA ALTERNATIVA APROVADA",
            },

        "entry_thresholds":
            {
                "strong":
                    0.75,

                "entry":
                    0.50,

                "wait":
                    0.25,
            },

        "historical_signal_benchmark":
            "SETOR — SOMENTE SNAPSHOTS ANTERIORES",

        "minimum_signal_history":
            MIN_HISTORY,
    }


# ======================================================================================
# IMPRESSÃO DE AUDITORIA
# ======================================================================================

def print_entry_audit(
    ranking: pd.DataFrame,
):
    """
    Impressão completa para GitHub Actions.
    """

    audit = audit_entry_ranking(
        ranking
    )

    print(
        "\n"
        +
        "=" * 110
    )

    print(
        "AUDITORIA — ENTRY ENGINE"
    )

    print(
        "=" * 110
    )

    print(
        f"\nAções                    : "
        f"{audit['number_of_stocks']}"
    )

    print(
        f"Setores                   : "
        f"{audit['number_of_sectors']}"
    )

    print(
        f"Entrada Forte             : "
        f"{audit['entry_strong']}"
    )

    print(
        f"Entrada                   : "
        f"{audit['entry']}"
    )

    print(
        f"Aguardar                  : "
        f"{audit['wait']}"
    )

    print(
        f"Não comprar agora         : "
        f"{audit['do_not_buy']}"
    )

    print(
        f"Violação Industrials      : "
        f"{audit['industrial_strong_violation']}"
    )

    print(
        f"Estrutura OK              : "
        f"{audit['structure_ok']}"
    )

    print(
        "\nComponentes:"
    )

    component_table = (
        audit_entry_components(
            ranking
        )
    )

    print(
        component_table.to_string(
            index=False
        )
    )


if __name__ == "__main__":

    print(
        "=" * 110
    )

    print(
        "PORTFOLIO ACOES AMERICANO — ENTRY ENGINE"
    )

    print(
        "=" * 110
    )

    print(
        "\nArquitetura congelada:"
    )

    print(
        "  Health Care"
        " -> 10% Valuation + 80% Desconto + 10% Fundamentos"
    )

    print(
        "  Industrials"
        " -> 20% Desconto + 80% Fundamentos — CONDICIONAL"
    )

    print(
        "  Information Technology"
        " -> Momentum 6M + Momentum 12M"
    )

    print(
        "\nNormalização:"
    )

    print(
        "  Valuation     -> próprio histórico da ação"
    )

    print(
        "  Desconto      -> próprio histórico da ação"
    )

    print(
        "  Fundamentos   -> próprio histórico da ação"
    )

    print(
        "  Sinal final   -> histórico anterior do SETOR"
    )

    print(
        "\nClassificação:"
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
        "\nProteção:"
    )

    print(
        "  Industrials nunca recebe ENTRADA FORTE."
    )

    print(
        "\nSTATUS: ENTRY ENGINE CARREGADO"
    )
