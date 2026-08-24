# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# entry.py — VERSÃO CORRIGIDA / ALINHADA À CÉLULA 41
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

def _prepare_facts(fundamentals_history: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    required = {"ticker", "metric", "value", "available_date", "end"}
    if fundamentals_history is None or fundamentals_history.empty:
        raise RuntimeError("Histórico fundamental vazio.")
    if not required.issubset(fundamentals_history.columns):
        raise RuntimeError(
            f"Histórico fundamental precisa conter: {sorted(required)}"
        )

    facts = fundamentals_history.copy()
    facts["ticker"] = facts["ticker"].astype(str).str.upper().str.strip()
    facts = facts[facts["ticker"].isin(tickers)].copy()
    facts["available_date"] = pd.to_datetime(facts["available_date"], errors="coerce")
    facts["end"] = pd.to_datetime(facts["end"], errors="coerce")
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")
    facts = facts.dropna(subset=["ticker", "metric", "value", "available_date", "end"])
    return facts.sort_values(["ticker", "metric", "available_date", "end"])


def _latest_metric(history: pd.DataFrame, metric: str, snapshot: pd.Timestamp):
    temp = history[
        (history["metric"] == metric)
        & (history["available_date"] <= snapshot)
    ]
    if temp.empty:
        return None
    return temp.sort_values(["available_date", "end"]).iloc[-1]


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
    raw = {}

    for metric in [
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
    ]:
        row = _latest_metric(ticker_history, metric, snapshot)
        raw[metric] = _num(row["value"]) if row is not None else np.nan

    revenue = raw["revenue"]
    net_income = raw["net_income"]
    ocf = raw["operating_cash_flow"]
    capex = raw["capex"]

    fcf = (
        ocf - abs(capex)
        if np.isfinite(ocf) and np.isfinite(capex)
        else np.nan
    )

    return {
        **raw,
        "revenue_growth_yoy": _growth_yoy(ticker_history, "revenue", snapshot),
        "net_income_growth_yoy": _growth_yoy(ticker_history, "net_income", snapshot),
        "operating_cash_flow_growth_yoy": _growth_yoy(
            ticker_history, "operating_cash_flow", snapshot
        ),
        "diluted_eps_growth_yoy": _growth_yoy(ticker_history, "diluted_eps", snapshot),
        "net_margin": _safe_div(net_income, revenue),
        "ocf_margin": _safe_div(ocf, revenue),
        "fcf_margin": _safe_div(fcf, revenue),
        "free_cash_flow": fcf,
    }


def _valuation_raw(price: float, f: Dict[str, float]) -> Dict[str, float]:
    shares = _num(f.get("diluted_shares"))
    market_cap = (
        price * shares
        if np.isfinite(price) and np.isfinite(shares) and shares > 0
        else np.nan
    )

    eps = _num(f.get("diluted_eps"))
    equity = _num(f.get("equity"))
    revenue = _num(f.get("revenue"))
    ocf = _num(f.get("operating_cash_flow"))
    fcf = _num(f.get("free_cash_flow"))

    # múltiplos negativos/não econômicos são tratados como ausentes
    pe = price / eps if np.isfinite(eps) and eps > 0 else np.nan
    pb = market_cap / equity if np.isfinite(market_cap) and np.isfinite(equity) and equity > 0 else np.nan
    ps = market_cap / revenue if np.isfinite(market_cap) and np.isfinite(revenue) and revenue > 0 else np.nan
    p_ocf = market_cap / ocf if np.isfinite(market_cap) and np.isfinite(ocf) and ocf > 0 else np.nan
    p_fcf = market_cap / fcf if np.isfinite(market_cap) and np.isfinite(fcf) and fcf > 0 else np.nan

    return {"pe": pe, "pb": pb, "ps": ps, "p_ocf": p_ocf, "p_fcf": p_fcf}


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

        if price_hist.empty:
            continue

        for snapshot in snapshot_dates:
            price_rows = price_hist[price_hist["date"] <= snapshot]
            if price_rows.empty:
                continue

            pr = price_rows.iloc[-1]
            # tolerância de 10 dias como no estudo
            if (snapshot - pd.Timestamp(pr["date"])).days > 10:
                continue

            f = _snapshot_fundamentals(fact_hist, snapshot)
            valuation = _valuation_raw(_num(pr["price"]), f)

            rows.append(
                {
                    "snapshot_date": snapshot,
                    "sector": sector_map[ticker],
                    "ticker": ticker,
                    "market_price": _num(pr["price"]),
                    **valuation,
                    **{k: f.get(k, np.nan) for k in FUNDAMENTAL_COMPONENTS},
                    **{k: _num(pr[k]) for k in DISCOUNT_METRICS},
                    "momentum_6m": _num(pr["momentum_6m"]),
                    "momentum_12m": _num(pr["momentum_12m"]),
                }
            )

    hist = pd.DataFrame(rows)
    if hist.empty:
        raise RuntimeError("Histórico mensal de entrada ficou vazio.")

    hist = hist.sort_values(["ticker", "snapshot_date"]).reset_index(drop=True)

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


if __name__ == "__main__":
    print("ENTRY ENGINE — lógica histórica da Célula 41 carregada.")
