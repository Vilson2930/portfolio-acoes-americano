# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# email_report.py
# ======================================================================================
from __future__ import annotations

import html
import mimetypes
import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RANKING_FILE = Path(os.getenv("RANKING_FILE", "output/portfolio_ranking.csv"))
PDF_FILE = Path(os.getenv("PDF_FILE", "output/portfolio_report.pdf"))
DEFAULT_EMAIL_TO = "vilsonjosepereirapinto@gmail.com"


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _signal_order(signal: str) -> int:
    return {"ENTRADA FORTE":0,"ENTRADA":1,"AGUARDAR":2,"NÃO COMPRAR AGORA":3}.get(str(signal).strip().upper(), 9)


def _pct(value) -> str:
    try:
        value = float(value)
        return f"{value * 100:.1f}%" if np.isfinite(value) else "N/D"
    except Exception:
        return "N/D"


def _signal_counts(ranking: pd.DataFrame) -> Dict[str, int]:
    counts = ranking.get("entry_signal", pd.Series(dtype=str)).fillna("N/D").astype(str).value_counts().to_dict()
    return {
        "ENTRADA FORTE": int(counts.get("ENTRADA FORTE", 0)),
        "ENTRADA": int(counts.get("ENTRADA", 0)),
        "AGUARDAR": int(counts.get("AGUARDAR", 0)),
        "NÃO COMPRAR AGORA": int(counts.get("NÃO COMPRAR AGORA", 0)),
    }


def _prepare_top(ranking: pd.DataFrame, limit: int = 7) -> pd.DataFrame:
    df = ranking.copy()
    df["_signal_order"] = df.get("entry_signal", pd.Series(index=df.index, dtype=str)).map(_signal_order).fillna(9)
    df["_percentile_sort"] = pd.to_numeric(df.get("signal_percentile", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(-1)
    return df.sort_values(["_signal_order", "_percentile_sort"], ascending=[True, False]).head(limit)


def build_plain_body(ranking: pd.DataFrame) -> str:
    counts = _signal_counts(ranking)
    top = _prepare_top(ranking)
    lines = [
        "PORTFOLIO AÇÕES AMERICANO — RESULTADO DIÁRIO", "",
        f"O motor concluiu a seleção e a classificação do ponto de entrada de {len(ranking)} ações.", "",
        "RESUMO DOS SINAIS",
        f"Entrada Forte       : {counts['ENTRADA FORTE']}",
        f"Entrada             : {counts['ENTRADA']}",
        f"Aguardar            : {counts['AGUARDAR']}",
        f"Não comprar agora  : {counts['NÃO COMPRAR AGORA']}", "",
        "PRINCIPAIS OPORTUNIDADES",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"{row.get('ticker','N/D')} | {row.get('sector','N/D')} | {row.get('entry_signal','N/D')} | "
            f"Score final {_pct(row.get('final_signal_score'))} | Percentil {_pct(row.get('signal_percentile'))}"
        )
    lines += [
        "", "COMO INTERPRETAR",
        "ENTRADA FORTE = oportunidade relativa excepcional segundo as regras do estudo.",
        "ENTRADA = ponto de compra aprovado pelo modelo.",
        "AGUARDAR = empresa selecionada, mas timing ainda sem confirmação suficiente.",
        "NÃO COMPRAR AGORA = manter em observação; não há aprovação para nova compra.",
        "", "O PDF completo está anexado separadamente e traz o ranking e a explicação de cada ação.",
        "", "Relatório quantitativo de apoio à decisão. Não constitui garantia de retorno."
    ]
    return "\n".join(lines)


def build_html_body(ranking: pd.DataFrame) -> str:
    counts = _signal_counts(ranking)
    top = _prepare_top(ranking)
    rows = []
    for _, row in top.iterrows():
        ticker = html.escape(str(row.get("ticker", "N/D")))
        sector = html.escape(str(row.get("sector", "N/D")))
        signal = html.escape(str(row.get("entry_signal", "N/D")))
        rows.append(
            f"<tr><td style='padding:8px;border-bottom:1px solid #e5e7eb'><b>{ticker}</b></td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb'>{sector}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb'><b>{signal}</b></td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb'>{_pct(row.get('final_signal_score'))}</td>"
            f"<td style='padding:8px;border-bottom:1px solid #e5e7eb'>{_pct(row.get('signal_percentile'))}</td></tr>"
        )

    return f"""
    <html><body style="font-family:Arial,Helvetica,sans-serif;color:#111827;background:#f3f4f6;margin:0;padding:24px">
    <div style="max-width:760px;margin:auto;background:#fff;border-radius:12px;padding:28px">
      <h1 style="margin:0 0 6px 0;font-size:24px">Portfolio Ações Americano</h1>
      <div style="color:#6b7280;margin-bottom:24px">Resultado diário — {datetime.now():%d/%m/%Y}</div>
      <p>O motor concluiu a seleção e a classificação do ponto de entrada de <b>{len(ranking)} ações</b>. O resumo está abaixo e o relatório completo está no <b>PDF anexado separadamente</b>.</p>
      <h2 style="font-size:18px">Resumo dos sinais</h2>
      <table style="width:100%;border-collapse:collapse;margin:10px 0 22px 0">
        <tr><td style="padding:10px;background:#f9fafb"><b>Entrada Forte</b></td><td style="padding:10px;background:#f9fafb;text-align:right">{counts['ENTRADA FORTE']}</td></tr>
        <tr><td style="padding:10px"><b>Entrada</b></td><td style="padding:10px;text-align:right">{counts['ENTRADA']}</td></tr>
        <tr><td style="padding:10px;background:#f9fafb"><b>Aguardar</b></td><td style="padding:10px;background:#f9fafb;text-align:right">{counts['AGUARDAR']}</td></tr>
        <tr><td style="padding:10px"><b>Não comprar agora</b></td><td style="padding:10px;text-align:right">{counts['NÃO COMPRAR AGORA']}</td></tr>
      </table>
      <h2 style="font-size:18px">Principais oportunidades</h2>
      <table style="width:100%;border-collapse:collapse;font-size:14px">
        <thead><tr style="background:#111827;color:#fff"><th style="padding:8px;text-align:left">Ticker</th><th style="padding:8px;text-align:left">Setor</th><th style="padding:8px;text-align:left">Sinal</th><th style="padding:8px;text-align:left">Score final</th><th style="padding:8px;text-align:left">Percentil</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <h2 style="font-size:18px;margin-top:28px">Como interpretar</h2>
      <p><b>Entrada Forte:</b> oportunidade relativa excepcional segundo as regras do estudo.</p>
      <p><b>Entrada:</b> ponto de compra aprovado pelo modelo.</p>
      <p><b>Aguardar:</b> empresa selecionada, porém o timing ainda precisa melhorar.</p>
      <p><b>Não comprar agora:</b> empresa permanece no acompanhamento, mas não há aprovação para nova compra.</p>
      <div style="margin-top:26px;padding:14px;background:#f9fafb;border-left:4px solid #6b7280;font-size:13px;color:#4b5563">O PDF anexado traz o ranking completo e a explicação individual de valuation, desconto, preservação fundamental, score e sinal de cada ação.</div>
      <p style="margin-top:24px;font-size:12px;color:#6b7280">Relatório quantitativo de apoio à decisão. Não constitui garantia de retorno.</p>
    </div></body></html>
    """


def validate_files() -> pd.DataFrame:
    if not RANKING_FILE.exists():
        raise RuntimeError(f"Ranking não encontrado: {RANKING_FILE}")
    if not PDF_FILE.exists():
        raise RuntimeError(f"PDF não encontrado: {PDF_FILE}")
    ranking = pd.read_csv(RANKING_FILE)
    if ranking.empty:
        raise RuntimeError("O ranking está vazio.")
    return ranking


def validate_smtp() -> Dict[str, str]:
    config = {
        "server": _env("SMTP_SERVER"),
        "port": _env("SMTP_PORT", "587"),
        "user": _env("SMTP_USER"),
        "password": _env("SMTP_PASSWORD"),
        "to": _env("EMAIL_TO", DEFAULT_EMAIL_TO),
    }
    missing = [k for k in ["server", "user", "password", "to"] if not config[k]]
    if missing:
        raise RuntimeError("Configuração SMTP incompleta. Campos ausentes: " + ", ".join(missing))
    int(config["port"])
    return config


def attach_pdf(message: EmailMessage) -> None:
    mime_type, _ = mimetypes.guess_type(str(PDF_FILE))
    mime_type = mime_type or "application/pdf"
    maintype, subtype = mime_type.split("/", 1)
    with PDF_FILE.open("rb") as file:
        message.add_attachment(file.read(), maintype=maintype, subtype=subtype, filename=PDF_FILE.name)


def send_email() -> None:
    ranking = validate_files()
    smtp = validate_smtp()
    counts = _signal_counts(ranking)

    strongest = ranking[ranking["entry_signal"] == "ENTRADA FORTE"] if "entry_signal" in ranking.columns else pd.DataFrame()
    if not strongest.empty:
        subject = f"Portfolio Ações EUA — {strongest.iloc[0].get('ticker','')} ENTRADA FORTE — {datetime.now():%d/%m/%Y}"
    else:
        subject = f"Portfolio Ações EUA — {counts['ENTRADA']} entradas — {datetime.now():%d/%m/%Y}"

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp["user"]
    message["To"] = smtp["to"]
    message.set_content(build_plain_body(ranking))
    message.add_alternative(build_html_body(ranking), subtype="html")
    attach_pdf(message)

    port = int(smtp["port"])
    context = ssl.create_default_context()

    if port == 465:
        with smtplib.SMTP_SSL(smtp["server"], port, context=context, timeout=60) as server:
            server.login(smtp["user"], smtp["password"])
            server.send_message(message)
    else:
        with smtplib.SMTP(smtp["server"], port, timeout=60) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(smtp["user"], smtp["password"])
            server.send_message(message)

    print("=" * 80)
    print("E-MAIL ENVIADO COM SUCESSO")
    print("=" * 80)
    print(f"Destinatário : {smtp['to']}")
    print(f"Assunto      : {subject}")
    print(f"PDF anexado  : {PDF_FILE}")


if __name__ == "__main__":
    send_email()
