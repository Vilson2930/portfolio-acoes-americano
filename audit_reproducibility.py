# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_reproducibility.py — VERSÃO OTIMIZADA
# ======================================================================================

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_JSON = OUTPUT_DIR / "audit_reproducibility.json"
REPORT_TXT = OUTPUT_DIR / "audit_reproducibility.txt"

PYTHON = sys.executable

CRITICAL_PYTHON_FILES = [
    "config.py",
    "data.py",
    "selection.py",
    "entry.py",
    "main.py",
    "audit_boundary.py",
    "audit_valuation_cell33.py",
]

ESSENTIAL_AUDITS = [
    ("audit_boundary.py", "Fronteira — Célula 31", 1200),
    ("audit_valuation_cell33.py", "Valuation — Célula 33B", 1200),
]


def header(title: str) -> None:
    print("\n" + "=" * 120, flush=True)
    print(title, flush=True)
    print("=" * 120, flush=True)


def stream_command(
    args: List[str],
    label: str,
    timeout_seconds: int,
) -> Tuple[int, str, float]:

    header(label)
    print("$ " + " ".join(args), flush=True)

    started = time.monotonic()

    process = subprocess.Popen(
        args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected: List[str] = []

    while True:
        if process.stdout is not None:
            line = process.stdout.readline()

            if line:
                print(line, end="", flush=True)
                collected.append(line)

        code = process.poll()

        if code is not None:
            if process.stdout is not None:
                rest = process.stdout.read()
                if rest:
                    print(rest, end="", flush=True)
                    collected.append(rest)

            elapsed = time.monotonic() - started
            return code, "".join(collected), elapsed

        elapsed = time.monotonic() - started

        if elapsed > timeout_seconds:
            process.kill()
            try:
                process.wait(timeout=10)
            except Exception:
                pass

            print(
                f"\nERRO: timeout interno após {timeout_seconds} segundos.",
                flush=True,
            )

            return 124, "".join(collected), elapsed

        time.sleep(0.1)


def compile_python_files() -> Dict:
    header("1. VALIDAÇÃO DE SINTAXE")

    rows = []
    all_ok = True

    for filename in CRITICAL_PYTHON_FILES:
        path = ROOT / filename

        if not path.exists():
            print(f"{filename:<35} AUSENTE", flush=True)

            rows.append(
                {
                    "file": filename,
                    "exists": False,
                    "compile_ok": False,
                }
            )

            all_ok = False
            continue

        try:
            compile(
                path.read_text(encoding="utf-8"),
                filename,
                "exec",
            )

            print(f"{filename:<35} OK", flush=True)

            rows.append(
                {
                    "file": filename,
                    "exists": True,
                    "compile_ok": True,
                }
            )

        except Exception as exc:
            print(f"{filename:<35} ERRO: {exc}", flush=True)

            rows.append(
                {
                    "file": filename,
                    "exists": True,
                    "compile_ok": False,
                    "error": str(exc),
                }
            )

            all_ok = False

    return {"ok": all_ok, "files": rows}


def run_essential_audits() -> Dict:
    header("2. AUDITORIAS ESSENCIAIS")

    rows = []
    all_ok = True

    for filename, description, timeout_seconds in ESSENTIAL_AUDITS:
        path = ROOT / filename

        if not path.exists():
            print(f"{filename}: AUSENTE", flush=True)
            rows.append(
                {
                    "file": filename,
                    "description": description,
                    "exists": False,
                    "ok": False,
                }
            )
            all_ok = False
            break

        code, output, elapsed = stream_command(
            [PYTHON, "-u", filename],
            f"{description} — {filename}",
            timeout_seconds=timeout_seconds,
        )

        ok = code == 0

        rows.append(
            {
                "file": filename,
                "description": description,
                "exists": True,
                "returncode": code,
                "elapsed_seconds": round(elapsed, 2),
                "ok": ok,
                "output_tail": output[-3000:],
            }
        )

        if not ok:
            all_ok = False
            print(
                f"\nERRO: {filename} não passou. O main.py não será executado.",
                flush=True,
            )
            break

    return {"ok": all_ok, "audits": rows}


def extract_final_portfolio(main_output: str) -> Dict[str, List[str]]:
    portfolio: Dict[str, List[str]] = {}

    match = re.search(
        r"10\. CARTEIRA FINAL — APÓS FRONTEIRA"
        r"(.*?)"
        r"(?:11\. CLASSIFICAÇÃO DE ENTRADA|\Z)",
        main_output,
        flags=re.S,
    )

    if not match:
        return portfolio

    block = match.group(1)

    for sector in [
        "Health Care",
        "Industrials",
        "Information Technology",
    ]:
        line = re.search(
            rf"{re.escape(sector)}\s*:\s*([A-Z0-9,\-\s]+)",
            block,
        )

        if line:
            portfolio[sector] = [
                ticker.strip()
                for ticker in line.group(1).strip().split(",")
                if ticker.strip()
            ]

    return portfolio


def validate_main_output(main_output: str, returncode: int) -> Dict:
    header("4. VALIDAÇÃO DO MOTOR PRINCIPAL")

    checks = {
        "config_validated":
            "Configuração validada." in main_output,

        "selection_stage":
            "SELEÇÃO FUNDAMENTAL — TOP 5 PURO" in main_output,

        "boundary_stage":
            "TESTE FINAL DA FRONTEIRA — CÉLULA 31" in main_output,

        "entry_41_stage":
            "CLASSIFICAÇÃO DE ENTRADA — HISTÓRICO CÉLULA 41"
            in main_output,

        "entry_structure_ok":
            "Estrutura OK         : True" in main_output,

        "main_success":
            "STATUS: SUCESSO" in main_output,

        "fundamental_errors_zero":
            bool(
                re.search(
                    r"Empresas com erro\s*:\s*0",
                    main_output,
                )
            ),
    }

    factor_checks = {
        "health_care_financial_strength":
            bool(
                re.search(
                    r"Health Care\s+financial_strength",
                    main_output,
                    flags=re.I,
                )
            ),

        "industrials_growth":
            bool(
                re.search(
                    r"Industrials\s+growth",
                    main_output,
                    flags=re.I,
                )
            ),

        "technology_financial_strength":
            bool(
                re.search(
                    r"Information Technology\s+financial_strength",
                    main_output,
                    flags=re.I,
                )
            ),
    }

    portfolio = extract_final_portfolio(main_output)

    sector_counts = {
        sector: len(tickers)
        for sector, tickers in portfolio.items()
    }

    portfolio_5_5_5 = (
        len(portfolio) == 3
        and all(
            sector_counts.get(sector) == 5
            for sector in [
                "Health Care",
                "Industrials",
                "Information Technology",
            ]
        )
    )

    all_tickers = [
        ticker
        for tickers in portfolio.values()
        for ticker in tickers
    ]

    portfolio_15_unique = (
        len(all_tickers) == 15
        and len(set(all_tickers)) == 15
    )

    main_ok = bool(
        returncode == 0
        and all(checks.values())
        and all(factor_checks.values())
        and portfolio_5_5_5
        and portfolio_15_unique
    )

    print(f"Return code do main.py                    : {returncode}", flush=True)
    print(f"Configuração validada                     : {checks['config_validated']}", flush=True)
    print(f"Seleção fundamental presente              : {checks['selection_stage']}", flush=True)
    print(f"Fronteira Célula 31 presente              : {checks['boundary_stage']}", flush=True)
    print(f"Entry Célula 41 presente                  : {checks['entry_41_stage']}", flush=True)
    print(f"Estrutura do Entry OK                     : {checks['entry_structure_ok']}", flush=True)
    print(f"Fundamentos sem erro                      : {checks['fundamental_errors_zero']}", flush=True)
    print(f"Health Care = financial_strength          : {factor_checks['health_care_financial_strength']}", flush=True)
    print(f"Industrials = growth                      : {factor_checks['industrials_growth']}", flush=True)
    print(f"Technology = financial_strength           : {factor_checks['technology_financial_strength']}", flush=True)
    print(f"Carteira final 5/5/5                      : {portfolio_5_5_5}", flush=True)
    print(f"Carteira final = 15 tickers únicos        : {portfolio_15_unique}", flush=True)
    print(f"STATUS: SUCESSO no motor                  : {checks['main_success']}", flush=True)

    if portfolio:
        print("\nCarteira dinâmica desta execução:", flush=True)
        for sector, tickers in portfolio.items():
            print(
                f"  {sector:<28}: {', '.join(tickers)}",
                flush=True,
            )

    return {
        "ok": main_ok,
        "returncode": returncode,
        "checks": checks,
        "factor_checks": factor_checks,
        "portfolio": portfolio,
        "sector_counts": sector_counts,
        "portfolio_5_5_5": portfolio_5_5_5,
        "portfolio_15_unique": portfolio_15_unique,
    }


def save_report(report: Dict) -> None:
    REPORT_JSON.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    REPORT_TXT.write_text(
        "\n".join(
            [
                "AUDITORIA FINAL DE REPRODUTIBILIDADE — COLAB x GITHUB",
                "=" * 80,
                "",
                f"Gerada em: {report['generated_at']}",
                f"Sintaxe: {report['syntax']['ok']}",
                f"Auditorias essenciais: {report['essential_audits']['ok']}",
                f"Motor principal: {report['main']['ok']}",
                "",
                f"STATUS FINAL: {report['status']}",
                "",
                "Interpretação:",
                (
                    "A carteira pode mudar com os dados atuais. "
                    "A aprovação demonstra que a mesma metodologia científica "
                    "está sendo aplicada pelo GitHub."
                ),
            ]
        ),
        encoding="utf-8",
    )


def fail_early(
    syntax: Dict,
    essential_audits: Dict,
    status: str,
) -> None:

    report = {
        "generated_at": datetime.now().isoformat(),
        "syntax": syntax,
        "essential_audits": essential_audits,
        "main": {
            "ok": False,
            "skipped": True,
        },
        "approved": False,
        "status": status,
    }

    save_report(report)

    header("RESULTADO FINAL")
    print(f"STATUS: {status}", flush=True)

    raise SystemExit(1)


def main() -> None:
    started = datetime.now()

    header(
        "AUDITORIA FINAL DE REPRODUTIBILIDADE — COLAB x GITHUB"
    )

    print(
        f"Início: {started:%d/%m/%Y %H:%M:%S}",
        flush=True,
    )

    syntax = compile_python_files()

    if not syntax["ok"]:
        fail_early(
            syntax=syntax,
            essential_audits={
                "ok": False,
                "audits": [],
            },
            status=(
                "AUDITORIA FINAL NÃO APROVADA — "
                "ERRO DE SINTAXE/ARQUIVO"
            ),
        )

    essential_audits = run_essential_audits()

    if not essential_audits["ok"]:
        fail_early(
            syntax=syntax,
            essential_audits=essential_audits,
            status=(
                "AUDITORIA FINAL NÃO APROVADA — "
                "AUDITORIA ESSENCIAL FALHOU"
            ),
        )

    main_code, main_output, main_elapsed = stream_command(
        [PYTHON, "-u", "main.py"],
        "3. EXECUÇÃO PONTA A PONTA — main.py",
        timeout_seconds=1800,
    )

    main_validation = validate_main_output(
        main_output=main_output,
        returncode=main_code,
    )

    main_validation["elapsed_seconds"] = round(
        main_elapsed,
        2,
    )

    approved = bool(
        syntax["ok"]
        and essential_audits["ok"]
        and main_validation["ok"]
    )

    finished = datetime.now()

    status = (
        "AUDITORIA FINAL APROVADA — "
        "METODOLOGIA COLAB = IMPLEMENTAÇÃO GITHUB"
        if approved
        else
        "AUDITORIA FINAL NÃO APROVADA — "
        "DIVERGÊNCIA ENCONTRADA"
    )

    report = {
        "generated_at": finished.isoformat(),
        "duration": str(finished - started),
        "syntax": syntax,
        "essential_audits": essential_audits,
        "main": main_validation,
        "approved": approved,
        "status": status,
    }

    save_report(report)

    header("5. RESULTADO FINAL")

    print(
        f"Sintaxe dos arquivos críticos             : {syntax['ok']}",
        flush=True,
    )

    print(
        f"Auditorias essenciais                     : {essential_audits['ok']}",
        flush=True,
    )

    print(
        f"Motor principal ponta a ponta             : {main_validation['ok']}",
        flush=True,
    )

    print(
        f"Tempo total                               : {finished - started}",
        flush=True,
    )

    print(
        f"\nSTATUS: {status}",
        flush=True,
    )

    print("\nArquivos:", flush=True)
    print(
        f"  {REPORT_JSON.relative_to(ROOT)}",
        flush=True,
    )
    print(
        f"  {REPORT_TXT.relative_to(ROOT)}",
        flush=True,
    )

    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
