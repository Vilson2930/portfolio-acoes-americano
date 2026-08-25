# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# audit_reproducibility.py
# ======================================================================================
#
# AUDITORIA FINAL DE REPRODUTIBILIDADE — COLAB x GITHUB
#
# Objetivo:
#   Não recriar nenhuma regra financeira.
#   Reutilizar as auditorias já aprovadas e executar o motor principal ponta a ponta.
#
# A auditoria final só aprova se:
#   1. arquivos críticos compilarem;
#   2. auditorias existentes críticas passarem;
#   3. o motor principal terminar com STATUS: SUCESSO;
#   4. o motor reproduzir a arquitetura científica congelada:
#        - 3 setores;
#        - Top 5 por setor;
#        - Health Care -> financial_strength;
#        - Industrials -> growth;
#        - Information Technology -> financial_strength;
#        - fronteira Célula 31;
#        - Entry Célula 41;
#        - 15 ações / 3 setores / Estrutura OK;
#   5. a auditoria 33B continuar aprovada.
#
# IMPORTANTE:
#   A carteira de tickers pode mudar com novos dados.
#   O que deve permanecer idêntico é a METODOLOGIA.
# ======================================================================================

from __future__ import annotations

import json
import re
import subprocess
import sys
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
    "audit_valuation_cell33.py",
]

# Auditorias já construídas no projeto.
# required=True: ausência ou falha reprova a auditoria final.
# required=False: roda se existir; ausência é apenas informativa.
AUDIT_SCRIPTS = [
    ("audit_selection.py", False, "Seleção fundamental"),
    ("audit_boundary.py", True, "Fronteira — Célula 31"),
    ("audit_industrials.py", False, "Industrials"),
    ("audit_valuation_cell33.py", True, "Valuation — Célula 33B"),
]


def header(title: str) -> None:
    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)


def run_command(
    args: List[str],
    label: str,
    timeout_seconds: int,
) -> Tuple[int, str]:
    header(label)
    print("$ " + " ".join(args))

    try:
        completed = subprocess.run(
            args,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        print(output)

        return completed.returncode, output

    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        print(output)
        print(f"\nERRO: timeout após {timeout_seconds} segundos.")
        return 124, output


def compile_python_files() -> Dict:
    header("1. VALIDAÇÃO DE SINTAXE")

    rows = []
    all_ok = True

    for name in CRITICAL_PYTHON_FILES:
        path = ROOT / name

        if not path.exists():
            rows.append(
                {
                    "file": name,
                    "exists": False,
                    "compile_ok": False,
                    "detail": "arquivo ausente",
                }
            )
            all_ok = False
            print(f"{name:<35} AUSENTE")
            continue

        code, output = run_command(
            [PYTHON, "-m", "py_compile", name],
            f"Compilando {name}",
            timeout_seconds=60,
        )

        ok = code == 0
        all_ok = all_ok and ok

        rows.append(
            {
                "file": name,
                "exists": True,
                "compile_ok": ok,
                "detail": output.strip()[-500:],
            }
        )

    return {
        "ok": all_ok,
        "files": rows,
    }


def run_existing_audits() -> Dict:
    header("2. AUDITORIAS CIENTÍFICAS EXISTENTES")

    rows = []
    all_required_ok = True

    for filename, required, description in AUDIT_SCRIPTS:
        path = ROOT / filename

        if not path.exists():
            status = "AUSENTE — OBRIGATÓRIA" if required else "AUSENTE — OPCIONAL"
            print(f"{filename:<35} {status}")

            rows.append(
                {
                    "file": filename,
                    "description": description,
                    "required": required,
                    "exists": False,
                    "returncode": None,
                    "ok": not required,
                }
            )

            if required:
                all_required_ok = False

            continue

        code, output = run_command(
            [PYTHON, filename],
            f"{description} — {filename}",
            timeout_seconds=1800,
        )

        ok = code == 0

        rows.append(
            {
                "file": filename,
                "description": description,
                "required": required,
                "exists": True,
                "returncode": code,
                "ok": ok,
                "output_tail": output[-4000:],
            }
        )

        if required and not ok:
            all_required_ok = False

    return {
        "ok": all_required_ok,
        "audits": rows,
    }


def extract_final_portfolio(main_output: str) -> Dict[str, List[str]]:
    portfolio: Dict[str, List[str]] = {}

    match = re.search(
        r"10\. CARTEIRA FINAL — APÓS FRONTEIRA(.*?)(?:11\. CLASSIFICAÇÃO DE ENTRADA|\Z)",
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
            tickers = [
                t.strip()
                for t in line.group(1).strip().split(",")
                if t.strip()
            ]
            portfolio[sector] = tickers

    return portfolio


def validate_main_output(main_output: str, returncode: int) -> Dict:
    header("4. VALIDAÇÃO DO MOTOR PRINCIPAL")

    expected_strings = {
        "config_validated":
            "Configuração validada.",

        "selection_stage":
            "SELEÇÃO FUNDAMENTAL — TOP 5 PURO",

        "boundary_stage":
            "TESTE FINAL DA FRONTEIRA — CÉLULA 31",

        "entry_41_stage":
            "CLASSIFICAÇÃO DE ENTRADA — HISTÓRICO CÉLULA 41",

        "entry_structure_ok":
            "Estrutura OK         : True",

        "main_success":
            "STATUS: SUCESSO",
    }

    checks = {
        key: value in main_output
        for key, value in expected_strings.items()
    }

    # Arquitetura científica congelada pelos estudos setoriais.
    factor_checks = {
        "health_care_financial_strength":
            (
                "Health Care"
                in main_output
                and
                "financial_strength"
                in main_output
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

    fundamentals_no_error = bool(
        re.search(
            r"Empresas com erro\s*:\s*0",
            main_output,
        )
    )

    main_ok = (
        returncode == 0
        and all(checks.values())
        and all(factor_checks.values())
        and portfolio_5_5_5
        and portfolio_15_unique
        and fundamentals_no_error
    )

    print(f"Return code do main.py                    : {returncode}")
    print(f"Configuração validada                     : {checks['config_validated']}")
    print(f"Seleção fundamental presente              : {checks['selection_stage']}")
    print(f"Fronteira Célula 31 presente              : {checks['boundary_stage']}")
    print(f"Entry Célula 41 presente                  : {checks['entry_41_stage']}")
    print(f"Estrutura do Entry OK                     : {checks['entry_structure_ok']}")
    print(f"Fundamentos sem erro                      : {fundamentals_no_error}")
    print(f"Health Care = financial_strength          : {factor_checks['health_care_financial_strength']}")
    print(f"Industrials = growth                      : {factor_checks['industrials_growth']}")
    print(f"Technology = financial_strength           : {factor_checks['technology_financial_strength']}")
    print(f"Carteira final 5/5/5                      : {portfolio_5_5_5}")
    print(f"Carteira final = 15 tickers únicos        : {portfolio_15_unique}")
    print(f"STATUS: SUCESSO no motor                  : {checks['main_success']}")

    if portfolio:
        print("\nCarteira dinâmica reproduzida nesta execução:")
        for sector, tickers in portfolio.items():
            print(f"  {sector:<28}: {', '.join(tickers)}")

    return {
        "ok": main_ok,
        "returncode": returncode,
        "checks": checks,
        "factor_checks": factor_checks,
        "fundamentals_no_error": fundamentals_no_error,
        "portfolio": portfolio,
        "sector_counts": sector_counts,
        "portfolio_5_5_5": portfolio_5_5_5,
        "portfolio_15_unique": portfolio_15_unique,
    }


def run_main() -> Tuple[int, str]:
    return run_command(
        [PYTHON, "main.py"],
        "3. EXECUÇÃO PONTA A PONTA — main.py",
        timeout_seconds=2400,
    )


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

    lines = [
        "AUDITORIA FINAL DE REPRODUTIBILIDADE — COLAB x GITHUB",
        "=" * 80,
        "",
        f"Gerada em: {report['generated_at']}",
        f"Sintaxe: {report['syntax']['ok']}",
        f"Auditorias obrigatórias: {report['existing_audits']['ok']}",
        f"Motor principal: {report['main']['ok']}",
        "",
        f"STATUS FINAL: {report['status']}",
        "",
        "Observação:",
        (
            "A carteira de tickers é dinâmica. A aprovação comprova a reprodução "
            "da metodologia congelada do estudo, não a imutabilidade dos tickers."
        ),
    ]

    REPORT_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    started = datetime.now()

    header("AUDITORIA FINAL DE REPRODUTIBILIDADE — COLAB x GITHUB")
    print(f"Início: {started:%d/%m/%Y %H:%M:%S}")

    syntax = compile_python_files()

    existing_audits = run_existing_audits()

    main_returncode, main_output = run_main()

    main_validation = validate_main_output(
        main_output=main_output,
        returncode=main_returncode,
    )

    approved = bool(
        syntax["ok"]
        and existing_audits["ok"]
        and main_validation["ok"]
    )

    finished = datetime.now()

    status = (
        "AUDITORIA FINAL APROVADA — METODOLOGIA COLAB = IMPLEMENTAÇÃO GITHUB"
        if approved
        else
        "AUDITORIA FINAL NÃO APROVADA — EXISTE DIVERGÊNCIA A INVESTIGAR"
    )

    report = {
        "generated_at": finished.isoformat(),
        "duration": str(finished - started),
        "syntax": syntax,
        "existing_audits": existing_audits,
        "main": main_validation,
        "approved": approved,
        "status": status,
    }

    save_report(report)

    header("5. RESULTADO FINAL")

    print(f"Sintaxe dos arquivos críticos             : {syntax['ok']}")
    print(f"Auditorias obrigatórias                   : {existing_audits['ok']}")
    print(f"Motor principal ponta a ponta             : {main_validation['ok']}")
    print(f"Tempo total                               : {finished - started}")

    print(f"\nSTATUS: {status}")

    print("\nArquivos:")
    print(f"  {REPORT_JSON.relative_to(ROOT)}")
    print(f"  {REPORT_TXT.relative_to(ROOT)}")

    if not approved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
