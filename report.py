# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# report.py
# ======================================================================================
#
# RESPONSABILIDADE
# ---------------
# Gerar os arquivos finais do sistema:
#
#   1. ranking CSV
#   2. carteira atual CSV
#   3. relatório PDF
#
# O relatório NÃO altera seleção ou sinais.
# Apenas apresenta os resultados produzidos pelo sistema.
#
# ======================================================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)

from config import (
    PROJECT_NAME,
    PORTFOLIO_SIZE,
    NUMBER_OF_SECTORS,
    STOCKS_PER_SECTOR,
    FINAL_CSV,
    FINAL_PDF,
    CURRENT_PORTFOLIO_FILE,
    OUTPUT_DIR,
)


# ======================================================================================
# 1. DIRETÓRIO
# ======================================================================================

OUTPUT_PATH = Path(
    OUTPUT_DIR
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True,
)


# ======================================================================================
# 2. HELPERS
# ======================================================================================

def format_pct(
    value,
    decimals: int = 1,
) -> str:

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return "N/D"

        return (
            f"{value * 100:.{decimals}f}%"
        )

    except Exception:

        return "N/D"


def format_score(
    value,
    decimals: int = 3,
) -> str:

    try:

        value = float(
            value
        )

        if not np.isfinite(
            value
        ):
            return "N/D"

        return (
            f"{value:.{decimals}f}"
        )

    except Exception:

        return "N/D"


def safe_text(
    value,
) -> str:

    if pd.isna(
        value
    ):
        return "N/D"

    return str(
        value
    )


# ======================================================================================
# 3. VALIDAR RANKING
# ======================================================================================

def validate_ranking(
    ranking: pd.DataFrame,
):

    required = {
        "ticker",
        "sector",
        "entry_signal",
    }

    missing = (
        required
        -
        set(
            ranking.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"Ranking sem colunas obrigatórias: "
            f"{sorted(missing)}"
        )

    if (
        ranking[
            "ticker"
        ]
        .nunique()
        !=
        PORTFOLIO_SIZE
    ):

        raise RuntimeError(
            "Ranking final não possui exatamente "
            f"{PORTFOLIO_SIZE} ações."
        )

    sector_counts = (
        ranking
        .groupby(
            "sector"
        )[
            "ticker"
        ]
        .nunique()
    )

    if len(
        sector_counts
    ) != NUMBER_OF_SECTORS:

        raise RuntimeError(
            "Número de setores diferente "
            "da arquitetura definida."
        )

    if not (
        sector_counts
        ==
        STOCKS_PER_SECTOR
    ).all():

        raise RuntimeError(
            "A distribuição por setor não é 5/5/5."
        )

    return True


# ======================================================================================
# 4. SALVAR CSV FINAL
# ======================================================================================

def save_ranking_csv(
    ranking: pd.DataFrame,
) -> Path:

    path = Path(
        FINAL_CSV
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_csv(
        path,
        index=False,
    )

    return path


# ======================================================================================
# 5. SALVAR CARTEIRA ATUAL
# ======================================================================================

def save_current_portfolio(
    ranking: pd.DataFrame,
) -> Path:

    columns = [
        col
        for col in [
            "sector",
            "ticker",
            "selection_factor",
            "selection_score",
            "selection_rank",
        ]
        if col in ranking.columns
    ]

    current = (
        ranking[
            columns
        ]
        .copy()
    )

    path = Path(
        CURRENT_PORTFOLIO_FILE
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current.to_csv(
        path,
        index=False,
    )

    return path


# ======================================================================================
# 6. CONTAGEM DOS SINAIS
# ======================================================================================

def signal_summary(
    ranking: pd.DataFrame,
) -> Dict[str, int]:

    counts = (
        ranking[
            "entry_signal"
        ]
        .value_counts()
        .to_dict()
    )

    return {

        "ENTRADA FORTE":
            int(
                counts.get(
                    "ENTRADA FORTE",
                    0,
                )
            ),

        "ENTRADA":
            int(
                counts.get(
                    "ENTRADA",
                    0,
                )
            ),

        "AGUARDAR":
            int(
                counts.get(
                    "AGUARDAR",
                    0,
                )
            ),

        "NÃO COMPRAR AGORA":
            int(
                counts.get(
                    "NÃO COMPRAR AGORA",
                    0,
                )
            ),
    }


# ======================================================================================
# 7. ORDENAR PARA O RELATÓRIO
# ======================================================================================

def order_for_report(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    df = ranking.copy()

    signal_order = {

        "ENTRADA FORTE":
            1,

        "ENTRADA":
            2,

        "AGUARDAR":
            3,

        "NÃO COMPRAR AGORA":
            4,
    }

    df[
        "_signal_order"
    ] = (
        df[
            "entry_signal"
        ]
        .map(
            signal_order
        )
        .fillna(
            99
        )
    )

    sort_columns = [
        "_signal_order"
    ]

    ascending = [
        True
    ]

    if (
        "signal_percentile"
        in df.columns
    ):

        sort_columns.append(
            "signal_percentile"
        )

        ascending.append(
            False
        )

    if (
        "final_signal_score"
        in df.columns
    ):

        sort_columns.append(
            "final_signal_score"
        )

        ascending.append(
            False
        )

    df = (
        df
        .sort_values(
            sort_columns,
            ascending=ascending,
            na_position="last",
        )
        .drop(
            columns=[
                "_signal_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ======================================================================================
# 8. ESTILOS PDF
# ======================================================================================

def build_styles():

    styles = (
        getSampleStyleSheet()
    )

    title = ParagraphStyle(
        "TitleCustom",
        parent=styles[
            "Title"
        ],
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=10,
    )

    subtitle = ParagraphStyle(
        "SubtitleCustom",
        parent=styles[
            "Normal"
        ],
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    heading = ParagraphStyle(
        "HeadingCustom",
        parent=styles[
            "Heading2"
        ],
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        spaceBefore=8,
        spaceAfter=6,
    )

    body = ParagraphStyle(
        "BodyCustom",
        parent=styles[
            "BodyText"
        ],
        fontSize=8,
        leading=10,
    )

    small = ParagraphStyle(
        "SmallCustom",
        parent=styles[
            "BodyText"
        ],
        fontSize=7,
        leading=9,
    )

    return {
        "title":
            title,

        "subtitle":
            subtitle,

        "heading":
            heading,

        "body":
            body,

        "small":
            small,
    }


# ======================================================================================
# 9. TABELA RESUMO
# ======================================================================================

def build_summary_table(
    ranking: pd.DataFrame,
):

    summary = (
        signal_summary(
            ranking
        )
    )

    data = [

        [
            "Indicador",
            "Resultado",
        ],

        [
            "Ações",
            str(
                ranking[
                    "ticker"
                ]
                .nunique()
            ),
        ],

        [
            "Setores",
            str(
                ranking[
                    "sector"
                ]
                .nunique()
            ),
        ],

        [
            "Distribuição",
            "5 / 5 / 5",
        ],

        [
            "Entradas fortes",
            str(
                summary[
                    "ENTRADA FORTE"
                ]
            ),
        ],

        [
            "Entradas",
            str(
                summary[
                    "ENTRADA"
                ]
            ),
        ],

        [
            "Aguardar",
            str(
                summary[
                    "AGUARDAR"
                ]
            ),
        ],

        [
            "Não comprar agora",
            str(
                summary[
                    "NÃO COMPRAR AGORA"
                ]
            ),
        ],
    ]

    table = Table(
        data,
        colWidths=[
            55 * mm,
            35 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8,
                ),

                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "CENTER",
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    return table


# ======================================================================================
# 10. TABELA PRINCIPAL
# ======================================================================================

def build_ranking_table(
    ranking: pd.DataFrame,
):

    df = (
        order_for_report(
            ranking
        )
    )

    header = [

        "Setor",
        "Prior.",
        "Ticker",
        "Valuation",
        "Desconto",
        "Fundamentos",
        "Score",
        "Percentil",
        "Entrada",
    ]

    data = [
        header
    ]

    for _, row in df.iterrows():

        data.append(
            [
                safe_text(
                    row.get(
                        "sector"
                    )
                ),

                safe_text(
                    row.get(
                        "buy_priority_sector"
                    )
                ),

                safe_text(
                    row.get(
                        "ticker"
                    )
                ),

                safe_text(
                    row.get(
                        "valuation_status"
                    )
                ),

                safe_text(
                    row.get(
                        "discount_status"
                    )
                ),

                safe_text(
                    row.get(
                        "fundamental_status"
                    )
                ),

                format_score(
                    row.get(
                        "final_signal_score"
                    )
                ),

                format_pct(
                    row.get(
                        "signal_percentile"
                    )
                ),

                safe_text(
                    row.get(
                        "entry_signal"
                    )
                ),
            ]
        )

    table = Table(
        data,
        repeatRows=1,
        colWidths=[
            38 * mm,
            13 * mm,
            14 * mm,
            22 * mm,
            20 * mm,
            24 * mm,
            16 * mm,
            18 * mm,
            30 * mm,
        ],
    )

    style_commands = [

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.lightgrey,
        ),

        (
            "FONTNAME",
            (0, 0),
            (-1, 0),
            "Helvetica-Bold",
        ),

        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.35,
            colors.grey,
        ),

        (
            "FONTSIZE",
            (0, 0),
            (-1, -1),
            6.5,
        ),

        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "MIDDLE",
        ),

        (
            "ALIGN",
            (1, 1),
            (2, -1),
            "CENTER",
        ),

        (
            "ALIGN",
            (6, 1),
            (7, -1),
            "CENTER",
        ),

        (
            "ALIGN",
            (8, 1),
            (8, -1),
            "CENTER",
        ),
    ]

    # Destaque visual leve dos sinais

    for index, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):

        signal = row.get(
            "entry_signal"
        )

        if signal == "ENTRADA FORTE":

            style_commands.append(
                (
                    "FONTNAME",
                    (8, index),
                    (8, index),
                    "Helvetica-Bold",
                )
            )

        elif signal == "ENTRADA":

            style_commands.append(
                (
                    "FONTNAME",
                    (8, index),
                    (8, index),
                    "Helvetica-Bold",
                )
            )

    table.setStyle(
        TableStyle(
            style_commands
        )
    )

    return table


# ======================================================================================
# 11. TABELA POR SETOR
# ======================================================================================

def build_sector_table(
    ranking: pd.DataFrame,
    sector: str,
):

    df = (
        ranking[
            ranking[
                "sector"
            ]
            ==
            sector
        ]
        .copy()
    )

    if (
        "buy_priority_sector"
        in df.columns
    ):

        df = (
            df
            .sort_values(
                "buy_priority_sector"
            )
        )

    header = [

        "Prior.",
        "Ticker",
        "Seleção",
        "Valuation",
        "Desconto",
        "Fundamentos",
        "Percentil",
        "Sinal",
    ]

    data = [
        header
    ]

    for _, row in df.iterrows():

        data.append(
            [
                safe_text(
                    row.get(
                        "buy_priority_sector"
                    )
                ),

                safe_text(
                    row.get(
                        "ticker"
                    )
                ),

                format_score(
                    row.get(
                        "selection_score"
                    )
                ),

                safe_text(
                    row.get(
                        "valuation_status"
                    )
                ),

                safe_text(
                    row.get(
                        "discount_status"
                    )
                ),

                safe_text(
                    row.get(
                        "fundamental_status"
                    )
                ),

                format_pct(
                    row.get(
                        "signal_percentile"
                    )
                ),

                safe_text(
                    row.get(
                        "entry_signal"
                    )
                ),
            ]
        )

    table = Table(
        data,
        repeatRows=1,
        colWidths=[
            15 * mm,
            18 * mm,
            20 * mm,
            28 * mm,
            25 * mm,
            30 * mm,
            20 * mm,
            35 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey,
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.35,
                    colors.grey,
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),

                (
                    "ALIGN",
                    (0, 1),
                    (2, -1),
                    "CENTER",
                ),

                (
                    "ALIGN",
                    (6, 1),
                    (-1, -1),
                    "CENTER",
                ),
            ]
        )
    )

    return table



# ======================================================================================
# 11B. EXPLICAÇÃO INDIVIDUAL DAS AÇÕES
# ======================================================================================

def build_stock_explanation(
    row: pd.Series,
    styles,
):
    """
    Traduz os resultados já calculados pelo motor em linguagem simples.
    Não recalcula nem altera nenhum sinal.
    """

    ticker = safe_text(
        row.get(
            "ticker"
        )
    )

    sector = safe_text(
        row.get(
            "sector"
        )
    )

    signal = safe_text(
        row.get(
            "entry_signal"
        )
    )

    valuation = safe_text(
        row.get(
            "valuation_status"
        )
    )

    discount = safe_text(
        row.get(
            "discount_status"
        )
    )

    fundamentals = safe_text(
        row.get(
            "fundamental_status"
        )
    )

    selection_score = format_score(
        row.get(
            "selection_score"
        )
    )

    final_score = format_score(
        row.get(
            "final_signal_score"
        )
    )

    percentile = format_pct(
        row.get(
            "signal_percentile"
        )
    )

    if signal == "ENTRADA FORTE":

        interpretation = (
            "O ponto de entrada é excepcional dentro das regras "
            "históricas do modelo e recebe a maior prioridade operacional."
        )

    elif signal == "ENTRADA":

        interpretation = (
            "O ponto de entrada é favorável e foi aprovado "
            "pelos critérios quantitativos do setor."
        )

    elif signal == "AGUARDAR":

        interpretation = (
            "A empresa continua entre as 15 selecionadas, mas o timing "
            "atual ainda não apresenta vantagem suficiente para nova compra."
        )

    elif signal == "NÃO COMPRAR AGORA":

        interpretation = (
            "A empresa permanece estruturalmente selecionada, porém o modelo "
            "não recomenda nova compra no momento."
        )

    else:

        interpretation = (
            "O motor não apresentou uma classificação operacional completa."
        )

    text = (
        f"<b>{ticker}</b> — {sector}<br/>"
        f"<b>Sinal:</b> {signal}. {interpretation}<br/>"
        f"<b>Seleção:</b> score {selection_score}. "
        f"<b>Valuation:</b> {valuation}. "
        f"<b>Desconto:</b> {discount}. "
        f"<b>Fundamentos:</b> {fundamentals}. "
        f"<b>Score final:</b> {final_score}. "
        f"<b>Percentil:</b> {percentile}."
    )

    return Paragraph(
        text,
        styles[
            "body"
        ],
    )


# ======================================================================================
# 12. GERAR PDF
# ======================================================================================

def generate_pdf(
    ranking: pd.DataFrame,
    generated_at: Optional[datetime] = None,
) -> Path:

    validate_ranking(
        ranking
    )

    if generated_at is None:

        generated_at = (
            datetime.now()
        )

    pdf_path = Path(
        FINAL_PDF
    )

    pdf_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = (
        build_styles()
    )

    document = SimpleDocTemplate(
        str(
            pdf_path
        ),
        pagesize=landscape(
            A4
        ),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        title=(
            "Portfolio Ações Americano"
        ),
        author=PROJECT_NAME,
    )

    story = []

    # ------------------------------------------------------------------
    # CAPA
    # ------------------------------------------------------------------

    story.append(
        Paragraph(
            "PORTFOLIO AÇÕES AMERICANO",
            styles[
                "title"
            ],
        )
    )

    story.append(
        Paragraph(
            "Relatório diário de seleção e prioridade de entrada",
            styles[
                "subtitle"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Gerado em "
                f"{generated_at.strftime('%d/%m/%Y %H:%M')}"
            ),
            styles[
                "subtitle"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        build_summary_table(
            ranking
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    story.append(
        Paragraph(
            "Arquitetura do modelo",
            styles[
                "heading"
            ],
        )
    )

    architecture_text = (
        "<b>Health Care:</b> Financial Strength → "
        "10% Valuation + 80% Desconto + 10% Fundamentos.<br/>"
        "<b>Industrials:</b> Growth → "
        "20% Desconto + 80% Fundamentos — CONDICIONAL.<br/>"
        "<b>Information Technology:</b> Financial Strength → "
        "Momentum 6M + 12M.<br/><br/>"
        "Estrutura fixa: 3 setores, 5 ações por setor, "
        "15 ações totais. Os tickers podem mudar conforme "
        "o ranking fundamental."
    )

    story.append(
        Paragraph(
            architecture_text,
            styles[
                "body"
            ],
        )
    )

    story.append(
        PageBreak()
    )

    # ------------------------------------------------------------------
    # RANKING FINAL
    # ------------------------------------------------------------------

    story.append(
        Paragraph(
            "Ranking final de compra",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        build_ranking_table(
            ranking
        )
    )

    story.append(
        Spacer(
            1,
            7 * mm,
        )
    )

    story.append(
        Paragraph(
            (
                "<b>Leitura:</b> ENTRADA FORTE representa a maior "
                "prioridade de compra; ENTRADA representa condição "
                "favorável; AGUARDAR indica ausência de vantagem "
                "suficiente; NÃO COMPRAR AGORA indica condição "
                "desfavorável segundo o modelo."
            ),
            styles[
                "small"
            ],
        )
    )

    story.append(
        PageBreak()
    )

    # ------------------------------------------------------------------
    # DETALHAMENTO POR SETOR
    # ------------------------------------------------------------------

    for position, sector in enumerate(
        [
            "Health Care",
            "Industrials",
            "Information Technology",
        ]
    ):

        story.append(
            Paragraph(
                sector,
                styles[
                    "heading"
                ],
            )
        )

        if sector == "Health Care":

            description = (
                "Critério de seleção: Financial Strength. "
                "Timing: 10% Valuation + 80% Desconto + "
                "10% Fundamentos."
            )

        elif sector == "Industrials":

            description = (
                "Critério de seleção: Growth. "
                "Timing: 20% Desconto + 80% Fundamentos. "
                "Validação condicional; o setor não recebe "
                "classificação ENTRADA FORTE."
            )

        else:

            description = (
                "Critério de seleção: Financial Strength. "
                "Timing: Momentum combinado de 6 e 12 meses."
            )

        story.append(
            Paragraph(
                description,
                styles[
                    "body"
                ],
            )
        )

        story.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        story.append(
            build_sector_table(
                ranking,
                sector,
            )
        )

        if position < 2:

            story.append(
                Spacer(
                    1,
                    8 * mm,
                )
            )

    # ------------------------------------------------------------------
    # EXPLICAÇÃO AÇÃO POR AÇÃO
    # ------------------------------------------------------------------

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Explicação individual das 15 ações",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "Esta seção apenas traduz os resultados já produzidos pelo motor. "
                "Ela não cria novos critérios e não altera os sinais."
            ),
            styles[
                "small"
            ],
        )
    )

    story.append(
        Spacer(
            1,
            4 * mm,
        )
    )

    explained = (
        order_for_report(
            ranking
        )
    )

    for _, row in explained.iterrows():

        story.append(
            build_stock_explanation(
                row,
                styles,
            )
        )

        story.append(
            Spacer(
                1,
                4 * mm,
            )
        )

    # ------------------------------------------------------------------
    # OBSERVAÇÃO
    # ------------------------------------------------------------------

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    story.append(
        Paragraph(
            "Observação metodológica",
            styles[
                "heading"
            ],
        )
    )

    story.append(
        Paragraph(
            (
                "O relatório separa duas decisões: seleção das empresas "
                "e momento de entrada. Uma empresa pode permanecer entre "
                "as 15 selecionadas e, ao mesmo tempo, receber AGUARDAR "
                "ou NÃO COMPRAR AGORA. O sinal de entrada não remove "
                "automaticamente a empresa da carteira."
            ),
            styles[
                "body"
            ],
        )
    )

    document.build(
        story
    )

    return pdf_path


# ======================================================================================
# 13. GERAR TODOS OS RELATÓRIOS
# ======================================================================================

def generate_reports(
    ranking: pd.DataFrame,
    generated_at: Optional[datetime] = None,
) -> Dict[str, str]:

    validate_ranking(
        ranking
    )

    csv_path = (
        save_ranking_csv(
            ranking
        )
    )

    portfolio_path = (
        save_current_portfolio(
            ranking
        )
    )

    pdf_path = (
        generate_pdf(
            ranking,
            generated_at=generated_at,
        )
    )

    return {

        "ranking_csv":
            str(
                csv_path
            ),

        "current_portfolio":
            str(
                portfolio_path
            ),

        "pdf":
            str(
                pdf_path
            ),
    }


# ======================================================================================
# 14. TESTE DO MÓDULO
# ======================================================================================

if __name__ == "__main__":

    print(
        "=" * 100
    )

    print(
        "PORTFOLIO ACOES AMERICANO — REPORT ENGINE"
    )

    print(
        "=" * 100
    )

    print(
        "\nSaídas:"
    )

    print(
        f"  CSV ranking : {FINAL_CSV}"
    )

    print(
        f"  Carteira    : {CURRENT_PORTFOLIO_FILE}"
    )

    print(
        f"  PDF         : {FINAL_PDF}"
    )

    print(
        "\nReport engine carregado com sucesso."
    )
