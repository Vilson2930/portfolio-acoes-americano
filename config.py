# ======================================================================================
# PORTFOLIO ACOES AMERICANO
# config.py
# ======================================================================================
#
# Arquivo central de configuração.
#
# PRINCÍPIO DO MODELO
# -------------------
# O sistema NÃO congela os tickers.
#
# Ele congela:
#
#   1. os três setores;
#   2. cinco ações por setor;
#   3. o fator vencedor de seleção de cada setor;
#   4. as regras de entrada validadas no estudo.
#
# Fluxo:
#
# UNIVERSO
#     ↓
# SETORES FIXOS
#     ↓
# RANKING FUNDAMENTAL
#     ↓
# TOP 5 DE CADA SETOR
#     ↓
# 15 AÇÕES
#     ↓
# FILTRO DE ENTRADA
#
# ======================================================================================


# ======================================================================================
# 1. CONFIGURAÇÃO GERAL
# ======================================================================================

PROJECT_NAME = "portfolio-acoes-americano"

PORTFOLIO_SIZE = 15

NUMBER_OF_SECTORS = 3

STOCKS_PER_SECTOR = 5


# ======================================================================================
# 2. SETORES FIXOS
# ======================================================================================

SECTORS = [

    "Health Care",

    "Industrials",

    "Information Technology",
]


# ======================================================================================
# 3. FATOR DE SELEÇÃO POR SETOR
# ======================================================================================
#
# Resultado do estudo histórico:
#
# Health Care
#     Financial Strength
#
# Industrials
#     Growth
#
# Information Technology
#     Financial Strength
#
# Esses fatores determinam QUAIS ações entram no Top 5.
#
# Regra vencedora validada nas Células 15, 16 e 17 do estudo:
#     Health Care              -> Financial Strength
#     Industrials              -> Growth
#     Information Technology   -> Financial Strength
#
# ======================================================================================

SELECTION_FACTORS = {

    "Health Care":
        "financial_strength",

    "Industrials":
        "growth",

    "Information Technology":
        "financial_strength",
}


# ======================================================================================
# 4. QUANTIDADE DE AÇÕES POR SETOR
# ======================================================================================

SECTOR_TARGETS = {

    "Health Care":
        5,

    "Industrials":
        5,

    "Information Technology":
        5,
}


# ======================================================================================
# 5. REGRAS DE ENTRADA
# ======================================================================================
#
# Health Care
# -----------
# Validação histórica aprovada.
#
# 10% valuation
# 80% desconto de preço
# 10% fundamentos preservados
#
#
# Industrials
# -----------
# Modelo historicamente útil, porém permaneceu CONDICIONAL.
#
# 20% desconto
# 80% fundamentos
#
# Por segurança:
#     ENTRADA FORTE não será permitida.
#
#
# Information Technology
# ----------------------
# Valuation / desconto / fundamentos não funcionaram bem como timing.
#
# O estudo encontrou:
#
#     Momentum 6M + Momentum 12M
#
# como regra alternativa aprovada.
#
# ======================================================================================

ENTRY_RULES = {

    "Health Care": {

        "method":
            "weighted_entry_score",

        "status":
            "APROVADO",

        "valuation_weight":
            0.10,

        "discount_weight":
            0.80,

        "fundamental_weight":
            0.10,

        "allow_strong_entry":
            True,
    },


    "Industrials": {

        "method":
            "weighted_entry_score",

        "status":
            "CONDICIONAL",

        "valuation_weight":
            0.00,

        "discount_weight":
            0.20,

        "fundamental_weight":
            0.80,

        "allow_strong_entry":
            False,
    },


    "Information Technology": {

        "method":
            "momentum",

        "status":
            "APROVADO",

        "momentum_6m_weight":
            0.50,

        "momentum_12m_weight":
            0.50,

        "allow_strong_entry":
            True,
    },
}


# ======================================================================================
# 6. CLASSIFICAÇÃO DE ENTRADA
# ======================================================================================
#
# A classificação utiliza o percentil histórico do sinal.
#
# Quanto maior o percentil:
#     melhor o momento de entrada.
#
# ======================================================================================

ENTRY_THRESHOLDS = {

    "ENTRADA_FORTE":
        0.75,

    "ENTRADA":
        0.50,

    "AGUARDAR":
        0.25,
}


# ======================================================================================
# 7. HISTÓRICO MÍNIMO
# ======================================================================================

MIN_HISTORY_MONTHS = 24


# ======================================================================================
# 8. MOMENTUM
# ======================================================================================

MOMENTUM_6M_DAYS = 126

MOMENTUM_12M_DAYS = 252


# ======================================================================================
# 9. DESCONTO DE PREÇO
# ======================================================================================
#
# Componentes utilizados no estudo:
#
#   • drawdown da máxima de 52 semanas;
#   • drawdown da máxima de 3 anos;
#   • distância da média móvel de 200 dias;
#   • posição dentro da faixa de preço de 3 anos.
#
# ======================================================================================

DISCOUNT_COMPONENTS = [

    "drawdown_52w",

    "drawdown_3y",

    "distance_ma200",

    "price_position_3y",
]


# ======================================================================================
# 10. VALUATION
# ======================================================================================

VALUATION_METRICS = [

    "pe",

    "pb",

    "ps",

    "p_ocf",

    "p_fcf",
]


# ======================================================================================
# 11. FUNDAMENTOS PRESERVADOS
# ======================================================================================

FUNDAMENTAL_COMPONENTS = [

    "revenue_growth_yoy",

    "net_income_growth_yoy",

    "operating_cash_flow_growth_yoy",

    "diluted_eps_growth_yoy",

    "net_margin",

    "ocf_margin",

    "fcf_margin",
]


# ======================================================================================
# 12. REGRA DE FRONTEIRA — 5º VS 6º
# ======================================================================================
#
# Não queremos trocar uma ação diariamente por uma diferença insignificante.
#
# A empresa nº 6 só deverá substituir a nº 5 quando existir diferença material.
#
# Este valor poderá ser refinado depois com os mesmos critérios usados
# na análise de fronteira do estudo.
#
# ======================================================================================

FRONTIER_MIN_RELATIVE_GAP = 0.03


# ======================================================================================
# 13. CONTROLE DE DADOS
# ======================================================================================

MAX_PRICE_DATA_AGE_DAYS = 7

MAX_FUNDAMENTAL_DATA_AGE_DAYS = 180


# ======================================================================================
# 14. SEC
# ======================================================================================

SEC_COMPANY_FACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/"
)


SEC_TICKER_MAP_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)


SEC_USER_AGENT = (
    "portfolio-acoes-americano research"
)


# ======================================================================================
# 15. DIRETÓRIOS
# ======================================================================================

DATA_DIR = "data"

OUTPUT_DIR = "output"

CACHE_DIR = "cache"


# ======================================================================================
# 16. ARQUIVOS DE SAÍDA
# ======================================================================================

FINAL_CSV = (
    f"{OUTPUT_DIR}/portfolio_ranking.csv"
)


FINAL_PDF = (
    f"{OUTPUT_DIR}/portfolio_report.pdf"
)


CURRENT_PORTFOLIO_FILE = (
    f"{OUTPUT_DIR}/current_portfolio.csv"
)


# ======================================================================================
# 17. PROTEÇÕES ESTRUTURAIS
# ======================================================================================

STRICT_SECTOR_STRUCTURE = True

ALLOW_SECTOR_CHANGE = False

ALLOW_PORTFOLIO_SIZE_CHANGE = False

ALLOW_STOCK_ROTATION = True


# ======================================================================================
# 18. VALIDAÇÃO DA CONFIGURAÇÃO
# ======================================================================================

def validate_config():

    if len(SECTORS) != NUMBER_OF_SECTORS:

        raise RuntimeError(
            "Número de setores diferente do definido."
        )


    if sum(SECTOR_TARGETS.values()) != PORTFOLIO_SIZE:

        raise RuntimeError(
            "A soma das ações por setor não é igual a 15."
        )


    expected_selection_factors = {

        "Health Care":
            "financial_strength",

        "Industrials":
            "growth",

        "Information Technology":
            "financial_strength",
    }


    for sector in SECTORS:

        if sector not in SELECTION_FACTORS:

            raise RuntimeError(
                f"Fator de seleção ausente para {sector}"
            )


        if (
            SELECTION_FACTORS[sector]
            !=
            expected_selection_factors[sector]
        ):

            raise RuntimeError(
                f"Fator de seleção divergente do estudo em {sector}: "
                f"{SELECTION_FACTORS[sector]} != "
                f"{expected_selection_factors[sector]}"
            )


        if sector not in ENTRY_RULES:

            raise RuntimeError(
                f"Regra de entrada ausente para {sector}"
            )


        if SECTOR_TARGETS[sector] != STOCKS_PER_SECTOR:

            raise RuntimeError(
                f"{sector} não possui exatamente 5 ações."
            )


    return True


# ======================================================================================
# 19. EXECUÇÃO DIRETA
# ======================================================================================

if __name__ == "__main__":

    validate_config()

    print(
        "Configuração validada com sucesso."
    )

    print(
        f"Portfólio: {PORTFOLIO_SIZE} ações"
    )

    print(
        f"Setores: {NUMBER_OF_SECTORS}"
    )

    print(
        f"Ações por setor: {STOCKS_PER_SECTOR}"
    )

    print(
        "\nSetores e fatores:"
    )


    for sector in SECTORS:

        print(
            f"  {sector}: "
            f"{SELECTION_FACTORS[sector]}"
        )
