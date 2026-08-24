# Portfolio Ações Americano

Motor quantitativo para seleção e priorização de entrada em ações americanas.

O sistema foi construído a partir de um estudo histórico que definiu:

- 3 setores fixos
- 5 ações por setor
- 15 ações no total
- fatores específicos de seleção por setor
- regras específicas de entrada por setor
- estrutura equal weight
- tickers dinâmicos

---

## Objetivo do motor

O sistema separa duas decisões:

### 1. Quais empresas devem fazer parte da carteira?

Essa etapa é feita pelo `selection.py`.

### 2. Entre as empresas selecionadas, quais apresentam melhor momento de compra?

Essa etapa é feita pelo `entry.py`.

Uma empresa pode continuar aprovada para a carteira e, ao mesmo tempo, receber:

- ENTRADA FORTE
- ENTRADA
- AGUARDAR
- NÃO COMPRAR AGORA

Portanto:

> seleção da empresa e momento de entrada são decisões diferentes.

---

# Arquitetura da carteira

A arquitetura é fixa:

| Setor | Quantidade |
|---|---:|
| Health Care | 5 |
| Industrials | 5 |
| Information Technology | 5 |
| **Total** | **15** |

Os setores não mudam.

O número de ações não muda.

Os tickers podem mudar conforme os fundamentos das empresas mudam.

---

# Seleção das ações

## Health Care

Critério:

**Financial Strength**

Componentes:

- Cash / Assets
- Debt / Assets
- Debt / Equity

Quanto maior o caixa relativo, melhor.

Quanto menor o endividamento, melhor.

São necessários pelo menos 2 dos 3 componentes.

---

## Industrials

Critério:

**Growth**

Componentes:

- Revenue Growth
- EPS Growth
- Operating Cash Flow Growth

São necessários pelo menos 2 dos 3 componentes.

---

## Information Technology

Critério:

**Quality**

Componentes:

- ROA
- ROE
- Operating Margin
- Net Margin

São necessários pelo menos 3 dos 4 componentes.

---

# Tratamento dos fatores

Antes do ranking:

1. os dados fundamentais são coletados;
2. valores extremos são tratados;
3. é aplicada winsorização P5-P95;
4. cada métrica é convertida em percentil dentro do próprio setor;
5. os percentis válidos são combinados pela média;
6. as empresas são ordenadas pelo score do fator.

O resultado é o Top 5 de cada setor.

---

# Regra da fronteira

O sistema também monitora a fronteira entre:

- 5ª colocada
- 6ª colocada

Isso existe para evitar trocas excessivas por diferenças insignificantes de score.

A função dessa auditoria é reduzir turnover e evitar substituir uma empresa por outra praticamente equivalente.

---

# Dados fundamentais

Os fundamentos são obtidos principalmente através do:

**SEC Company Facts**

O sistema utiliza a data `filed` como data de disponibilidade da informação.

Isso significa que um dado fundamental só pode ser utilizado depois de ter sido oficialmente protocolado na SEC.

Essa regra reduz risco de:

**look-ahead bias**

---

# Momento de entrada

Depois que as 15 empresas são selecionadas, o sistema calcula o momento de entrada.

Cada setor possui uma regra própria.

---

## Health Care

Regra validada:

- 10% Valuation
- 80% Desconto
- 10% Fundamentos

O desconto possui peso dominante.

O objetivo é priorizar empresas aprovadas que estejam negociando em condições mais atrativas.

---

## Industrials

Regra:

- 20% Desconto
- 80% Fundamentos

Status:

**CONDICIONAL**

Os fundamentos têm peso dominante.

Por regra de segurança, Industrials não recebe classificação:

**ENTRADA FORTE**

O maior sinal permitido é:

**ENTRADA**

---

## Information Technology

A metodologia tradicional de valuation/desconto não apresentou a mesma robustez.

A regra aprovada foi:

**Momentum 6M + Momentum 12M**

O sistema combina a força relativa de preço dos últimos 6 e 12 meses.

---

# Valuation

O módulo de valuation utiliza, quando disponíveis:

- P/E
- P/B
- P/S
- P/Operating Cash Flow
- P/Free Cash Flow

O objetivo não é comparar diretamente empresas diferentes.

O valuation deve ser analisado principalmente contra o:

**próprio histórico da empresa**

---

# Desconto de preço

O estudo utiliza:

- drawdown da máxima de 52 semanas
- drawdown da máxima de 3 anos
- distância da média móvel de 200 dias
- posição do preço dentro da faixa de 3 anos

O objetivo é identificar se uma empresa aprovada está negociando em condição historicamente mais descontada.

---

# Classificação de entrada

O resultado operacional é dividido em quatro grupos:

| Classificação | Interpretação |
|---|---|
| ENTRADA FORTE | Maior prioridade de compra |
| ENTRADA | Condição favorável |
| AGUARDAR | Sem vantagem suficiente no momento |
| NÃO COMPRAR AGORA | Condição desfavorável |

A classificação de entrada não remove automaticamente uma ação da carteira.

---

# Arquivos do projeto

## `config.py`

Centraliza:

- setores
- quantidade de ações
- fatores
- regras de entrada
- parâmetros do sistema
- caminhos de saída

---

## `data.py`

Responsável por:

- universo de empresas
- ticker / CIK
- SEC Company Facts
- classificação setorial
- preços históricos
- crescimento YoY
- snapshot fundamental
- regras point-in-time
- cache

---

## `selection.py`

Responsável por:

- Financial Strength
- Growth
- Quality
- winsorização
- percentis
- scores
- Top 5 de cada setor
- auditoria da fronteira

---

## `entry.py`

Responsável por:

- valuation
- desconto
- fundamentos preservados
- momentum 6M
- momentum 12M
- score de entrada
- prioridade de compra
- classificação final

---

## `report.py`

Responsável por gerar:

- portfolio_ranking.csv
- current_portfolio.csv
- portfolio_report.pdf

---

## `main.py`

É o orquestrador.

Executa:

```text
UNIVERSO
    ↓
DADOS
    ↓
FUNDAMENTOS
    ↓
RANKING SETORIAL
    ↓
TOP 5 POR SETOR
    ↓
15 AÇÕES
    ↓
FILTRO DE ENTRADA
    ↓
PRIORIDADE DE COMPRA
    ↓
CSV + PDF
