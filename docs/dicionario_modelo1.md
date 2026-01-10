# Dicionário de Dados – Modelo 1 (Saúde Fiscal)

## Objetivo do Modelo
Avaliar a **capacidade da empresa de manter impostos em dia**, apoiando
priorização comercial e análise de risco fiscal.

## Variável Alvo
| Variável | Descrição |
|--------|-----------|
| y_risco_fiscal | Indicador de risco fiscal baseado no histórico de inscrições na PGFN |

## Variáveis Explicativas

| Variável | Justificativa de Uso |
|--------|----------------------|
| pgfn_valor_sum | Dívidas maiores indicam maior pressão fiscal |
| pgfn_qtd_inscricoes | Reincidência de débitos |
| pgfn_qtd_ajuizadas | Indica maior severidade do passivo |
| pgfn_pct_ajuizadas | Proporção de judicialização |
| idade_empresa_dias | Empresas mais antigas tendem a padrões fiscais mais estáveis |

## Observação
As variáveis foram selecionadas priorizando **explicabilidade**
e **interpretação direta para áreas comerciais**.
