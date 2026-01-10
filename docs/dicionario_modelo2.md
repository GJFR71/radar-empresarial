# Dicionário de Dados – Modelo 2 (Descontinuidade)

## Objetivo do Modelo
Estimar o **risco de a empresa encerrar suas operações**, com base
nas informações públicas mais recentes disponíveis.

## Variável Alvo
| Variável | Descrição |
|--------|-----------|
| y_descontinuidade | Indica empresas sem estabelecimentos ativos na base pública mais recente |

## Variáveis Explicativas

| Variável | Justificativa de Uso |
|--------|----------------------|
| qtd_ativos | Indicador direto de operação |
| qtd_estabelecimentos | Estrutura operacional |
| idade_empresa_dias | Tempo de sobrevivência |
| pgfn_valor_sum | Pressão financeira acumulada |
| pgfn_qtd_ajuizadas | Histórico de conflitos fiscais |

## Observação Importante
Este modelo **não prevê falência jurídica**, mas um **indicador operacional**
baseado na ausência de estabelecimentos ativos nos dados públicos.
