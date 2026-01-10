# Dicionário de Dados – ABT Completa (PGFN + CNPJ)

## Descrição
Base Analítica consolidada por **CNPJ raiz**, construída a partir da integração
dos dados públicos da PGFN e do CNPJ.  
Esta base serve como insumo comum para os Modelos 1 e 2.

## Estrutura Geral
- Unidade de análise: CNPJ raiz
- Granularidade: 1 linha por empresa
- Origem: PGFN + Receita Federal (CNPJ)

## Variáveis

| Variável | Origem | Tipo | Descrição | Uso |
|--------|-------|------|-----------|-----|
| cnpj_raiz | CNPJ | String | Identificador da empresa (8 dígitos) | M1, M2 |
| qtd_estabelecimentos | CNPJ | Inteiro | Número total de estabelecimentos | M2 |
| qtd_ativos | CNPJ | Inteiro | Número de estabelecimentos ativos | M2 |
| qtd_inativos | CNPJ | Inteiro | Número de estabelecimentos inativos | M2 |
| idade_empresa_dias | CNPJ | Inteiro | Tempo de existência da empresa em dias | M1, M2 |
| valor_consolidado | PGFN | Numérico | Valor total da dívida inscrita | M1, M2 |
| indicador_ajuizado | PGFN | Binária | Indica se a dívida foi judicializada | M1, M2 |
| receita_principal | PGFN | Categórica | Tipo de receita da dívida | M1 |
| data_inscricao | PGFN | Data | Data de inscrição da dívida | M1 |
