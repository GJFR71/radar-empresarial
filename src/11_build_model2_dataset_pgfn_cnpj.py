# 11_build_model2_dataset_pgfn_cnpj.py
"""
Gera dataset agregado (1 linha por cnpj_raiz) para o Modelo 2 — descontinuidade.
Base: data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet

Alvo (proxy observável no snapshot CNPJ 2025-12):
- y_descontinuidade = 1 se qtd_ativos == 0
- y_descontinuidade = 0 se qtd_ativos > 0

Saída:
- data/processed/model/model2_dataset_YYYYMMDD.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "abt" / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
OUT_PATH = OUT_DIR / f"model2_dataset_{RUN_DATE}.parquet"


def main(sample_hash_threshold: int = 0) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Parquet não encontrado: {IN_PATH}")

    con = duckdb.connect(database=":memory:")
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=true;")

    where = ""
    if sample_hash_threshold and 0 < sample_hash_threshold < 100:
        # amostragem estável por raiz (0..99)
        where = f"WHERE (abs(hash(cnpj_raiz)) % 100) < {int(sample_hash_threshold)}"

    # Agregação por cnpj_raiz:
    # - alvo: baseado em qtd_ativos (do CNPJ ABT) -> max() por segurança
    # - features fiscais: contagens/valores/proporções por situação/ajuizado e diversidade de receitas
    # - features CNPJ: qtd_estabelecimentos/qtd_ativos/idade etc.
    sql = f"""
    COPY (
      WITH base AS (
        SELECT
          cnpj_raiz,
          numero_inscricao,
          tipo_situacao_inscricao,
          situacao_inscricao,
          receita_principal,
          indicador_ajuizado,
          valor_consolidado,
          data_inscricao,
          qtd_estabelecimentos,
          qtd_ativos,
          qtd_inativos,
          dt_inicio_min,
          dt_inicio_max,
          dt_situacao_max,
          idade_empresa_dias
        FROM read_parquet('{IN_PATH.as_posix()}')
        {where}
      ),

      -- IMPORTANTE: só considero empresas com dados CNPJ válidos para definir y
      base_ok AS (
        SELECT *
        FROM base
        WHERE
          cnpj_raiz IS NOT NULL
          AND qtd_ativos IS NOT NULL
          AND qtd_estabelecimentos IS NOT NULL
      ),

      agg AS (
        SELECT
          cnpj_raiz,

          -- ----------- ALVO (proxy) -----------
          CASE WHEN max(qtd_ativos) = 0 THEN 1 ELSE 0 END AS y_descontinuidade,

          -- ----------- CNPJ (agregado por raiz) -----------
          max(qtd_estabelecimentos) AS qtd_estabelecimentos,
          max(qtd_ativos) AS qtd_ativos,
          max(qtd_inativos) AS qtd_inativos,
          max(idade_empresa_dias) AS idade_empresa_dias,
          min(dt_inicio_min) AS dt_inicio_min,
          max(dt_inicio_max) AS dt_inicio_max,
          max(dt_situacao_max) AS dt_situacao_max,

          -- ----------- PGFN (agregado por raiz) -----------
          count(*) AS pgfn_linhas,
          count(distinct numero_inscricao) AS pgfn_qtd_inscricoes,
          sum(coalesce(valor_consolidado, 0)) AS pgfn_valor_sum,
          avg(coalesce(valor_consolidado, 0)) AS pgfn_valor_mean,
          max(coalesce(valor_consolidado, 0)) AS pgfn_valor_max,

          sum(CASE WHEN indicador_ajuizado = 1 THEN 1 ELSE 0 END) AS pgfn_qtd_ajuizadas,
          (sum(CASE WHEN indicador_ajuizado = 1 THEN 1 ELSE 0 END) * 1.0) / nullif(count(*), 0) AS pgfn_pct_ajuizadas,

          count(distinct receita_principal) AS pgfn_receitas_distintas,
          count(distinct situacao_inscricao) AS pgfn_situacoes_distintas,

          min(data_inscricao) AS dt_inscricao_min,
          max(data_inscricao) AS dt_inscricao_max,

          -- ----------- distribuição tipo_situacao_inscricao -----------
          sum(CASE WHEN tipo_situacao_inscricao = 'irregular' THEN 1 ELSE 0 END) AS ts_irregular,
          sum(CASE WHEN tipo_situacao_inscricao = 'beneficio_fiscal' THEN 1 ELSE 0 END) AS ts_beneficio_fiscal,
          sum(CASE WHEN tipo_situacao_inscricao = 'negociacao' THEN 1 ELSE 0 END) AS ts_negociacao,
          sum(CASE WHEN tipo_situacao_inscricao = 'suspenso_judicial' THEN 1 ELSE 0 END) AS ts_suspenso_judicial,
          sum(CASE WHEN tipo_situacao_inscricao = 'garantia' THEN 1 ELSE 0 END) AS ts_garantia,

          -- proporções (normaliza pelas linhas)
          (sum(CASE WHEN tipo_situacao_inscricao = 'irregular' THEN 1 ELSE 0 END) * 1.0) / nullif(count(*),0) AS pct_irregular,
          (sum(CASE WHEN tipo_situacao_inscricao = 'beneficio_fiscal' THEN 1 ELSE 0 END) * 1.0) / nullif(count(*),0) AS pct_beneficio_fiscal,
          (sum(CASE WHEN tipo_situacao_inscricao = 'negociacao' THEN 1 ELSE 0 END) * 1.0) / nullif(count(*),0) AS pct_negociacao,
          (sum(CASE WHEN tipo_situacao_inscricao = 'suspenso_judicial' THEN 1 ELSE 0 END) * 1.0) / nullif(count(*),0) AS pct_suspenso_judicial,
          (sum(CASE WHEN tipo_situacao_inscricao = 'garantia' THEN 1 ELSE 0 END) * 1.0) / nullif(count(*),0) AS pct_garantia

        FROM base_ok
        GROUP BY 1
      )

      SELECT
        *,

        -- features derivadas simples (explicáveis)
        CASE
          WHEN qtd_estabelecimentos > 0 THEN (qtd_ativos * 1.0) / qtd_estabelecimentos
          ELSE 0
        END AS pct_ativos,

        log(1 + coalesce(qtd_estabelecimentos, 0)) AS log_qtd_estabelecimentos,
        (coalesce(idade_empresa_dias, 0) * 1.0) / 365.25 AS idade_empresa_anos,
        log(1 + coalesce(pgfn_valor_sum, 0)) AS log_pgfn_valor_sum

      FROM agg
    ) TO '{OUT_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION 'SNAPPY');
    """

    con.execute(sql)

    # Contagem rápida pra log
    n = con.execute(f"SELECT count(*) FROM read_parquet('{OUT_PATH.as_posix()}')").fetchone()[0]
    pos = con.execute(f"SELECT sum(y_descontinuidade) FROM read_parquet('{OUT_PATH.as_posix()}')").fetchone()[0]

    con.close()

    print(f"[DONE] model2_dataset gerado: {OUT_PATH} | empresas: {n:,} | y=1: {int(pos):,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sample_hash_threshold",
        type=int,
        default=0,
        help="0=full; senão 1..99 (amostra estável por hash)"
    )
    args = ap.parse_args()
    main(sample_hash_threshold=args.sample_hash_threshold)

