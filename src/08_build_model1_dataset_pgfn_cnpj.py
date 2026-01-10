# 08_build_model1_dataset_pgfn_cnpj.py
"""
Modelo 1 — Saúde fiscal
Cria dataset agregado por cnpj_raiz (1 linha por empresa) a partir da ABT FULL (PGFN + CNPJ).

Entrada:
- data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet

Saídas:
- data/processed/model/model1_dataset_20260109.parquet (agregado + alvo)
- reports/model1_dataset_summary_YYYYMMDD.txt
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "processed" / "abt" / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "model"
REPORTS = PROJECT_ROOT / "reports"
RUN_DATE = datetime.now().strftime("%Y%m%d")


def main(sample_groups: int | None = None) -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {IN_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / f"model1_dataset_{RUN_DATE}.parquet"
    summary_path = REPORTS / f"model1_dataset_summary_{RUN_DATE}.txt"

    con = duckdb.connect(database=":memory:")

    # Para performance
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='6GB';")  # ajuste se precisar
    con.execute("PRAGMA temp_directory='.tmp';")

    # (Opcional) limitar row groups para testes rápidos
    # Se sample_groups for None, usa tudo.
    if sample_groups is None:
        pgfn_view = f"read_parquet('{IN_PATH.as_posix()}')"
        groups_info = "ALL"
    else:
        # DuckDB não filtra row_groups diretamente; workaround: amostragem por hash do cnpj_raiz
        # Mantém representatividade e evita reler tudo quando você só quer “testar o pipeline”.
        pgfn_view = f"""
        (SELECT * FROM read_parquet('{IN_PATH.as_posix()}')
         WHERE (hash(cnpj_raiz) % 1000) < {max(1, min(999, sample_groups))})
        """
        groups_info = f"hash%1000 < {sample_groups}"

    # 1) Agregar por empresa (cnpj_raiz)
    # Observação: CNPJ features já estão repetidas por linha; usamos MAX (são constantes por raiz).
    con.execute(
        f"""
        CREATE OR REPLACE TABLE agg AS
        SELECT
            cnpj_raiz,

            -- PGFN: volume e intensidade da dívida
            COUNT(*)::BIGINT                                           AS qtd_inscricoes,
            SUM(valor_consolidado)::DOUBLE                             AS divida_total,
            AVG(valor_consolidado)::DOUBLE                             AS divida_media,
            MAX(valor_consolidado)::DOUBLE                             AS divida_max,

            -- PGFN: comportamento / judicialização
            AVG(indicador_ajuizado)::DOUBLE                            AS pct_ajuizado,
            SUM(CASE WHEN indicador_ajuizado = 1 THEN 1 ELSE 0 END)::BIGINT AS qtd_ajuizado,

            -- PGFN: temporalidade (datas já estão como timestamp no parquet)
            MIN(data_inscricao)                                        AS dt_inscricao_min,
            MAX(data_inscricao)                                        AS dt_inscricao_max,

            -- PGFN: presença no tempo observado (ano,trimestre)
            COUNT(DISTINCT CAST(ano AS VARCHAR) || '-' || CAST(trimestre AS VARCHAR))::INT AS qtd_periodos,

            -- PGFN: tipo_situacao_inscricao (one-hot em contagens)
            SUM(CASE WHEN tipo_situacao_inscricao='irregular' THEN 1 ELSE 0 END)::BIGINT          AS qtd_irregular,
            SUM(CASE WHEN tipo_situacao_inscricao='beneficio_fiscal' THEN 1 ELSE 0 END)::BIGINT    AS qtd_beneficio_fiscal,
            SUM(CASE WHEN tipo_situacao_inscricao='negociacao' THEN 1 ELSE 0 END)::BIGINT          AS qtd_negociacao,
            SUM(CASE WHEN tipo_situacao_inscricao='suspenso_judicial' THEN 1 ELSE 0 END)::BIGINT   AS qtd_suspenso_judicial,
            SUM(CASE WHEN tipo_situacao_inscricao='garantia' THEN 1 ELSE 0 END)::BIGINT            AS qtd_garantia,

            -- CNPJ (constantes por raiz na ABT FULL)
            MAX(qtd_estabelecimentos)::INT                              AS qtd_estabelecimentos,
            MAX(qtd_ativos)::INT                                        AS qtd_ativos,
            MAX(qtd_inativos)::INT                                      AS qtd_inativos,
            MAX(idade_empresa_dias)::BIGINT                             AS idade_empresa_dias

        FROM {pgfn_view}
        GROUP BY cnpj_raiz;
        """
    )

    # 2) Features derivadas (mais explicáveis)
    con.execute(
        """
        CREATE OR REPLACE TABLE feat AS
        SELECT
            *,
            CASE WHEN qtd_estabelecimentos > 0 THEN divida_total / qtd_estabelecimentos ELSE NULL END AS divida_por_estab,
            CASE WHEN qtd_ativos > 0 THEN divida_total / qtd_ativos ELSE NULL END                     AS divida_por_ativo,
            CASE WHEN qtd_periodos > 0 THEN qtd_inscricoes * 1.0 / qtd_periodos ELSE NULL END         AS inscricoes_por_periodo,
            CASE WHEN qtd_inscricoes > 0 THEN qtd_ajuizado * 1.0 / qtd_inscricoes ELSE NULL END        AS pct_ajuizado_chk
        FROM agg;
        """
    )

    # 3) Alvo: risco fiscal alto = ajuizado OU dívida_total >= p90
    p90 = con.execute("SELECT quantile_cont(divida_total, 0.90) FROM feat;").fetchone()[0]

    con.execute(
        f"""
        CREATE OR REPLACE TABLE final AS
        SELECT
            *,
            CASE
                WHEN qtd_ajuizado > 0 THEN 1
                WHEN divida_total >= {p90} THEN 1
                ELSE 0
            END AS y_risco_fiscal
        FROM feat;
        """
    )

    # 4) Exportar
    con.execute(
        f"""
        COPY (SELECT * FROM final)
        TO '{out_path.as_posix()}'
        (FORMAT PARQUET, CODEC 'SNAPPY');
        """
    )

    n = con.execute("SELECT COUNT(*) FROM final;").fetchone()[0]
    pos = con.execute("SELECT SUM(y_risco_fiscal) FROM final;").fetchone()[0]
    con.close()

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Modelo 1 — Dataset agregado (por cnpj_raiz)\n")
        f.write(f"Input:  {IN_PATH}\n")
        f.write(f"Filtro: {groups_info}\n")
        f.write(f"Output: {out_path}\n")
        f.write(f"Empresas (linhas): {n:,}\n")
        f.write(f"Positivos y_risco_fiscal=1: {pos:,} ({(pos/n*100 if n else 0):.2f}%)\n")
        f.write(f"Regra: y=1 se (qtd_ajuizado>0) OU (divida_total >= p90)\n")
        f.write("Obs.: p90 calculado no universo processado.\n")

    print(f"[DONE] model1_dataset gerado: {out_path} | empresas: {n:,} | y=1: {pos:,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_hash_threshold", type=int, default=0,
                    help="0=usa tudo; 1..999 usa filtro: hash(cnpj_raiz)%1000 < N (só para teste rápido)")
    args = ap.parse_args()

    sample = None if args.sample_hash_threshold == 0 else args.sample_hash_threshold
    main(sample_groups=sample)
