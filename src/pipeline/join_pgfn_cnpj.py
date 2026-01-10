"""
07_join_abt_pgfn_cnpj.py

Join da ABT PGFN (2024-2025) com ABT CNPJ (snapshot 2025-12) por cnpj_raiz (8 dígitos).

Entradas:
- data/processed/pgfn/pgfn_abt_2024_2025.parquet
- data/processed/cnpj/cnpj_abt_2025_12.parquet

Saídas:
- data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet
- reports/abt_full_head_YYYYMMDD.csv
- reports/abt_full_schema_YYYYMMDD.csv
- reports/abt_full_summary_YYYYMMDD.txt
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PGFN_ABT = PROJECT_ROOT / "data" / "processed" / "pgfn" / "pgfn_abt_2024_2025.parquet"
CNPJ_ABT = PROJECT_ROOT / "data" / "processed" / "cnpj" / "cnpj_abt_2025_12.parquet"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "abt"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DATE = datetime.now().strftime("%Y%m%d")
OUT_PATH = OUT_DIR / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"

REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

HEAD_PATH = REPORTS / f"abt_full_head_{RUN_DATE}.csv"
SCHEMA_PATH = REPORTS / f"abt_full_schema_{RUN_DATE}.csv"
SUMMARY_PATH = REPORTS / f"abt_full_summary_{RUN_DATE}.txt"


def main() -> None:
    if not PGFN_ABT.exists():
        raise FileNotFoundError(f"Não encontrado: {PGFN_ABT}")
    if not CNPJ_ABT.exists():
        raise FileNotFoundError(f"Não encontrado: {CNPJ_ABT}")

    print(f"[INFO] PGFN: {PGFN_ABT}")
    print(f"[INFO] CNPJ: {CNPJ_ABT}")
    print(f"[INFO] OUT : {OUT_PATH}")

    con = duckdb.connect(database=":memory:")

    # Dica: threads (use o número de núcleos que fizer sentido)
    con.execute("PRAGMA threads=4;")
    # Evita que o DuckDB tente baixar plugins/HTTPFS etc.
    con.execute("PRAGMA enable_progress_bar=true;")

    # Join direto parquet->parquet
    # Mantemos tudo da PGFN e anexamos features do CNPJ
    con.execute(
        f"""
        COPY (
            SELECT
                p.*,
                c.qtd_estabelecimentos,
                c.qtd_ativos,
                c.qtd_inativos,
                c.dt_inicio_min,
                c.dt_inicio_max,
                c.dt_situacao_max,
                c.idade_empresa_dias
            FROM read_parquet('{PGFN_ABT.as_posix()}') p
            LEFT JOIN read_parquet('{CNPJ_ABT.as_posix()}') c
            ON p.cnpj_raiz = c.cnpj_raiz
        ) TO '{OUT_PATH.as_posix()}' (FORMAT PARQUET, CODEC 'SNAPPY');
        """
    )

    pf = pq.ParquetFile(OUT_PATH)
    nrows = pf.metadata.num_rows
    ncols = pf.metadata.num_columns
    print(f"[DONE] ABT FULL: {nrows:,} linhas | {ncols} colunas")

    # Reports leves
    head_df = pq.read_table(OUT_PATH).to_pandas().head(30)
    head_df.to_csv(HEAD_PATH, index=False, encoding="utf-8")

    sch = pf.schema_arrow
    pd.DataFrame(
        {"field": sch.names, "type": [str(sch.field(i).type) for i in range(len(sch.names))]}
    ).to_csv(SCHEMA_PATH, index=False, encoding="utf-8")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("Etapa 7 — Join ABT PGFN + ABT CNPJ\n")
        f.write(f"PGFN:   {PGFN_ABT}\n")
        f.write(f"CNPJ:   {CNPJ_ABT}\n")
        f.write(f"Output: {OUT_PATH}\n\n")
        f.write(f"Linhas: {nrows:,}\n")
        f.write(f"Colunas: {ncols}\n")

    print(f"[DONE] Reports: {HEAD_PATH.name}, {SCHEMA_PATH.name}, {SUMMARY_PATH.name}")


if __name__ == "__main__":
    main()
