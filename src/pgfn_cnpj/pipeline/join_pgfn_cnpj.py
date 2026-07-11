"""Integra as ABTs da PGFN e do CNPJ.

O relacionamento é realizado pela variável ``cnpj_raiz``, mantendo todos os
registros existentes na ABT da PGFN e adicionando informações cadastrais
agregadas da base do CNPJ.

Entradas
--------
- data/processed/pgfn/pgfn_abt_2024_2025.parquet
- data/processed/cnpj/cnpj_abt_2025_12.parquet

Saída principal
---------------
- data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    ABT_PROCESSED_DIR,
    CNPJ_PROCESSED_DIR,
    PGFN_PROCESSED_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


PGFN_ABT_PATH = PGFN_PROCESSED_DIR / "pgfn_abt_2024_2025.parquet"
CNPJ_ABT_PATH = CNPJ_PROCESSED_DIR / "cnpj_abt_2025_12.parquet"

OUTPUT_PATH = (
    ABT_PROCESSED_DIR
    / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"
)

RUN_DATE = datetime.now().strftime("%Y%m%d")

HEAD_REPORT_PATH = (
    REPORTS_SAMPLES_DIR
    / f"abt_full_head_{RUN_DATE}.csv"
)

SCHEMA_REPORT_PATH = (
    REPORTS_TABLES_DIR
    / f"abt_full_schema_{RUN_DATE}.csv"
)

SUMMARY_REPORT_PATH = (
    REPORTS_METRICS_DIR
    / f"abt_full_summary_{RUN_DATE}.txt"
)


def validate_input_files(*paths: Path) -> None:
    """Verifica se todos os arquivos necessários estão disponíveis."""

    missing_files = [path for path in paths if not path.exists()]

    if missing_files:
        missing_list = "\n".join(
            f"- {path}" for path in missing_files
        )

        raise FileNotFoundError(
            "Os seguintes arquivos de entrada não foram encontrados:\n"
            f"{missing_list}"
        )


def escape_sql_path(path: Path) -> str:
    """Prepara um caminho para utilização em comandos SQL do DuckDB."""

    return path.as_posix().replace("'", "''")


def build_integrated_abt(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Realiza o join e grava a ABT integrada em formato Parquet."""

    pgfn_path = escape_sql_path(PGFN_ABT_PATH)
    cnpj_path = escape_sql_path(CNPJ_ABT_PATH)
    output_path = escape_sql_path(OUTPUT_PATH)

    query = f"""
        COPY (
            SELECT
                pgfn.*,
                cnpj.qtd_estabelecimentos,
                cnpj.qtd_ativos,
                cnpj.qtd_inativos,
                cnpj.dt_inicio_min,
                cnpj.dt_inicio_max,
                cnpj.dt_situacao_max,
                cnpj.idade_empresa_dias
            FROM read_parquet('{pgfn_path}') AS pgfn
            LEFT JOIN read_parquet('{cnpj_path}') AS cnpj
                ON pgfn.cnpj_raiz = cnpj.cnpj_raiz
        )
        TO '{output_path}'
        (
            FORMAT PARQUET,
            CODEC 'SNAPPY'
        );
    """

    connection.execute(query)


def create_sample_report(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Salva uma pequena amostra da ABT sem carregar o arquivo completo."""

    output_path = escape_sql_path(OUTPUT_PATH)

    sample = connection.execute(
        f"""
        SELECT *
        FROM read_parquet('{output_path}')
        LIMIT 30
        """
    ).fetch_df()

    sample.to_csv(
        HEAD_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_schema_report(
    parquet_file: pq.ParquetFile,
) -> None:
    """Salva os nomes e tipos das variáveis da ABT integrada."""

    schema = parquet_file.schema_arrow

    schema_table = pd.DataFrame(
        {
            "variavel": schema.names,
            "tipo": [
                str(schema.field(index).type)
                for index in range(len(schema.names))
            ],
        }
    )

    schema_table.to_csv(
        SCHEMA_REPORT_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary_report(
    row_count: int,
    column_count: int,
) -> None:
    """Registra um resumo textual da execução do pipeline."""

    content = (
        "Etapa — Integração das ABTs PGFN e CNPJ\n"
        "========================================\n\n"
        f"ABT PGFN: {PGFN_ABT_PATH}\n"
        f"ABT CNPJ: {CNPJ_ABT_PATH}\n"
        f"ABT integrada: {OUTPUT_PATH}\n\n"
        f"Quantidade de linhas: {row_count:,}\n"
        f"Quantidade de colunas: {column_count}\n"
    )

    SUMMARY_REPORT_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Executa a integração das ABTs e gera relatórios de controle."""

    ensure_project_directories()

    validate_input_files(
        PGFN_ABT_PATH,
        CNPJ_ABT_PATH,
    )

    print(f"[INFO] ABT PGFN: {PGFN_ABT_PATH}")
    print(f"[INFO] ABT CNPJ: {CNPJ_ABT_PATH}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")

    with duckdb.connect(database=":memory:") as connection:
        connection.execute("PRAGMA threads=4")
        connection.execute("PRAGMA enable_progress_bar=true")

        build_integrated_abt(connection)

        parquet_file = pq.ParquetFile(OUTPUT_PATH)

        row_count = parquet_file.metadata.num_rows
        column_count = parquet_file.metadata.num_columns

        create_sample_report(connection)
        create_schema_report(parquet_file)
        create_summary_report(
            row_count=row_count,
            column_count=column_count,
        )

    print(
        "[OK] ABT integrada criada: "
        f"{row_count:,} linhas e {column_count} colunas."
    )

    print(
        "[OK] Relatórios gerados em "
        f"{REPORTS_SAMPLES_DIR}, "
        f"{REPORTS_TABLES_DIR} e "
        f"{REPORTS_METRICS_DIR}."
    )


if __name__ == "__main__":
    main()