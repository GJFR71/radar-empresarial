"""Constrói a ABT cadastral do CNPJ para o snapshot de dezembro de 2025.

A unidade de análise é a raiz do CNPJ, com uma linha por empresa.

O processamento utiliza o DuckDB diretamente sobre o arquivo Parquet,
evitando carregar toda a base cadastral na memória.

Entrada
-------
data/processed/cnpj/cnpj_core_2025_12.parquet

Saída principal
---------------
data/processed/cnpj/cnpj_abt_2025_12.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    CNPJ_PROCESSED_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    TEMP_DIR,
    ensure_project_directories,
)


SNAPSHOT_DATE = "2025-12-31"

INPUT_PATH = (
    CNPJ_PROCESSED_DIR
    / "cnpj_core_2025_12.parquet"
)

OUTPUT_PATH = (
    CNPJ_PROCESSED_DIR
    / "cnpj_abt_2025_12.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "cnpj_abt_2025_12_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_abt_2025_12_schema.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "cnpj_abt_2025_12_summary.txt"
)


REQUIRED_COLUMNS = {
    "cnpj_raiz",
    "situacao_cadastral",
    "data_inicio_atividade",
    "data_situacao_cadastral",
}


def escape_sql_path(path: Path) -> str:
    """Prepara um caminho para utilização em comandos SQL."""

    return path.as_posix().replace("'", "''")


def validate_parameters(
    threads: int,
    memory_limit: str,
) -> None:
    """Valida os parâmetros de execução."""

    if threads < 1:
        raise ValueError(
            "threads deve ser maior ou igual a 1."
        )

    if not memory_limit.strip():
        raise ValueError(
            "memory_limit não pode ser vazio."
        )


def validate_input_file() -> pq.ParquetFile:
    """Verifica a existência e o esquema do Parquet de entrada."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "O arquivo cadastral consolidado não foi encontrado:\n"
            f"{INPUT_PATH}"
        )

    parquet_file = pq.ParquetFile(
        INPUT_PATH
    )

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    missing_columns = sorted(
        REQUIRED_COLUMNS
        - available_columns
    )

    if missing_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise ValueError(
            "O arquivo de entrada não possui as "
            "variáveis necessárias:\n"
            f"{formatted_columns}"
        )

    return parquet_file


def validate_output_files(
    force: bool,
) -> None:
    """Evita a substituição acidental dos resultados."""

    output_files = (
        OUTPUT_PATH,
        SAMPLE_PATH,
        SCHEMA_PATH,
        SUMMARY_PATH,
    )

    existing_files = [
        path
        for path in output_files
        if path.exists()
    ]

    if existing_files and not force:
        formatted_files = "\n".join(
            f"- {path}"
            for path in existing_files
        )

        raise FileExistsError(
            "Os seguintes arquivos já existem:\n"
            f"{formatted_files}\n\n"
            "Use --force para substituí-los."
        )

    if force:
        for path in existing_files:
            path.unlink()


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    threads: int,
    memory_limit: str,
) -> None:
    """Configura os recursos utilizados pelo DuckDB."""

    temporary_directory = escape_sql_path(
        TEMP_DIR
    )

    connection.execute(
        f"PRAGMA threads={threads}"
    )

    connection.execute(
        f"PRAGMA memory_limit='{memory_limit}'"
    )

    connection.execute(
        f"PRAGMA temp_directory='{temporary_directory}'"
    )

    connection.execute(
        "PRAGMA enable_progress_bar=true"
    )


def build_abt(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Agrega os estabelecimentos por raiz do CNPJ."""

    input_path = escape_sql_path(
        INPUT_PATH
    )

    output_path = escape_sql_path(
        OUTPUT_PATH
    )

    query = f"""
        COPY (
            WITH source_data AS (
                SELECT
                    regexp_replace(
                        trim(
                            CAST(cnpj_raiz AS VARCHAR)
                        ),
                        '[^0-9]',
                        '',
                        'g'
                    ) AS cnpj_digits,

                    lpad(
                        trim(
                            CAST(
                                situacao_cadastral
                                AS VARCHAR
                            )
                        ),
                        2,
                        '0'
                    ) AS situacao_cadastral,

                    try_cast(
                        data_inicio_atividade
                        AS DATE
                    ) AS data_inicio_atividade,

                    try_cast(
                        data_situacao_cadastral
                        AS DATE
                    ) AS data_situacao_cadastral

                FROM read_parquet('{input_path}')
            ),

            normalized AS (
                SELECT
                    lpad(
                        cnpj_digits,
                        8,
                        '0'
                    ) AS cnpj_raiz,

                    CASE
                        WHEN situacao_cadastral = '02'
                        THEN 1
                        ELSE 0
                    END::INTEGER
                        AS indicador_ativo,

                    data_inicio_atividade,
                    data_situacao_cadastral

                FROM source_data

                WHERE
                    length(cnpj_digits)
                    BETWEEN 1 AND 8
            ),

            aggregated AS (
                SELECT
                    cnpj_raiz,

                    COUNT(*)::BIGINT
                        AS qtd_estabelecimentos,

                    SUM(
                        indicador_ativo
                    )::BIGINT
                        AS qtd_ativos,

                    SUM(
                        1 - indicador_ativo
                    )::BIGINT
                        AS qtd_inativos,

                    MIN(
                        data_inicio_atividade
                    )
                        AS dt_inicio_min,

                    MAX(
                        data_inicio_atividade
                    )
                        AS dt_inicio_max,

                    MAX(
                        data_situacao_cadastral
                    )
                        AS dt_situacao_max

                FROM normalized

                GROUP BY cnpj_raiz
            )

            SELECT
                *,

                CASE
                    WHEN dt_inicio_min IS NULL
                    THEN NULL

                    ELSE greatest(
                        date_diff(
                            'day',
                            dt_inicio_min,
                            DATE '{SNAPSHOT_DATE}'
                        ),
                        0
                    )
                END::BIGINT
                    AS idade_empresa_dias

            FROM aggregated
        )

        TO '{output_path}'

        (
            FORMAT PARQUET,
            CODEC 'SNAPPY'
        )
    """

    connection.execute(
        query
    )


def create_sample_report(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Salva uma pequena amostra da ABT."""

    output_path = escape_sql_path(
        OUTPUT_PATH
    )

    sample = connection.execute(
        f"""
        SELECT *
        FROM read_parquet('{output_path}')

        ORDER BY cnpj_raiz

        LIMIT 30
        """
    ).fetch_df()

    sample.to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )


def create_schema_report(
    parquet_file: pq.ParquetFile,
) -> None:
    """Salva os nomes e tipos das variáveis."""

    schema = parquet_file.schema_arrow

    schema_table = pd.DataFrame(
        {
            "variavel": schema.names,
            "tipo": [
                str(
                    schema.field(index).type
                )
                for index in range(
                    len(schema.names)
                )
            ],
        }
    )

    schema_table.to_csv(
        SCHEMA_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary_report(
    connection: duckdb.DuckDBPyConnection,
    input_row_groups: int,
) -> None:
    """Registra os controles da ABT gerada."""

    output_path = escape_sql_path(
        OUTPUT_PATH
    )

    (
        company_count,
        establishment_count,
        active_count,
        inactive_count,
        minimum_age,
        maximum_age,
    ) = connection.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT,
            SUM(qtd_estabelecimentos)::BIGINT,
            SUM(qtd_ativos)::BIGINT,
            SUM(qtd_inativos)::BIGINT,
            MIN(idade_empresa_dias)::BIGINT,
            MAX(idade_empresa_dias)::BIGINT

        FROM read_parquet('{output_path}')
        """
    ).fetchone()

    establishment_count = int(
        establishment_count or 0
    )

    active_count = int(
        active_count or 0
    )

    inactive_count = int(
        inactive_count or 0
    )

    consistency_ok = (
        establishment_count
        == active_count + inactive_count
    )

    content = (
        "ABT cadastral do CNPJ — snapshot 2025-12\n"
        "========================================\n\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Data de referência: {SNAPSHOT_DATE}\n"
        f"Row groups de entrada: {input_row_groups:,}\n\n"
        "Dimensões e controles\n"
        "--------------------\n"
        f"Empresas: {int(company_count or 0):,}\n"
        f"Estabelecimentos: {establishment_count:,}\n"
        f"Estabelecimentos ativos: {active_count:,}\n"
        f"Estabelecimentos inativos: {inactive_count:,}\n"
        "Ativos + inativos = estabelecimentos: "
        f"{consistency_ok}\n"
        f"Idade mínima em dias: {minimum_age}\n"
        f"Idade máxima em dias: {maximum_age}\n\n"
        "Variáveis produzidas\n"
        "--------------------\n"
        "- cnpj_raiz\n"
        "- qtd_estabelecimentos\n"
        "- qtd_ativos\n"
        "- qtd_inativos\n"
        "- dt_inicio_min\n"
        "- dt_inicio_max\n"
        "- dt_situacao_max\n"
        "- idade_empresa_dias\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_pipeline(
    threads: int = 4,
    memory_limit: str = "4GB",
    force: bool = False,
) -> None:
    """Executa a construção completa da ABT cadastral."""

    validate_parameters(
        threads=threads,
        memory_limit=memory_limit,
    )

    ensure_project_directories()

    input_parquet = validate_input_file()

    validate_output_files(
        force=force
    )

    print(f"[INFO] Entrada: {INPUT_PATH}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(
        "[INFO] Data de referência: "
        f"{SNAPSHOT_DATE}"
    )

    with duckdb.connect(
        database=":memory:"
    ) as connection:
        configure_connection(
            connection=connection,
            threads=threads,
            memory_limit=memory_limit,
        )

        build_abt(
            connection
        )

        output_parquet = pq.ParquetFile(
            OUTPUT_PATH
        )

        create_sample_report(
            connection
        )

        create_schema_report(
            output_parquet
        )

        create_summary_report(
            connection=connection,
            input_row_groups=(
                input_parquet.num_row_groups
            ),
        )

        company_count = (
            output_parquet.metadata.num_rows
        )

    print(
        "[OK] ABT cadastral criada: "
        f"{company_count:,} empresas."
    )

    print(
        "[OK] Resumo: "
        f"{SUMMARY_PATH}"
    )

    print(
        "[OK] Amostra: "
        f"{SAMPLE_PATH}"
    )

    print(
        "[OK] Esquema: "
        f"{SCHEMA_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Constrói a ABT cadastral do CNPJ "
            "para dezembro de 2025."
        )
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help=(
            "Quantidade de threads utilizadas "
            "pelo DuckDB."
        ),
    )

    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help=(
            "Limite de memória utilizado "
            "pelo DuckDB."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Substitui os resultados existentes."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_pipeline(
        threads=arguments.threads,
        memory_limit=arguments.memory_limit,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()