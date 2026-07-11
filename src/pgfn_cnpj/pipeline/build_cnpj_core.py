"""Integra empresas e estabelecimentos dos Dados Abertos do CNPJ.

O relacionamento é realizado pela raiz do CNPJ:

- ``estabelecimentos_core_2025_12.parquet`` contém uma linha por
  estabelecimento;
- ``empresas_lookup_2025_12.parquet`` contém dados da empresa vinculados
  à raiz cadastral.

O lookup de empresas é deduplicado antes do join para impedir multiplicação
das linhas de estabelecimentos.

Entradas
--------
data/processed/cnpj/empresas_lookup_2025_12.parquet
data/processed/cnpj/estabelecimentos_core_2025_12.parquet

Saída principal
---------------
data/processed/cnpj/cnpj_core_2025_12.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


EMPRESAS_LOOKUP_PATH = (
    CNPJ_PROCESSED_DIR
    / "empresas_lookup_2025_12.parquet"
)

ESTABELECIMENTOS_CORE_PATH = (
    CNPJ_PROCESSED_DIR
    / "estabelecimentos_core_2025_12.parquet"
)

OUTPUT_PATH = (
    CNPJ_PROCESSED_DIR
    / "cnpj_core_2025_12.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "cnpj_core_2025_12_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_core_2025_12_schema.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "cnpj_core_2025_12_summary.txt"
)


EMPRESAS_REQUIRED_COLUMNS = {
    "cnpj_basico",
    "razao_social",
    "porte_empresa",
}

ESTABELECIMENTOS_REQUIRED_COLUMNS = {
    "cnpj",
    "cnpj_raiz",
    "situacao_cadastral",
    "data_situacao_cadastral",
    "cnae_fiscal_principal",
    "uf",
    "municipio",
    "data_inicio_atividade",
}


def escape_sql_path(path: Path) -> str:
    """Prepara um caminho para utilização em comandos SQL."""

    return path.as_posix().replace(
        "'",
        "''",
    )


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


def validate_parquet_schema(
    path: Path,
    required_columns: set[str],
    description: str,
) -> pq.ParquetFile:
    """Verifica a existência e o esquema de um arquivo Parquet."""

    if not path.exists():
        raise FileNotFoundError(
            f"{description} não encontrado:\n"
            f"{path}"
        )

    parquet_file = pq.ParquetFile(
        path
    )

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    missing_columns = sorted(
        required_columns
        - available_columns
    )

    if missing_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise ValueError(
            f"{description} não possui as variáveis necessárias:\n"
            f"{formatted_columns}"
        )

    return parquet_file


def validate_inputs() -> tuple[
    pq.ParquetFile,
    pq.ParquetFile,
]:
    """Valida os dois arquivos utilizados no relacionamento."""

    empresas_parquet = validate_parquet_schema(
        path=EMPRESAS_LOOKUP_PATH,
        required_columns=EMPRESAS_REQUIRED_COLUMNS,
        description="Lookup de empresas",
    )

    estabelecimentos_parquet = validate_parquet_schema(
        path=ESTABELECIMENTOS_CORE_PATH,
        required_columns=(
            ESTABELECIMENTOS_REQUIRED_COLUMNS
        ),
        description="Núcleo de estabelecimentos",
    )

    return (
        empresas_parquet,
        estabelecimentos_parquet,
    )


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

    duckdb_temp_directory = (
        TEMP_DIR
        / "duckdb"
    )

    duckdb_temp_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = escape_sql_path(
        duckdb_temp_directory
    )

    connection.execute(
        f"PRAGMA threads={threads}"
    )

    connection.execute(
        f"PRAGMA memory_limit='{memory_limit}'"
    )

    connection.execute(
        f"PRAGMA temp_directory='{temporary_path}'"
    )

    connection.execute(
        "PRAGMA enable_progress_bar=true"
    )


def register_sources(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Registra os arquivos Parquet como views do DuckDB."""

    empresas_path = escape_sql_path(
        EMPRESAS_LOOKUP_PATH
    )

    estabelecimentos_path = escape_sql_path(
        ESTABELECIMENTOS_CORE_PATH
    )

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW empresas_raw AS

        SELECT *
        FROM read_parquet('{empresas_path}')
        """
    )

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW estabelecimentos AS

        SELECT *
        FROM read_parquet('{estabelecimentos_path}')
        """
    )


def create_deduplicated_companies(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Mantém uma linha determinística por raiz do CNPJ."""

    connection.execute(
        """
        CREATE OR REPLACE TEMPORARY TABLE empresas AS

        SELECT
            cnpj_basico,

            MIN(razao_social)
                FILTER (
                    WHERE razao_social IS NOT NULL
                )
                AS razao_social,

            MIN(porte_empresa)
                FILTER (
                    WHERE porte_empresa IS NOT NULL
                )
                AS porte_empresa

        FROM empresas_raw

        WHERE
            cnpj_basico IS NOT NULL
            AND length(cnpj_basico) = 8

        GROUP BY cnpj_basico
        """
    )


def collect_source_statistics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, int]:
    """Calcula controles das fontes antes do relacionamento."""

    (
        empresas_rows,
        empresas_valid_rows,
        empresas_distinct_roots,
    ) = connection.execute(
        """
        SELECT
            COUNT(*)::BIGINT,

            COUNT(*) FILTER (
                WHERE
                    cnpj_basico IS NOT NULL
                    AND length(cnpj_basico) = 8
            )::BIGINT,

            COUNT(
                DISTINCT cnpj_basico
            ) FILTER (
                WHERE
                    cnpj_basico IS NOT NULL
                    AND length(cnpj_basico) = 8
            )::BIGINT

        FROM empresas_raw
        """
    ).fetchone()

    duplicate_excess_rows = connection.execute(
        """
        SELECT
            COALESCE(
                SUM(quantidade - 1),
                0
            )::BIGINT

        FROM (
            SELECT
                cnpj_basico,
                COUNT(*)::BIGINT
                    AS quantidade

            FROM empresas_raw

            WHERE
                cnpj_basico IS NOT NULL
                AND length(cnpj_basico) = 8

            GROUP BY cnpj_basico

            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    establishments_rows = connection.execute(
        """
        SELECT COUNT(*)::BIGINT
        FROM estabelecimentos
        """
    ).fetchone()[0]

    return {
        "empresas_rows": int(
            empresas_rows or 0
        ),
        "empresas_valid_rows": int(
            empresas_valid_rows or 0
        ),
        "empresas_distinct_roots": int(
            empresas_distinct_roots or 0
        ),
        "duplicate_excess_rows": int(
            duplicate_excess_rows or 0
        ),
        "establishments_rows": int(
            establishments_rows or 0
        ),
    }


def create_core_view(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Cria a relação final sem gravá-la imediatamente."""

    connection.execute(
        """
        CREATE OR REPLACE TEMPORARY VIEW cnpj_core AS

        SELECT
            establishments.cnpj,
            establishments.cnpj_raiz,

            companies.razao_social,
            companies.porte_empresa,

            establishments.situacao_cadastral,
            establishments.data_situacao_cadastral,
            establishments.cnae_fiscal_principal,
            establishments.uf,
            establishments.municipio,
            establishments.data_inicio_atividade,

            CASE
                WHEN companies.cnpj_basico IS NOT NULL
                THEN 1
                ELSE 0
            END::INTEGER
                AS indicador_match_empresa

        FROM estabelecimentos AS establishments

        LEFT JOIN empresas AS companies
            ON
                establishments.cnpj_raiz
                = companies.cnpj_basico
        """
    )


def collect_join_statistics(
    connection: duckdb.DuckDBPyConnection,
) -> dict[str, int | float]:
    """Calcula controles de correspondência e cardinalidade."""

    (
        output_rows,
        matched_rows,
        unmatched_rows,
    ) = connection.execute(
        """
        SELECT
            COUNT(*)::BIGINT,

            SUM(
                indicador_match_empresa
            )::BIGINT,

            SUM(
                CASE
                    WHEN indicador_match_empresa = 0
                    THEN 1
                    ELSE 0
                END
            )::BIGINT

        FROM cnpj_core
        """
    ).fetchone()

    output_rows = int(
        output_rows or 0
    )

    matched_rows = int(
        matched_rows or 0
    )

    unmatched_rows = int(
        unmatched_rows or 0
    )

    match_rate = (
        matched_rows / output_rows
        if output_rows
        else 0.0
    )

    return {
        "output_rows": output_rows,
        "matched_rows": matched_rows,
        "unmatched_rows": unmatched_rows,
        "match_rate": match_rate,
    }


def export_core(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Grava o núcleo cadastral de maneira atômica."""

    temporary_path = OUTPUT_PATH.with_suffix(
        ".temporary.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    temporary_sql_path = escape_sql_path(
        temporary_path
    )

    try:
        connection.execute(
            f"""
            COPY (
                SELECT
                    cnpj,
                    cnpj_raiz,
                    razao_social,
                    porte_empresa,
                    situacao_cadastral,
                    data_situacao_cadastral,
                    cnae_fiscal_principal,
                    uf,
                    municipio,
                    data_inicio_atividade

                FROM cnpj_core
            )

            TO '{temporary_sql_path}'

            (
                FORMAT PARQUET,
                COMPRESSION 'snappy'
            )
            """
        )

        if (
            not temporary_path.exists()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "O Parquet temporário não foi gerado "
                "ou possui tamanho zero."
            )

        temporary_path.replace(
            OUTPUT_PATH
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise


def create_sample_report(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Salva uma pequena amostra do núcleo integrado."""

    output_path = escape_sql_path(
        OUTPUT_PATH
    )

    sample = connection.execute(
        f"""
        SELECT *
        FROM read_parquet('{output_path}')

        ORDER BY cnpj

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
    source_statistics: dict[str, int],
    join_statistics: dict[str, Any],
    output_parquet: pq.ParquetFile,
) -> None:
    """Registra os controles do relacionamento cadastral."""

    row_count_consistency = (
        source_statistics[
            "establishments_rows"
        ]
        == join_statistics[
            "output_rows"
        ]
    )

    content = (
        "Núcleo cadastral integrado do CNPJ — snapshot 2025-12\n"
        "=====================================================\n\n"
        f"Lookup de empresas: {EMPRESAS_LOOKUP_PATH}\n"
        "Núcleo de estabelecimentos: "
        f"{ESTABELECIMENTOS_CORE_PATH}\n"
        f"Saída: {OUTPUT_PATH}\n\n"
        "Lookup de empresas\n"
        "------------------\n"
        "Linhas recebidas: "
        f"{source_statistics['empresas_rows']:,}\n"
        "Linhas com raiz válida: "
        f"{source_statistics['empresas_valid_rows']:,}\n"
        "Raízes distintas: "
        f"{source_statistics['empresas_distinct_roots']:,}\n"
        "Linhas excedentes por duplicidade: "
        f"{source_statistics['duplicate_excess_rows']:,}\n\n"
        "Relacionamento\n"
        "--------------\n"
        "Estabelecimentos de entrada: "
        f"{source_statistics['establishments_rows']:,}\n"
        "Linhas na saída: "
        f"{join_statistics['output_rows']:,}\n"
        "Linhas com correspondência: "
        f"{join_statistics['matched_rows']:,}\n"
        "Linhas sem correspondência: "
        f"{join_statistics['unmatched_rows']:,}\n"
        "Taxa de correspondência: "
        f"{join_statistics['match_rate']:.2%}\n"
        "Cardinalidade preservada: "
        f"{row_count_consistency}\n\n"
        "Metadados do Parquet\n"
        "--------------------\n"
        f"Linhas: {output_parquet.metadata.num_rows:,}\n"
        f"Colunas: {output_parquet.metadata.num_columns:,}\n"
        f"Row groups: {output_parquet.num_row_groups:,}\n\n"
        "Critério de deduplicação\n"
        "------------------------\n"
        "Foi mantida uma linha por cnpj_basico. Quando existiam "
        "valores distintos, foi selecionado o menor valor textual "
        "não nulo para razão social e porte, garantindo resultado "
        "determinístico.\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_pipeline(
    threads: int = 8,
    memory_limit: str = "4GB",
    force: bool = False,
) -> None:
    """Executa a integração cadastral completa."""

    validate_parameters(
        threads=threads,
        memory_limit=memory_limit,
    )

    ensure_project_directories()

    validate_inputs()

    validate_output_files(
        force=force
    )

    print(
        f"[INFO] Empresas: {EMPRESAS_LOOKUP_PATH}"
    )

    print(
        "[INFO] Estabelecimentos: "
        f"{ESTABELECIMENTOS_CORE_PATH}"
    )

    print(
        f"[INFO] Saída: {OUTPUT_PATH}"
    )

    with duckdb.connect(
        database=":memory:"
    ) as connection:
        configure_connection(
            connection=connection,
            threads=threads,
            memory_limit=memory_limit,
        )

        register_sources(
            connection
        )

        create_deduplicated_companies(
            connection
        )

        source_statistics = (
            collect_source_statistics(
                connection
            )
        )

        create_core_view(
            connection
        )

        join_statistics = (
            collect_join_statistics(
                connection
            )
        )

        export_core(
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
            source_statistics=source_statistics,
            join_statistics=join_statistics,
            output_parquet=output_parquet,
        )

    print(
        "[OK] Núcleo cadastral criado: "
        f"{output_parquet.metadata.num_rows:,} linhas."
    )

    print(
        "[OK] Taxa de correspondência: "
        f"{join_statistics['match_rate']:.2%}"
    )

    print(
        f"[OK] Saída: {OUTPUT_PATH}"
    )

    print(
        f"[OK] Resumo: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Integra empresas e estabelecimentos do CNPJ."
        )
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help=(
            "Quantidade de threads utilizadas pelo DuckDB."
        ),
    )

    parser.add_argument(
        "--memory-limit",
        type=str,
        default="4GB",
        help="Limite de memória utilizado pelo DuckDB.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Substitui os resultados existentes.",
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