"""Constrói o dataset agregado de descontinuidade empresarial.

A unidade de análise é a raiz do CNPJ, com uma linha por empresa.

O alvo ``y_descontinuidade`` é uma proxy observável no snapshot cadastral:

- y = 1 quando a empresa não possui estabelecimentos ativos;
- y = 0 quando existe pelo menos um estabelecimento ativo.

Esse alvo representa a situação observada no snapshot cadastral e não deve
ser interpretado, isoladamente, como previsão de encerramento futuro.

Entrada
-------
data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet

Saída
-----
data/processed/model/business_discontinuity_dataset.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import duckdb

from pgfn_cnpj.settings import (
    ABT_PROCESSED_DIR,
    MODEL_DATA_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    TEMP_DIR,
    ensure_project_directories,
)


TARGET_COLUMN = "y_descontinuidade"

INPUT_PATH = (
    ABT_PROCESSED_DIR
    / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"
)

OUTPUT_PATH = (
    MODEL_DATA_DIR
    / "business_discontinuity_dataset.parquet"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "business_discontinuity_dataset_summary.txt"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "business_discontinuity_dataset_sample.csv"
)


def escape_sql_path(path: Path) -> str:
    """Prepara um caminho para utilização em comandos SQL."""

    return path.as_posix().replace("'", "''")


def validate_parameters(
    sample_hash_threshold: int,
    threads: int,
) -> None:
    """Valida os parâmetros fornecidos na execução."""

    if not 0 <= sample_hash_threshold <= 999:
        raise ValueError(
            "sample_hash_threshold deve estar entre 0 e 999."
        )

    if threads < 1:
        raise ValueError(
            "threads deve ser maior ou igual a 1."
        )


def validate_input_file() -> None:
    """Verifica se a ABT integrada está disponível."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "A ABT integrada não foi encontrada:\n"
            f"{INPUT_PATH}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.pipeline.join_pgfn_cnpj"
        )


def validate_output_files(force: bool) -> None:
    """Evita a substituição acidental dos resultados oficiais."""

    output_files = (
        OUTPUT_PATH,
        SUMMARY_PATH,
        SAMPLE_PATH,
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

    temporary_directory = escape_sql_path(TEMP_DIR)

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


def build_source_relation(
    sample_hash_threshold: int,
) -> tuple[str, str]:
    """Define o uso da base completa ou de amostra determinística."""

    input_path = escape_sql_path(INPUT_PATH)

    if sample_hash_threshold == 0:
        return (
            f"read_parquet('{input_path}')",
            "base completa",
        )

    relation = f"""
        (
            SELECT *
            FROM read_parquet('{input_path}')
            WHERE
                hash(cnpj_raiz) % 1000
                < {sample_hash_threshold}
        )
    """

    description = (
        "amostra determinística: "
        "hash(cnpj_raiz) % 1000 "
        f"< {sample_hash_threshold}"
    )

    return relation, description


def create_dataset_table(
    connection: duckdb.DuckDBPyConnection,
    source_relation: str,
) -> None:
    """Agrega dados da PGFN e do CNPJ por raiz empresarial."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE business_discontinuity_dataset AS

        WITH source_data AS (
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

            FROM {source_relation}
        ),

        valid_data AS (
            SELECT *
            FROM source_data

            WHERE
                cnpj_raiz IS NOT NULL
                AND qtd_ativos IS NOT NULL
                AND qtd_estabelecimentos IS NOT NULL
        ),

        aggregated AS (
            SELECT
                cnpj_raiz,

                CASE
                    WHEN MAX(qtd_ativos) = 0 THEN 1
                    ELSE 0
                END::INTEGER
                    AS {TARGET_COLUMN},

                MAX(qtd_estabelecimentos)::INTEGER
                    AS qtd_estabelecimentos,

                MAX(qtd_ativos)::INTEGER
                    AS qtd_ativos,

                MAX(qtd_inativos)::INTEGER
                    AS qtd_inativos,

                MAX(idade_empresa_dias)::BIGINT
                    AS idade_empresa_dias,

                MIN(dt_inicio_min)
                    AS dt_inicio_min,

                MAX(dt_inicio_max)
                    AS dt_inicio_max,

                MAX(dt_situacao_max)
                    AS dt_situacao_max,

                COUNT(*)::BIGINT
                    AS pgfn_qtd_registros,

                COUNT(
                    DISTINCT numero_inscricao
                )::BIGINT
                    AS pgfn_qtd_inscricoes,

                SUM(
                    COALESCE(valor_consolidado, 0)
                )::DOUBLE
                    AS pgfn_divida_total,

                AVG(
                    COALESCE(valor_consolidado, 0)
                )::DOUBLE
                    AS pgfn_divida_media,

                MAX(
                    COALESCE(valor_consolidado, 0)
                )::DOUBLE
                    AS pgfn_divida_max,

                SUM(
                    CASE
                        WHEN indicador_ajuizado = 1 THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_ajuizadas,

                (
                    SUM(
                        CASE
                            WHEN indicador_ajuizado = 1 THEN 1
                            ELSE 0
                        END
                    ) * 1.0
                )
                / NULLIF(COUNT(*), 0)
                    AS pgfn_pct_ajuizadas,

                COUNT(
                    DISTINCT receita_principal
                )::INTEGER
                    AS pgfn_qtd_receitas_distintas,

                COUNT(
                    DISTINCT situacao_inscricao
                )::INTEGER
                    AS pgfn_qtd_situacoes_distintas,

                MIN(data_inscricao)
                    AS dt_inscricao_min,

                MAX(data_inscricao)
                    AS dt_inscricao_max,

                SUM(
                    CASE
                        WHEN tipo_situacao_inscricao = 'irregular'
                        THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_irregular,

                SUM(
                    CASE
                        WHEN tipo_situacao_inscricao = 'beneficio_fiscal'
                        THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_beneficio_fiscal,

                SUM(
                    CASE
                        WHEN tipo_situacao_inscricao = 'negociacao'
                        THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_negociacao,

                SUM(
                    CASE
                        WHEN tipo_situacao_inscricao = 'suspenso_judicial'
                        THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_suspenso_judicial,

                SUM(
                    CASE
                        WHEN tipo_situacao_inscricao = 'garantia'
                        THEN 1
                        ELSE 0
                    END
                )::BIGINT
                    AS pgfn_qtd_garantia

            FROM valid_data

            GROUP BY cnpj_raiz
        )

        SELECT
            *,

            CASE
                WHEN qtd_estabelecimentos > 0
                THEN qtd_ativos * 1.0 / qtd_estabelecimentos
                ELSE NULL
            END
                AS pct_ativos,

            CASE
                WHEN pgfn_qtd_registros > 0
                THEN pgfn_qtd_irregular * 1.0
                     / pgfn_qtd_registros
                ELSE NULL
            END
                AS pgfn_pct_irregular,

            CASE
                WHEN pgfn_qtd_registros > 0
                THEN pgfn_qtd_beneficio_fiscal * 1.0
                     / pgfn_qtd_registros
                ELSE NULL
            END
                AS pgfn_pct_beneficio_fiscal,

            CASE
                WHEN pgfn_qtd_registros > 0
                THEN pgfn_qtd_negociacao * 1.0
                     / pgfn_qtd_registros
                ELSE NULL
            END
                AS pgfn_pct_negociacao,

            CASE
                WHEN pgfn_qtd_registros > 0
                THEN pgfn_qtd_suspenso_judicial * 1.0
                     / pgfn_qtd_registros
                ELSE NULL
            END
                AS pgfn_pct_suspenso_judicial,

            CASE
                WHEN pgfn_qtd_registros > 0
                THEN pgfn_qtd_garantia * 1.0
                     / pgfn_qtd_registros
                ELSE NULL
            END
                AS pgfn_pct_garantia,

            LN(
                1
                + GREATEST(
                    COALESCE(qtd_estabelecimentos, 0),
                    0
                )
            )
                AS log1p_qtd_estabelecimentos,

            COALESCE(
                idade_empresa_dias,
                0
            ) / 365.25
                AS idade_empresa_anos,

            LN(
                1
                + GREATEST(
                    COALESCE(pgfn_divida_total, 0),
                    0
                )
            )
                AS log1p_pgfn_divida_total

        FROM aggregated
        """
    )


def export_dataset(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Grava o dataset final em formato Parquet."""

    output_path = escape_sql_path(OUTPUT_PATH)

    connection.execute(
        f"""
        COPY business_discontinuity_dataset

        TO '{output_path}'

        (
            FORMAT PARQUET,
            CODEC 'SNAPPY'
        )
        """
    )


def create_sample_report(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Salva uma pequena amostra adequada para o repositório."""

    sample = connection.execute(
        f"""
        SELECT *
        FROM business_discontinuity_dataset

        ORDER BY
            {TARGET_COLUMN} DESC,
            cnpj_raiz

        LIMIT 30
        """
    ).fetch_df()

    sample.to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary_report(
    connection: duckdb.DuckDBPyConnection,
    sample_description: str,
) -> None:
    """Registra as principais informações da construção do dataset."""

    (
        observation_count,
        positive_count,
        active_count,
    ) = connection.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT,
            SUM({TARGET_COLUMN})::BIGINT,
            SUM(
                CASE
                    WHEN qtd_ativos > 0 THEN 1
                    ELSE 0
                END
            )::BIGINT

        FROM business_discontinuity_dataset
        """
    ).fetchone()

    observation_count = int(
        observation_count or 0
    )

    positive_count = int(
        positive_count or 0
    )

    active_count = int(
        active_count or 0
    )

    positive_rate = (
        positive_count / observation_count
        if observation_count
        else 0.0
    )

    generated_at = datetime.now().isoformat(
        timespec="seconds"
    )

    content = (
        "Dataset de descontinuidade empresarial\n"
        "=======================================\n\n"
        f"Gerado em: {generated_at}\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Recorte: {sample_description}\n\n"
        "Distribuição do alvo\n"
        "--------------------\n"
        f"Empresas: {observation_count:,}\n"
        f"Empresas sem estabelecimento ativo: {positive_count:,}\n"
        f"Empresas com estabelecimento ativo: {active_count:,}\n"
        f"Taxa positiva: {positive_rate:.2%}\n\n"
        "Definição do alvo\n"
        "-----------------\n"
        "- y_descontinuidade = 1 quando qtd_ativos = 0;\n"
        "- y_descontinuidade = 0 quando qtd_ativos > 0.\n\n"
        "Observação metodológica\n"
        "-----------------------\n"
        "O alvo é uma proxy observada no snapshot cadastral. "
        "Ele não representa, isoladamente, uma previsão de "
        "encerramento futuro.\n\n"
        "Variáveis reservadas para auditoria\n"
        "------------------------------------\n"
        "- qtd_ativos;\n"
        "- qtd_inativos;\n"
        "- pct_ativos.\n\n"
        "Essas variáveis não deverão ser utilizadas como "
        "preditoras no treinamento porque estão diretamente "
        "relacionadas à definição do alvo.\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def build_dataset(
    sample_hash_threshold: int = 0,
    threads: int = 4,
    memory_limit: str = "4GB",
    force: bool = False,
) -> None:
    """Executa a construção completa do dataset."""

    validate_parameters(
        sample_hash_threshold=sample_hash_threshold,
        threads=threads,
    )

    ensure_project_directories()
    validate_input_file()
    validate_output_files(force=force)

    source_relation, sample_description = (
        build_source_relation(
            sample_hash_threshold
        )
    )

    print(f"[INFO] Entrada: {INPUT_PATH}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(f"[INFO] Recorte: {sample_description}")

    with duckdb.connect(
        database=":memory:"
    ) as connection:
        configure_connection(
            connection=connection,
            threads=threads,
            memory_limit=memory_limit,
        )

        create_dataset_table(
            connection=connection,
            source_relation=source_relation,
        )

        export_dataset(connection)

        create_sample_report(connection)

        create_summary_report(
            connection=connection,
            sample_description=sample_description,
        )

        (
            observation_count,
            positive_count,
        ) = connection.execute(
            f"""
            SELECT
                COUNT(*)::BIGINT,
                SUM({TARGET_COLUMN})::BIGINT

            FROM business_discontinuity_dataset
            """
        ).fetchone()

    print(
        "[OK] Dataset criado: "
        f"{OUTPUT_PATH}"
    )

    print(
        "[OK] Empresas: "
        f"{int(observation_count or 0):,}"
    )

    print(
        "[OK] Casos positivos: "
        f"{int(positive_count or 0):,}"
    )

    print(
        "[OK] Resumo: "
        f"{SUMMARY_PATH}"
    )

    print(
        "[OK] Amostra: "
        f"{SAMPLE_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Constrói o dataset agregado de "
            "descontinuidade empresarial."
        )
    )

    parser.add_argument(
        "--sample-hash-threshold",
        type=int,
        default=0,
        help=(
            "0 usa a base completa. Valores entre 1 e 999 "
            "geram uma amostra determinística por hash."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Quantidade de threads utilizadas pelo DuckDB.",
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

    build_dataset(
        sample_hash_threshold=(
            arguments.sample_hash_threshold
        ),
        threads=arguments.threads,
        memory_limit=arguments.memory_limit,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()