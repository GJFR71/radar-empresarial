"""Constrói o dataset agregado para análise de risco fiscal.

A unidade de análise é a raiz do CNPJ, com uma linha por empresa.

O alvo atual é uma regra operacional contemporânea:

- presença de inscrição ajuizada; ou
- dívida total igual ou superior ao percentil 90.

Essa definição será tratada como um critério de priorização fiscal. O módulo
de treinamento deverá impedir que variáveis diretamente utilizadas nessa
regra sejam empregadas como preditores.

Entrada
-------
data/processed/abt/abt_full_pgfn_cnpj_2024_2025_2025_12.parquet

Saída
-----
data/processed/model/fiscal_risk_dataset.parquet
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


INPUT_PATH = (
    ABT_PROCESSED_DIR
    / "abt_full_pgfn_cnpj_2024_2025_2025_12.parquet"
)

OUTPUT_PATH = MODEL_DATA_DIR / "fiscal_risk_dataset.parquet"

RUN_DATE = datetime.now().strftime("%Y%m%d")

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / f"fiscal_risk_dataset_summary_{RUN_DATE}.txt"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / f"fiscal_risk_dataset_sample_{RUN_DATE}.csv"
)


def escape_sql_path(path: Path) -> str:
    """Prepara caminhos para utilização em comandos SQL do DuckDB."""

    return path.as_posix().replace("'", "''")


def validate_parameters(
    sample_hash_threshold: int,
    threads: int,
) -> None:
    """Valida os parâmetros recebidos pela linha de comando."""

    if not 0 <= sample_hash_threshold <= 999:
        raise ValueError(
            "sample_hash_threshold deve estar entre 0 e 999."
        )

    if threads < 1:
        raise ValueError("threads deve ser maior ou igual a 1.")


def validate_input_file() -> None:
    """Verifica se a ABT integrada está disponível."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "A ABT integrada não foi encontrada:\n"
            f"{INPUT_PATH}\n\n"
            "Execute primeiro o módulo:\n"
            "python -m pgfn_cnpj.pipeline.join_pgfn_cnpj"
        )


def configure_connection(
    connection: duckdb.DuckDBPyConnection,
    threads: int,
    memory_limit: str,
) -> None:
    """Configura recursos utilizados pelo DuckDB."""

    temp_directory = escape_sql_path(TEMP_DIR)

    connection.execute(f"PRAGMA threads={threads}")
    connection.execute(f"PRAGMA memory_limit='{memory_limit}'")
    connection.execute(
        f"PRAGMA temp_directory='{temp_directory}'"
    )
    connection.execute("PRAGMA enable_progress_bar=true")


def build_source_relation(
    sample_hash_threshold: int,
) -> tuple[str, str]:
    """Define a fonte completa ou uma amostra determinística por CNPJ."""

    input_path = escape_sql_path(INPUT_PATH)

    if sample_hash_threshold == 0:
        relation = f"read_parquet('{input_path}')"
        filter_description = "base completa"

        return relation, filter_description

    relation = f"""
        (
            SELECT *
            FROM read_parquet('{input_path}')
            WHERE
                hash(cnpj_raiz) % 1000
                < {sample_hash_threshold}
        )
    """

    filter_description = (
        "amostra determinística: "
        "hash(cnpj_raiz) % 1000 "
        f"< {sample_hash_threshold}"
    )

    return relation, filter_description


def create_aggregated_table(
    connection: duckdb.DuckDBPyConnection,
    source_relation: str,
) -> None:
    """Agrega as inscrições da PGFN por raiz do CNPJ."""

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE aggregated AS

        SELECT
            cnpj_raiz,

            COUNT(*)::BIGINT
                AS qtd_inscricoes,

            SUM(valor_consolidado)::DOUBLE
                AS divida_total,

            AVG(valor_consolidado)::DOUBLE
                AS divida_media,

            MAX(valor_consolidado)::DOUBLE
                AS divida_max,

            AVG(indicador_ajuizado)::DOUBLE
                AS pct_ajuizado,

            SUM(
                CASE
                    WHEN indicador_ajuizado = 1 THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_ajuizado,

            MIN(data_inscricao)
                AS dt_inscricao_min,

            MAX(data_inscricao)
                AS dt_inscricao_max,

            COUNT(
                DISTINCT
                CAST(ano AS VARCHAR)
                || '-'
                || CAST(trimestre AS VARCHAR)
            )::INTEGER
                AS qtd_periodos,

            SUM(
                CASE
                    WHEN tipo_situacao_inscricao = 'irregular'
                    THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_irregular,

            SUM(
                CASE
                    WHEN tipo_situacao_inscricao = 'beneficio_fiscal'
                    THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_beneficio_fiscal,

            SUM(
                CASE
                    WHEN tipo_situacao_inscricao = 'negociacao'
                    THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_negociacao,

            SUM(
                CASE
                    WHEN tipo_situacao_inscricao = 'suspenso_judicial'
                    THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_suspenso_judicial,

            SUM(
                CASE
                    WHEN tipo_situacao_inscricao = 'garantia'
                    THEN 1
                    ELSE 0
                END
            )::BIGINT
                AS qtd_garantia,

            MAX(qtd_estabelecimentos)::INTEGER
                AS qtd_estabelecimentos,

            MAX(qtd_ativos)::INTEGER
                AS qtd_ativos,

            MAX(qtd_inativos)::INTEGER
                AS qtd_inativos,

            MAX(idade_empresa_dias)::BIGINT
                AS idade_empresa_dias

        FROM {source_relation}

        WHERE cnpj_raiz IS NOT NULL

        GROUP BY cnpj_raiz
        """
    )


def create_feature_table(
    connection: duckdb.DuckDBPyConnection,
) -> None:
    """Cria indicadores derivados e interpretáveis."""

    connection.execute(
        """
        CREATE OR REPLACE TABLE features AS

        SELECT
            *,

            CASE
                WHEN qtd_estabelecimentos > 0
                THEN divida_total / qtd_estabelecimentos
                ELSE NULL
            END
                AS divida_por_estabelecimento,

            CASE
                WHEN qtd_ativos > 0
                THEN divida_total / qtd_ativos
                ELSE NULL
            END
                AS divida_por_estabelecimento_ativo,

            CASE
                WHEN qtd_periodos > 0
                THEN qtd_inscricoes * 1.0 / qtd_periodos
                ELSE NULL
            END
                AS inscricoes_por_periodo,

            CASE
                WHEN qtd_inscricoes > 0
                THEN qtd_ajuizado * 1.0 / qtd_inscricoes
                ELSE NULL
            END
                AS proporcao_ajuizada_calculada

        FROM aggregated
        """
    )


def calculate_debt_percentile(
    connection: duckdb.DuckDBPyConnection,
) -> float:
    """Calcula o percentil 90 da dívida total."""

    result = connection.execute(
        """
        SELECT quantile_cont(divida_total, 0.90)
        FROM features
        """
    ).fetchone()

    percentile_90 = result[0] if result else None

    if percentile_90 is None:
        raise RuntimeError(
            "Não foi possível calcular o percentil 90 da dívida."
        )

    return float(percentile_90)


def create_target_table(
    connection: duckdb.DuckDBPyConnection,
    debt_percentile_90: float,
) -> None:
    """Cria o alvo operacional de priorização fiscal."""

    percentile_sql = format(debt_percentile_90, ".17g")

    connection.execute(
        f"""
        CREATE OR REPLACE TABLE final_dataset AS

        SELECT
            *,

            CASE
                WHEN qtd_ajuizado > 0 THEN 1
                WHEN divida_total >= {percentile_sql} THEN 1
                ELSE 0
            END::INTEGER
                AS y_risco_fiscal

        FROM features
        """
    )


def export_dataset(
    connection: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    """Grava o dataset final em formato Parquet."""

    if OUTPUT_PATH.exists():
        if not force:
            raise FileExistsError(
                f"O arquivo de saída já existe: {OUTPUT_PATH}\n"
                "Use --force para substituí-lo."
            )

        OUTPUT_PATH.unlink()

    output_path = escape_sql_path(OUTPUT_PATH)

    connection.execute(
        f"""
        COPY final_dataset
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
    """Salva uma pequena amostra para inspeção no repositório."""

    sample = connection.execute(
        """
        SELECT *
        FROM final_dataset
        ORDER BY cnpj_raiz
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
    filter_description: str,
    debt_percentile_90: float,
) -> None:
    """Registra as principais informações da execução."""

    row_count, positive_count = connection.execute(
        """
        SELECT
            COUNT(*)::BIGINT,
            SUM(y_risco_fiscal)::BIGINT
        FROM final_dataset
        """
    ).fetchone()

    row_count = int(row_count or 0)
    positive_count = int(positive_count or 0)

    positive_rate = (
        positive_count / row_count
        if row_count
        else 0.0
    )

    content = (
        "Dataset de priorização fiscal\n"
        "==============================\n\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Recorte: {filter_description}\n\n"
        f"Empresas: {row_count:,}\n"
        f"Casos positivos: {positive_count:,}\n"
        f"Taxa positiva: {positive_rate:.2%}\n"
        f"Percentil 90 da dívida: {debt_percentile_90:,.2f}\n\n"
        "Regra atual do alvo:\n"
        "- y = 1 quando qtd_ajuizado > 0; ou\n"
        "- y = 1 quando divida_total >= percentil 90.\n\n"
        "Observação metodológica:\n"
        "O alvo representa uma regra operacional contemporânea. "
        "Variáveis que participam diretamente dessa definição não "
        "devem ser usadas como preditores no treinamento.\n"
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
    """Executa todas as etapas de construção do dataset."""

    validate_parameters(
        sample_hash_threshold=sample_hash_threshold,
        threads=threads,
    )

    ensure_project_directories()
    validate_input_file()

    source_relation, filter_description = (
        build_source_relation(sample_hash_threshold)
    )

    print(f"[INFO] Entrada: {INPUT_PATH}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(f"[INFO] Recorte: {filter_description}")

    with duckdb.connect(database=":memory:") as connection:
        configure_connection(
            connection=connection,
            threads=threads,
            memory_limit=memory_limit,
        )

        create_aggregated_table(
            connection=connection,
            source_relation=source_relation,
        )

        create_feature_table(connection)

        debt_percentile_90 = calculate_debt_percentile(
            connection
        )

        create_target_table(
            connection=connection,
            debt_percentile_90=debt_percentile_90,
        )

        export_dataset(
            connection=connection,
            force=force,
        )

        create_sample_report(connection)

        create_summary_report(
            connection=connection,
            filter_description=filter_description,
            debt_percentile_90=debt_percentile_90,
        )

    print(f"[OK] Dataset criado: {OUTPUT_PATH}")
    print(f"[OK] Resumo criado: {SUMMARY_PATH}")
    print(f"[OK] Amostra criada: {SAMPLE_PATH}")


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Constrói o dataset agregado de priorização fiscal."
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
        help="Substitui o arquivo de saída, caso já exista.",
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    build_dataset(
        sample_hash_threshold=arguments.sample_hash_threshold,
        threads=arguments.threads,
        memory_limit=arguments.memory_limit,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()