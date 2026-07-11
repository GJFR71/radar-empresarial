"""Executa a validação estrutural da base consolidada da PGFN.

A rotina utiliza metadados e uma amostra distribuída entre os row groups,
evitando carregar o Parquet completo na memória.

Entrada
-------
data/processed/pgfn/pgfn_sida_2024_2025.parquet

Saídas
------
reports/samples/pgfn_validation_sample.csv
reports/tables/pgfn_validation_schema.csv
reports/tables/pgfn_validation_nulls.csv
reports/tables/pgfn_validation_decisions.csv
reports/tables/pgfn_validation_domains.csv
reports/tables/pgfn_validation_uf_check.csv
reports/metrics/pgfn_validation_info.txt
reports/metrics/pgfn_validation_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    PGFN_PROCESSED_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)
from pgfn_cnpj.validation.common import (
    build_decision_table,
    build_null_profile,
    build_schema_table,
    build_systematic_sample,
    dataframe_info_text,
    evenly_spaced_indices,
    read_row_group_dataframe,
)


INPUT_PATH = (
    PGFN_PROCESSED_DIR
    / "pgfn_sida_2024_2025.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "pgfn_validation_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_validation_schema.csv"
)

NULLS_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_validation_nulls.csv"
)

DECISIONS_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_validation_decisions.csv"
)

DOMAINS_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_validation_domains.csv"
)

UF_CHECK_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_validation_uf_check.csv"
)

INFO_PATH = (
    REPORTS_METRICS_DIR
    / "pgfn_validation_info.txt"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "pgfn_validation_summary.txt"
)


REQUIRED_COLUMNS = {
    "tipo_pessoa",
    "cpf_cnpj",
    "tipo_devedor",
    "numero_inscricao",
    "situacao_inscricao",
    "tipo_situacao_inscricao",
    "receita_principal",
    "data_inscricao",
    "indicador_ajuizado",
    "valor_consolidado",
    "unidade_responsavel",
    "ano",
    "trimestre",
}

PROTECTED_COLUMNS = (
    REQUIRED_COLUMNS
    | {
        "arquivo_origem",
        "uf",
        "uf_devedor",
        "nome_devedor",
    }
)

DOMAIN_COLUMNS = [
    "tipo_pessoa",
    "tipo_devedor",
    "situacao_inscricao",
    "tipo_situacao_inscricao",
    "indicador_ajuizado",
    "unidade_responsavel",
    "ano",
    "trimestre",
    "uf",
    "uf_devedor",
]


def validate_parameters(
    sample_rows: int,
    sample_row_groups: int,
    saved_rows: int,
    uf_checks: int,
) -> None:
    """Valida os parâmetros da execução."""

    if sample_rows < 1:
        raise ValueError(
            "sample_rows deve ser maior ou igual a 1."
        )

    if sample_row_groups < 1:
        raise ValueError(
            "sample_row_groups deve ser maior ou igual a 1."
        )

    if saved_rows < 1:
        raise ValueError(
            "saved_rows deve ser maior ou igual a 1."
        )

    if uf_checks < 1:
        raise ValueError(
            "uf_checks deve ser maior ou igual a 1."
        )


def validate_input_file() -> pq.ParquetFile:
    """Verifica existência, volume e esquema do Parquet."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "A base consolidada da PGFN não foi encontrada:\n"
            f"{INPUT_PATH}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.pipeline.transform_pgfn"
        )

    parquet_file = pq.ParquetFile(
        INPUT_PATH
    )

    if parquet_file.metadata.num_rows == 0:
        raise ValueError(
            "O arquivo consolidado da PGFN não possui linhas."
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
            "A base da PGFN não é compatível com a etapa "
            "de construção da ABT. Variáveis ausentes:\n"
            f"{formatted_columns}"
        )

    return parquet_file


def validate_output_files(
    force: bool,
) -> None:
    """Evita substituição acidental dos relatórios."""

    output_files = (
        SAMPLE_PATH,
        SCHEMA_PATH,
        NULLS_PATH,
        DECISIONS_PATH,
        DOMAINS_PATH,
        UF_CHECK_PATH,
        INFO_PATH,
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
            "Os seguintes relatórios já existem:\n"
            f"{formatted_files}\n\n"
            "Use --force para substituí-los."
        )

    if force:
        for path in existing_files:
            path.unlink()


def build_domains_table(
    sample: pd.DataFrame,
    limit_per_column: int = 30,
) -> pd.DataFrame:
    """Registra os valores mais frequentes das dimensões."""

    rows: list[dict[str, object]] = []

    available_columns = [
        column
        for column in DOMAIN_COLUMNS
        if column in sample.columns
    ]

    for column in available_columns:
        frequencies = (
            sample[column]
            .astype("string")
            .value_counts(
                dropna=False
            )
            .head(
                limit_per_column
            )
        )

        for value, count in frequencies.items():
            rows.append(
                {
                    "variavel": column,
                    "valor": str(value),
                    "frequencia": int(count),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "variavel",
            "valor",
            "frequencia",
        ],
    )


def select_uf_column(
    parquet_file: pq.ParquetFile,
) -> str | None:
    """Seleciona a melhor variável geográfica disponível."""

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    if "uf_devedor" in available_columns:
        return "uf_devedor"

    if "uf" in available_columns:
        return "uf"

    return None


def build_uf_check_table(
    parquet_file: pq.ParquetFile,
    number_of_checks: int,
) -> pd.DataFrame:
    """Verifica a variável de UF em row groups distribuídos."""

    uf_column = select_uf_column(
        parquet_file
    )

    output_columns = [
        "variavel",
        "row_group",
        "linhas",
        "nulos",
        "percentual_nulos",
        "valores_distintos",
    ]

    if uf_column is None:
        return pd.DataFrame(
            columns=output_columns
        )

    row_group_indices = evenly_spaced_indices(
        total=parquet_file.num_row_groups,
        count=number_of_checks,
    )

    rows: list[dict[str, object]] = []

    for row_group_index in row_group_indices:
        dataframe = read_row_group_dataframe(
            parquet_file=parquet_file,
            row_group_index=row_group_index,
            columns=[
                uf_column,
            ],
        )

        series = dataframe[
            uf_column
        ].astype("string")

        total_rows = len(series)

        null_rows = int(
            series.isna().sum()
        )

        null_rate = (
            null_rows / total_rows
            if total_rows
            else 0.0
        )

        rows.append(
            {
                "variavel": uf_column,
                "row_group": row_group_index,
                "linhas": total_rows,
                "nulos": null_rows,
                "percentual_nulos": round(
                    null_rate * 100,
                    2,
                ),
                "valores_distintos": int(
                    series.nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=output_columns,
    )


def monetary_summary(
    sample: pd.DataFrame,
) -> str:
    """Resume a variável valor_consolidado da amostra."""

    if "valor_consolidado" not in sample.columns:
        return (
            "A variável valor_consolidado não foi encontrada."
        )

    values = pd.to_numeric(
        sample["valor_consolidado"],
        errors="coerce",
    )

    valid_values = values.dropna()

    if valid_values.empty:
        return (
            "A variável valor_consolidado não possui "
            "valores numéricos válidos na amostra."
        )

    quantiles = valid_values.quantile(
        [
            0.25,
            0.50,
            0.75,
        ]
    )

    return (
        f"Valores válidos: {len(valid_values):,}\n"
        f"Valores ausentes: {int(values.isna().sum()):,}\n"
        f"Mínimo: {valid_values.min():,.2f}\n"
        f"Primeiro quartil: {quantiles.loc[0.25]:,.2f}\n"
        f"Mediana: {quantiles.loc[0.50]:,.2f}\n"
        f"Terceiro quartil: {quantiles.loc[0.75]:,.2f}\n"
        f"Máximo: {valid_values.max():,.2f}\n"
        f"Média: {valid_values.mean():,.2f}\n"
    )


def create_summary_report(
    parquet_file: pq.ParquetFile,
    sample: pd.DataFrame,
    decisions: pd.DataFrame,
    uf_check: pd.DataFrame,
    sample_rows: int,
    sample_row_groups: int,
) -> None:
    """Registra os principais resultados da validação."""

    review_count = int(
        decisions[
            "acao_sugerida"
        ].eq(
            "revisar"
        ).sum()
    )

    uf_column = select_uf_column(
        parquet_file
    )

    content = (
        "Validação estrutural da PGFN — período 2024–2025\n"
        "================================================\n\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Linhas no Parquet: {parquet_file.metadata.num_rows:,}\n"
        f"Colunas no Parquet: {parquet_file.metadata.num_columns:,}\n"
        f"Row groups: {parquet_file.num_row_groups:,}\n\n"
        "Compatibilidade com a ABT\n"
        "-------------------------\n"
        "Todas as variáveis obrigatórias estão presentes: True\n\n"
        "Amostra diagnóstica\n"
        "-------------------\n"
        f"Limite solicitado: {sample_rows:,}\n"
        f"Linhas obtidas: {len(sample):,}\n"
        f"Row groups solicitados: {sample_row_groups:,}\n"
        "Estratégia: seleção aproximadamente equidistante "
        "ao longo do Parquet.\n\n"
        "Recomendações\n"
        "-------------\n"
        f"Variáveis marcadas para revisão: {review_count:,}\n"
        "Nenhuma variável é removida automaticamente por esta etapa.\n\n"
        "Verificação geográfica\n"
        "----------------------\n"
        f"Variável utilizada: {uf_column or 'não disponível'}\n"
        f"Row groups verificados: {len(uf_check):,}\n\n"
        "Resumo de valor_consolidado na amostra\n"
        "--------------------------------------\n"
        f"{monetary_summary(sample)}\n"
        "Relatórios gerados\n"
        "------------------\n"
        f"- {SAMPLE_PATH}\n"
        f"- {SCHEMA_PATH}\n"
        f"- {NULLS_PATH}\n"
        f"- {DECISIONS_PATH}\n"
        f"- {DOMAINS_PATH}\n"
        f"- {UF_CHECK_PATH}\n"
        f"- {INFO_PATH}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_validation(
    sample_rows: int = 200_000,
    sample_row_groups: int = 10,
    saved_rows: int = 100,
    uf_checks: int = 10,
    force: bool = False,
) -> None:
    """Executa a validação estrutural completa."""

    validate_parameters(
        sample_rows=sample_rows,
        sample_row_groups=sample_row_groups,
        saved_rows=saved_rows,
        uf_checks=uf_checks,
    )

    ensure_project_directories()

    parquet_file = validate_input_file()

    validate_output_files(
        force=force
    )

    print(
        f"[INFO] Entrada: {INPUT_PATH}"
    )

    print(
        "[INFO] Linhas: "
        f"{parquet_file.metadata.num_rows:,}"
    )

    print(
        "[INFO] Colunas: "
        f"{parquet_file.metadata.num_columns:,}"
    )

    print(
        "[INFO] Row groups: "
        f"{parquet_file.num_row_groups:,}"
    )

    sample = build_systematic_sample(
        parquet_file=parquet_file,
        max_rows=sample_rows,
        max_row_groups=sample_row_groups,
    )

    print(
        "[INFO] Amostra: "
        f"{len(sample):,} linhas x "
        f"{len(sample.columns):,} colunas."
    )

    schema_table = build_schema_table(
        parquet_file
    )

    nulls_table = build_null_profile(
        sample
    )

    decisions_table = build_decision_table(
        sample=sample,
        protected_columns=PROTECTED_COLUMNS,
    )

    domains_table = build_domains_table(
        sample
    )

    uf_check_table = build_uf_check_table(
        parquet_file=parquet_file,
        number_of_checks=uf_checks,
    )

    sample.head(
        saved_rows
    ).to_csv(
        SAMPLE_PATH,
        index=False,
        encoding="utf-8",
    )

    schema_table.to_csv(
        SCHEMA_PATH,
        index=False,
        encoding="utf-8",
    )

    nulls_table.to_csv(
        NULLS_PATH,
        index=False,
        encoding="utf-8",
    )

    decisions_table.to_csv(
        DECISIONS_PATH,
        index=False,
        encoding="utf-8",
    )

    domains_table.to_csv(
        DOMAINS_PATH,
        index=False,
        encoding="utf-8",
    )

    uf_check_table.to_csv(
        UF_CHECK_PATH,
        index=False,
        encoding="utf-8",
    )

    INFO_PATH.write_text(
        dataframe_info_text(
            sample
        ),
        encoding="utf-8",
    )

    create_summary_report(
        parquet_file=parquet_file,
        sample=sample,
        decisions=decisions_table,
        uf_check=uf_check_table,
        sample_rows=sample_rows,
        sample_row_groups=sample_row_groups,
    )

    print(
        "[OK] Validação estrutural concluída."
    )

    print(
        f"[OK] Resumo: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos da linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Valida a estrutura da base consolidada da PGFN."
        )
    )

    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200_000,
        help=(
            "Quantidade máxima de linhas da amostra "
            "utilizada nos diagnósticos."
        ),
    )

    parser.add_argument(
        "--sample-row-groups",
        type=int,
        default=10,
        help=(
            "Quantidade máxima de row groups distribuídos "
            "ao longo do Parquet."
        ),
    )

    parser.add_argument(
        "--saved-rows",
        type=int,
        default=100,
        help=(
            "Quantidade de linhas da amostra gravadas "
            "no relatório CSV."
        ),
    )

    parser.add_argument(
        "--uf-checks",
        type=int,
        default=10,
        help=(
            "Quantidade de row groups utilizados "
            "na verificação geográfica."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Substitui os relatórios existentes.",
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_validation(
        sample_rows=arguments.sample_rows,
        sample_row_groups=arguments.sample_row_groups,
        saved_rows=arguments.saved_rows,
        uf_checks=arguments.uf_checks,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()