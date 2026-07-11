"""Executa a validação estrutural do núcleo cadastral do CNPJ.

A rotina utiliza os metadados do Parquet e uma amostra distribuída entre
os row groups, sem carregar o conjunto completo na memória.

Também são verificadas:

- presença das variáveis esperadas;
- formato da raiz do CNPJ;
- formato do CNPJ completo;
- consistência entre o CNPJ completo e sua raiz;
- domínios das principais variáveis cadastrais.

Entrada
-------
data/processed/cnpj/cnpj_core_2025_12.parquet

Saídas
------
reports/samples/cnpj_validation_sample.csv
reports/tables/cnpj_validation_schema.csv
reports/tables/cnpj_validation_nulls.csv
reports/tables/cnpj_validation_decisions.csv
reports/tables/cnpj_validation_domains.csv
reports/tables/cnpj_validation_keys.csv
reports/metrics/cnpj_validation_info.txt
reports/metrics/cnpj_validation_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    CNPJ_PROCESSED_DIR,
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
    CNPJ_PROCESSED_DIR
    / "cnpj_core_2025_12.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "cnpj_validation_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_validation_schema.csv"
)

NULLS_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_validation_nulls.csv"
)

DECISIONS_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_validation_decisions.csv"
)

DOMAINS_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_validation_domains.csv"
)

KEYS_PATH = (
    REPORTS_TABLES_DIR
    / "cnpj_validation_keys.csv"
)

INFO_PATH = (
    REPORTS_METRICS_DIR
    / "cnpj_validation_info.txt"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "cnpj_validation_summary.txt"
)


EXPECTED_COLUMNS = {
    "cnpj",
    "cnpj_raiz",
    "razao_social",
    "porte_empresa",
    "situacao_cadastral",
    "data_situacao_cadastral",
    "cnae_fiscal_principal",
    "uf",
    "municipio",
    "data_inicio_atividade",
}

ABT_REQUIRED_COLUMNS = {
    "cnpj_raiz",
    "situacao_cadastral",
    "data_inicio_atividade",
    "data_situacao_cadastral",
}

PROTECTED_COLUMNS = (
    EXPECTED_COLUMNS
    | ABT_REQUIRED_COLUMNS
)

DOMAIN_COLUMNS = [
    "porte_empresa",
    "situacao_cadastral",
    "uf",
    "cnae_fiscal_principal",
    "municipio",
]


def validate_parameters(
    sample_rows: int,
    sample_row_groups: int,
    saved_rows: int,
    key_checks: int,
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

    if key_checks < 1:
        raise ValueError(
            "key_checks deve ser maior ou igual a 1."
        )


def validate_input_file() -> pq.ParquetFile:
    """Verifica existência, volume e esquema do Parquet."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "O núcleo cadastral do CNPJ não foi encontrado:\n"
            f"{INPUT_PATH}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.pipeline.build_cnpj_core"
        )

    parquet_file = pq.ParquetFile(
        INPUT_PATH
    )

    if parquet_file.metadata.num_rows == 0:
        raise ValueError(
            "O núcleo cadastral do CNPJ não possui linhas."
        )

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    missing_columns = sorted(
        EXPECTED_COLUMNS
        - available_columns
    )

    if missing_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise ValueError(
            "O núcleo cadastral do CNPJ não possui "
            "todas as variáveis esperadas:\n"
            f"{formatted_columns}"
        )

    missing_abt_columns = sorted(
        ABT_REQUIRED_COLUMNS
        - available_columns
    )

    if missing_abt_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_abt_columns
        )

        raise ValueError(
            "O arquivo não é compatível com a construção "
            "da ABT cadastral. Variáveis ausentes:\n"
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
        KEYS_PATH,
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


def normalize_identifier(
    series: pd.Series,
) -> pd.Series:
    """Remove espaços e mantém o identificador como texto."""

    normalized = (
        series.astype("string")
        .str.strip()
    )

    return normalized.mask(
        normalized.eq("")
    )


def build_key_check_table(
    parquet_file: pq.ParquetFile,
    number_of_checks: int,
) -> pd.DataFrame:
    """Verifica as chaves em row groups distribuídos."""

    output_columns = [
        "row_group",
        "linhas",
        "cnpj_nulos",
        "cnpj_raiz_nulos",
        "cnpj_formato_valido",
        "cnpj_raiz_formato_valido",
        "cnpj_raiz_correspondente",
        "percentual_cnpj_valido",
        "percentual_raiz_valida",
        "percentual_raiz_correspondente",
    ]

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
                "cnpj",
                "cnpj_raiz",
            ],
        )

        cnpj = normalize_identifier(
            dataframe["cnpj"]
        )

        cnpj_root = normalize_identifier(
            dataframe["cnpj_raiz"]
        )

        valid_cnpj = (
            cnpj.str.fullmatch(
                r"\d{14}"
            )
            .fillna(False)
        )

        valid_root = (
            cnpj_root.str.fullmatch(
                r"\d{8}"
            )
            .fillna(False)
        )

        matching_root = (
            valid_cnpj
            & valid_root
            & cnpj.str.slice(
                0,
                8,
            ).eq(
                cnpj_root
            )
        ).fillna(False)

        total_rows = len(
            dataframe
        )

        valid_cnpj_count = int(
            valid_cnpj.sum()
        )

        valid_root_count = int(
            valid_root.sum()
        )

        matching_root_count = int(
            matching_root.sum()
        )

        cnpj_rate = (
            valid_cnpj_count
            / total_rows
            if total_rows
            else 0.0
        )

        root_rate = (
            valid_root_count
            / total_rows
            if total_rows
            else 0.0
        )

        matching_rate = (
            matching_root_count
            / total_rows
            if total_rows
            else 0.0
        )

        rows.append(
            {
                "row_group": row_group_index,
                "linhas": total_rows,
                "cnpj_nulos": int(
                    cnpj.isna().sum()
                ),
                "cnpj_raiz_nulos": int(
                    cnpj_root.isna().sum()
                ),
                "cnpj_formato_valido": (
                    valid_cnpj_count
                ),
                "cnpj_raiz_formato_valido": (
                    valid_root_count
                ),
                "cnpj_raiz_correspondente": (
                    matching_root_count
                ),
                "percentual_cnpj_valido": round(
                    cnpj_rate * 100,
                    2,
                ),
                "percentual_raiz_valida": round(
                    root_rate * 100,
                    2,
                ),
                "percentual_raiz_correspondente": round(
                    matching_rate * 100,
                    2,
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=output_columns,
    )


def summarize_key_checks(
    key_checks: pd.DataFrame,
) -> dict[str, int | float]:
    """Consolida os controles das chaves verificadas."""

    if key_checks.empty:
        return {
            "rows_checked": 0,
            "cnpj_nulls": 0,
            "root_nulls": 0,
            "valid_cnpj": 0,
            "valid_root": 0,
            "matching_root": 0,
            "valid_cnpj_rate": 0.0,
            "valid_root_rate": 0.0,
            "matching_root_rate": 0.0,
        }

    rows_checked = int(
        key_checks["linhas"].sum()
    )

    valid_cnpj = int(
        key_checks[
            "cnpj_formato_valido"
        ].sum()
    )

    valid_root = int(
        key_checks[
            "cnpj_raiz_formato_valido"
        ].sum()
    )

    matching_root = int(
        key_checks[
            "cnpj_raiz_correspondente"
        ].sum()
    )

    return {
        "rows_checked": rows_checked,
        "cnpj_nulls": int(
            key_checks[
                "cnpj_nulos"
            ].sum()
        ),
        "root_nulls": int(
            key_checks[
                "cnpj_raiz_nulos"
            ].sum()
        ),
        "valid_cnpj": valid_cnpj,
        "valid_root": valid_root,
        "matching_root": matching_root,
        "valid_cnpj_rate": (
            valid_cnpj / rows_checked
            if rows_checked
            else 0.0
        ),
        "valid_root_rate": (
            valid_root / rows_checked
            if rows_checked
            else 0.0
        ),
        "matching_root_rate": (
            matching_root / rows_checked
            if rows_checked
            else 0.0
        ),
    }


def create_summary_report(
    parquet_file: pq.ParquetFile,
    sample: pd.DataFrame,
    decisions: pd.DataFrame,
    key_checks: pd.DataFrame,
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

    key_summary = summarize_key_checks(
        key_checks
    )

    content = (
        "Validação estrutural do CNPJ — snapshot 2025-12\n"
        "===============================================\n\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Linhas no Parquet: {parquet_file.metadata.num_rows:,}\n"
        f"Colunas no Parquet: {parquet_file.metadata.num_columns:,}\n"
        f"Row groups: {parquet_file.num_row_groups:,}\n\n"
        "Compatibilidade com a ABT cadastral\n"
        "-----------------------------------\n"
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
        "Verificação das chaves\n"
        "----------------------\n"
        "Linhas verificadas: "
        f"{key_summary['rows_checked']:,}\n"
        "CNPJs nulos: "
        f"{key_summary['cnpj_nulls']:,}\n"
        "Raízes nulas: "
        f"{key_summary['root_nulls']:,}\n"
        "CNPJs com 14 dígitos: "
        f"{key_summary['valid_cnpj_rate']:.2%}\n"
        "Raízes com 8 dígitos: "
        f"{key_summary['valid_root_rate']:.2%}\n"
        "Raízes correspondentes aos 8 primeiros dígitos "
        "do CNPJ: "
        f"{key_summary['matching_root_rate']:.2%}\n\n"
        "Relatórios gerados\n"
        "------------------\n"
        f"- {SAMPLE_PATH}\n"
        f"- {SCHEMA_PATH}\n"
        f"- {NULLS_PATH}\n"
        f"- {DECISIONS_PATH}\n"
        f"- {DOMAINS_PATH}\n"
        f"- {KEYS_PATH}\n"
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
    key_checks: int = 10,
    force: bool = False,
) -> None:
    """Executa a validação estrutural completa."""

    validate_parameters(
        sample_rows=sample_rows,
        sample_row_groups=sample_row_groups,
        saved_rows=saved_rows,
        key_checks=key_checks,
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

    key_checks_table = build_key_check_table(
        parquet_file=parquet_file,
        number_of_checks=key_checks,
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

    key_checks_table.to_csv(
        KEYS_PATH,
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
        key_checks=key_checks_table,
        sample_rows=sample_rows,
        sample_row_groups=sample_row_groups,
    )

    key_summary: dict[str, Any] = (
        summarize_key_checks(
            key_checks_table
        )
    )

    print(
        "[OK] Validação estrutural concluída."
    )

    print(
        "[OK] CNPJs com formato válido: "
        f"{key_summary['valid_cnpj_rate']:.2%}"
    )

    print(
        "[OK] Raízes correspondentes: "
        f"{key_summary['matching_root_rate']:.2%}"
    )

    print(
        f"[OK] Resumo: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos da linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Valida a estrutura do núcleo cadastral do CNPJ."
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
        "--key-checks",
        type=int,
        default=10,
        help=(
            "Quantidade de row groups utilizados "
            "na verificação das chaves."
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
        key_checks=arguments.key_checks,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()