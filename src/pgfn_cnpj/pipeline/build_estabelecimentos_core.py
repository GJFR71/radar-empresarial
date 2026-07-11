"""Constrói o núcleo cadastral de estabelecimentos do CNPJ.

O processamento lê recursivamente os arquivos extraídos do conjunto
``Estabelecimentos`` da Receita Federal e mantém apenas as variáveis
necessárias para a integração cadastral e para os modelos do projeto.

Entrada
-------
data/staging/cnpj/2025-12/estabelecimentos/

Saída principal
---------------
data/processed/cnpj/estabelecimentos_core_2025_12.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    CNPJ_PROCESSED_DIR,
    CNPJ_STAGING_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


SNAPSHOT = "2025-12"

INPUT_DIR = (
    CNPJ_STAGING_DIR
    / SNAPSHOT
    / "estabelecimentos"
)

OUTPUT_PATH = (
    CNPJ_PROCESSED_DIR
    / "estabelecimentos_core_2025_12.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "estabelecimentos_core_2025_12_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "estabelecimentos_core_2025_12_schema.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "estabelecimentos_core_2025_12_summary.txt"
)


ESTABLISHMENT_COLUMNS = [
    "cnpj_basico",
    "cnpj_ordem",
    "cnpj_dv",
    "identificador_matriz_filial",
    "nome_fantasia",
    "situacao_cadastral",
    "data_situacao_cadastral",
    "motivo_situacao_cadastral",
    "nome_cidade_exterior",
    "pais",
    "data_inicio_atividade",
    "cnae_fiscal_principal",
    "cnae_fiscal_secundaria",
    "tipo_logradouro",
    "logradouro",
    "numero",
    "complemento",
    "bairro",
    "cep",
    "uf",
    "municipio",
    "ddd1",
    "telefone1",
    "ddd2",
    "telefone2",
    "ddd_fax",
    "fax",
    "correio_eletronico",
    "situacao_especial",
    "data_situacao_especial",
]

USE_COLUMNS = [
    0,   # cnpj_basico
    1,   # cnpj_ordem
    2,   # cnpj_dv
    5,   # situacao_cadastral
    6,   # data_situacao_cadastral
    10,  # data_inicio_atividade
    11,  # cnae_fiscal_principal
    19,  # uf
    20,  # municipio
]

OUTPUT_SCHEMA = pa.schema(
    [
        ("cnpj", pa.string()),
        ("cnpj_raiz", pa.string()),
        ("situacao_cadastral", pa.string()),
        ("data_situacao_cadastral", pa.timestamp("ns")),
        ("cnae_fiscal_principal", pa.string()),
        ("uf", pa.string()),
        ("municipio", pa.string()),
        ("data_inicio_atividade", pa.timestamp("ns")),
    ]
)


@dataclass
class ProcessingStatistics:
    """Acumula controles do processamento."""

    source_files: int = 0
    processed_chunks: int = 0
    input_rows: int = 0
    output_rows: int = 0
    invalid_cnpj_rows: int = 0
    invalid_start_date_rows: int = 0
    invalid_status_date_rows: int = 0
    empty_chunks: int = 0


def validate_parameters(
    chunk_size: int,
) -> None:
    """Valida os parâmetros de execução."""

    if chunk_size < 1:
        raise ValueError(
            "chunk_size deve ser maior ou igual a 1."
        )


def list_source_files(
    directory: Path,
) -> list[Path]:
    """Localiza recursivamente os arquivos extraídos."""

    if not directory.exists():
        raise FileNotFoundError(
            "O diretório de estabelecimentos não foi encontrado:\n"
            f"{directory}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.ingestion.extract_cnpj "
            "--groups estabelecimentos"
        )

    files = sorted(
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and not path.name.startswith(".")
            and path.stat().st_size > 0
        )
    )

    if not files:
        raise FileNotFoundError(
            "Nenhum arquivo de estabelecimentos foi encontrado em:\n"
            f"{directory}"
        )

    return files


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


def normalize_digits(
    series: pd.Series,
    width: int,
) -> pd.Series:
    """Mantém dígitos e completa zeros à esquerda."""

    digits = (
        series.astype("string")
        .str.strip()
        .str.replace(
            r"\D",
            "",
            regex=True,
        )
    )

    valid_mask = (
        digits.notna()
        & digits.str.len().between(
            1,
            width,
        )
    )

    normalized = digits.str.zfill(
        width
    )

    return normalized.where(
        valid_mask
    )


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """Remove espaços e converte textos vazios em valores ausentes."""

    normalized = (
        series.astype("string")
        .str.strip()
    )

    return normalized.mask(
        normalized.eq("")
    )


def normalize_status(
    series: pd.Series,
) -> pd.Series:
    """Padroniza o código da situação cadastral com dois dígitos."""

    return normalize_digits(
        series=series,
        width=2,
    )


def normalize_cnae(
    series: pd.Series,
) -> pd.Series:
    """Padroniza o CNAE principal com sete dígitos."""

    return normalize_digits(
        series=series,
        width=7,
    )


def normalize_state(
    series: pd.Series,
) -> pd.Series:
    """Padroniza a sigla da unidade federativa."""

    normalized = (
        normalize_text(series)
        .str.upper()
    )

    return normalized.where(
        normalized.str.len().eq(2)
    )


def parse_rfb_date(
    series: pd.Series,
) -> pd.Series:
    """Converte datas no formato AAAAMMDD."""

    normalized = (
        series.astype("string")
        .str.strip()
    )

    normalized = normalized.mask(
        normalized.isin(
            [
                "",
                "0",
                "00000000",
                "<NA>",
            ]
        )
    )

    return pd.to_datetime(
        normalized,
        format="%Y%m%d",
        errors="coerce",
    )


def process_chunk(
    chunk: pd.DataFrame,
    statistics: ProcessingStatistics,
) -> pd.DataFrame:
    """Limpa e seleciona as variáveis de um lote."""

    statistics.processed_chunks += 1
    statistics.input_rows += len(chunk)

    if chunk.empty:
        statistics.empty_chunks += 1

        return pd.DataFrame(
            columns=OUTPUT_SCHEMA.names
        )

    cnpj_root = normalize_digits(
        series=chunk["cnpj_basico"],
        width=8,
    )

    cnpj_order = normalize_digits(
        series=chunk["cnpj_ordem"],
        width=4,
    )

    cnpj_check_digits = normalize_digits(
        series=chunk["cnpj_dv"],
        width=2,
    )

    valid_cnpj_mask = (
        cnpj_root.notna()
        & cnpj_order.notna()
        & cnpj_check_digits.notna()
    )

    statistics.invalid_cnpj_rows += int(
        (~valid_cnpj_mask).sum()
    )

    start_date_raw = (
        chunk["data_inicio_atividade"]
        .astype("string")
        .str.strip()
    )

    status_date_raw = (
        chunk["data_situacao_cadastral"]
        .astype("string")
        .str.strip()
    )

    start_date = parse_rfb_date(
        chunk["data_inicio_atividade"]
    )

    status_date = parse_rfb_date(
        chunk["data_situacao_cadastral"]
    )

    start_date_informed = (
        start_date_raw.notna()
        & ~start_date_raw.isin(
            [
                "",
                "0",
                "00000000",
                "<NA>",
            ]
        )
    )

    status_date_informed = (
        status_date_raw.notna()
        & ~status_date_raw.isin(
            [
                "",
                "0",
                "00000000",
                "<NA>",
            ]
        )
    )

    statistics.invalid_start_date_rows += int(
        (
            start_date_informed
            & start_date.isna()
        ).sum()
    )

    statistics.invalid_status_date_rows += int(
        (
            status_date_informed
            & status_date.isna()
        ).sum()
    )

    output = pd.DataFrame(
        {
            "cnpj": (
                cnpj_root
                + cnpj_order
                + cnpj_check_digits
            ),
            "cnpj_raiz": cnpj_root,
            "situacao_cadastral": normalize_status(
                chunk["situacao_cadastral"]
            ),
            "data_situacao_cadastral": status_date,
            "cnae_fiscal_principal": normalize_cnae(
                chunk["cnae_fiscal_principal"]
            ),
            "uf": normalize_state(
                chunk["uf"]
            ),
            "municipio": normalize_digits(
                series=chunk["municipio"],
                width=4,
            ),
            "data_inicio_atividade": start_date,
        }
    )

    output = (
        output.loc[
            valid_cnpj_mask
        ]
        .reset_index(
            drop=True
        )
    )

    text_columns = [
        "cnpj",
        "cnpj_raiz",
        "situacao_cadastral",
        "cnae_fiscal_principal",
        "uf",
        "municipio",
    ]

    for column in text_columns:
        output[column] = output[
            column
        ].astype("string")

    statistics.output_rows += len(
        output
    )

    return output


def create_sample_report(
    parquet_file: pq.ParquetFile,
) -> None:
    """Salva uma pequena amostra do núcleo cadastral."""

    if parquet_file.num_row_groups == 0:
        sample = pd.DataFrame(
            columns=OUTPUT_SCHEMA.names
        )

    else:
        sample = (
            parquet_file.read_row_group(0)
            .slice(
                0,
                30,
            )
            .to_pandas()
        )

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
    parquet_file: pq.ParquetFile,
    statistics: ProcessingStatistics,
    chunk_size: int,
) -> None:
    """Registra os principais controles do processamento."""

    valid_rate = (
        statistics.output_rows
        / statistics.input_rows
        if statistics.input_rows
        else 0.0
    )

    content = (
        "Núcleo cadastral de estabelecimentos — snapshot 2025-12\n"
        "========================================================\n\n"
        f"Diretório de entrada: {INPUT_DIR}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Tamanho dos lotes: {chunk_size:,}\n\n"
        "Controles de processamento\n"
        "--------------------------\n"
        f"Arquivos processados: {statistics.source_files:,}\n"
        f"Lotes processados: {statistics.processed_chunks:,}\n"
        f"Lotes vazios: {statistics.empty_chunks:,}\n"
        f"Linhas lidas: {statistics.input_rows:,}\n"
        f"Linhas gravadas: {statistics.output_rows:,}\n"
        "CNPJs inválidos ou ausentes: "
        f"{statistics.invalid_cnpj_rows:,}\n"
        "Datas de início inválidas: "
        f"{statistics.invalid_start_date_rows:,}\n"
        "Datas de situação inválidas: "
        f"{statistics.invalid_status_date_rows:,}\n"
        f"Taxa de registros válidos: {valid_rate:.2%}\n\n"
        "Metadados do Parquet\n"
        "--------------------\n"
        f"Linhas: {parquet_file.metadata.num_rows:,}\n"
        f"Colunas: {parquet_file.metadata.num_columns:,}\n"
        f"Row groups: {parquet_file.num_row_groups:,}\n\n"
        "Variáveis mantidas\n"
        "------------------\n"
        "- cnpj\n"
        "- cnpj_raiz\n"
        "- situacao_cadastral\n"
        "- data_situacao_cadastral\n"
        "- cnae_fiscal_principal\n"
        "- uf\n"
        "- municipio\n"
        "- data_inicio_atividade\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_pipeline(
    chunk_size: int = 200_000,
    force: bool = False,
) -> None:
    """Executa a construção do núcleo de estabelecimentos."""

    validate_parameters(
        chunk_size=chunk_size
    )

    ensure_project_directories()

    source_files = list_source_files(
        INPUT_DIR
    )

    validate_output_files(
        force=force
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".temporary.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    statistics = ProcessingStatistics(
        source_files=len(source_files)
    )

    writer: pq.ParquetWriter | None = None

    print(f"[INFO] Diretório: {INPUT_DIR}")
    print(f"[INFO] Arquivos: {len(source_files)}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(f"[INFO] Tamanho do lote: {chunk_size:,}")

    try:
        for file_index, source_file in enumerate(
            source_files,
            start=1,
        ):
            print(
                "[READ] "
                f"({file_index}/{len(source_files)}) "
                f"{source_file.relative_to(INPUT_DIR)}"
            )

            reader = pd.read_csv(
                source_file,
                sep=";",
                header=None,
                names=ESTABLISHMENT_COLUMNS,
                usecols=USE_COLUMNS,
                dtype="string",
                encoding="latin1",
                chunksize=chunk_size,
                engine="c",
                on_bad_lines="skip",
            )

            for chunk in reader:
                output = process_chunk(
                    chunk=chunk,
                    statistics=statistics,
                )

                if output.empty:
                    continue

                table = pa.Table.from_pandas(
                    output,
                    schema=OUTPUT_SCHEMA,
                    preserve_index=False,
                    safe=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        OUTPUT_SCHEMA,
                        compression="snappy",
                        use_dictionary=True,
                    )

                writer.write_table(
                    table
                )

    except Exception:
        if writer is not None:
            writer.close()

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    if writer is None:
        raise RuntimeError(
            "Nenhuma linha válida foi produzida."
        )

    writer.close()

    temporary_path.replace(
        OUTPUT_PATH
    )

    parquet_file = pq.ParquetFile(
        OUTPUT_PATH
    )

    create_sample_report(
        parquet_file
    )

    create_schema_report(
        parquet_file
    )

    create_summary_report(
        parquet_file=parquet_file,
        statistics=statistics,
        chunk_size=chunk_size,
    )

    print(
        "[OK] Núcleo cadastral criado: "
        f"{parquet_file.metadata.num_rows:,} linhas."
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
            "Constrói o núcleo cadastral de estabelecimentos "
            "para dezembro de 2025."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help=(
            "Quantidade de registros processados por lote."
        ),
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
        chunk_size=arguments.chunk_size,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()