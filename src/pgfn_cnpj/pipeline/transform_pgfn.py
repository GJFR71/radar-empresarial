"""Consolida os arquivos trimestrais da PGFN em um único Parquet.

O processamento ocorre em lotes para evitar o carregamento integral da
base na memória. Os nomes das variáveis são padronizados, o valor
consolidado é convertido para número e as partições temporais são
incorporadas ao conjunto final.

Entrada
-------
data/staging/pgfn/ano=YYYY/trimestre=T/*.csv

Saída principal
---------------
data/processed/pgfn/pgfn_sida_2024_2025.parquet
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    PGFN_PROCESSED_DIR,
    PGFN_STAGING_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


START_YEAR = 2024
END_YEAR = 2025

INPUT_DIR = PGFN_STAGING_DIR

OUTPUT_PATH = (
    PGFN_PROCESSED_DIR
    / "pgfn_sida_2024_2025.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "pgfn_sida_2024_2025_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_sida_2024_2025_schema.csv"
)

SOURCES_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_sida_2024_2025_sources.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "pgfn_sida_2024_2025_summary.txt"
)


UF_PATTERN = re.compile(
    r"_SIDA_([A-Z]{2})(?:_|\.|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceFile:
    """Armazena os metadados necessários para ler um CSV."""

    path: Path
    year: int
    quarter: int
    state: str
    separator: str
    encoding: str
    columns: tuple[str, ...]


@dataclass
class ProcessingStatistics:
    """Acumula controles da transformação."""

    source_files: int = 0
    processed_chunks: int = 0
    input_rows: int = 0
    output_rows: int = 0
    invalid_value_rows: int = 0
    empty_chunks: int = 0


def validate_parameters(
    chunk_size: int,
    sample_rows: int,
) -> None:
    """Valida os parâmetros informados pela interface."""

    if chunk_size < 1:
        raise ValueError(
            "chunk_size deve ser maior ou igual a 1."
        )

    if sample_rows < 0:
        raise ValueError(
            "sample_rows deve ser maior ou igual a zero."
        )


def slugify_column(name: object) -> str:
    """Converte o nome de uma variável para snake_case."""

    normalized = unicodedata.normalize(
        "NFKD",
        str(name),
    )

    normalized = (
        normalized.encode(
            "ascii",
            "ignore",
        )
        .decode("ascii")
        .strip()
        .lower()
    )

    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        normalized,
    )

    return normalized.strip("_")


def normalize_column_names(
    columns: list[object],
) -> list[str]:
    """Padroniza os nomes e impede colisões após a conversão."""

    normalized = [
        slugify_column(column)
        for column in columns
    ]

    empty_positions = [
        index
        for index, column in enumerate(normalized)
        if not column
    ]

    if empty_positions:
        raise ValueError(
            "Foram encontrados nomes de coluna vazios "
            "após a padronização."
        )

    duplicated = sorted(
        {
            column
            for column in normalized
            if normalized.count(column) > 1
        }
    )

    if duplicated:
        formatted_columns = ", ".join(
            duplicated
        )

        raise ValueError(
            "Colunas duplicadas após a padronização: "
            f"{formatted_columns}."
        )

    return normalized


def parse_partitions_from_path(
    csv_path: Path,
) -> tuple[int, int]:
    """Obtém ano e trimestre a partir das pastas de partição."""

    year: int | None = None
    quarter: int | None = None

    for part in csv_path.parts:
        if part.startswith("ano="):
            year = int(
                part.split("=", 1)[1]
            )

        elif part.startswith("trimestre="):
            quarter = int(
                part.split("=", 1)[1]
            )

    if year is None or quarter is None:
        raise ValueError(
            "Não foi possível identificar ano e trimestre "
            f"no caminho: {csv_path}"
        )

    if quarter not in {1, 2, 3, 4}:
        raise ValueError(
            f"Trimestre inválido no caminho: {csv_path}"
        )

    return year, quarter


def parse_state_from_filename(
    csv_path: Path,
) -> str:
    """Obtém a UF a partir do nome de um arquivo da PGFN."""

    match = UF_PATTERN.search(
        csv_path.name
    )

    if match is None:
        return "NA"

    return match.group(1).upper()


def detect_separator(
    path: Path,
) -> str:
    """Identifica se o arquivo usa ponto e vírgula ou vírgula."""

    with path.open("rb") as file:
        sample = file.read(
            16_384
        )

    text = sample.decode(
        "latin-1",
        errors="ignore",
    )

    first_line = text.splitlines()[0] if text else ""

    semicolons = first_line.count(";")
    commas = first_line.count(",")

    return ";" if semicolons >= commas else ","


def detect_encoding(
    path: Path,
) -> str:
    """Distingue arquivos UTF-8 dos arquivos compatíveis com Latin-1."""

    with path.open("rb") as file:
        sample = file.read(
            1_048_576
        )

    try:
        sample.decode(
            "utf-8-sig",
            errors="strict",
        )

    except UnicodeDecodeError:
        return "latin-1"

    return "utf-8-sig"


def read_header(
    path: Path,
    separator: str,
    encoding: str,
) -> tuple[str, ...]:
    """Lê e padroniza somente o cabeçalho de um CSV."""

    header = pd.read_csv(
        path,
        sep=separator,
        encoding=encoding,
        nrows=0,
        dtype="string",
        engine="c",
        on_bad_lines="skip",
    )

    columns = normalize_column_names(
        header.columns.tolist()
    )

    if not columns:
        raise ValueError(
            f"O arquivo não possui cabeçalho válido: {path}"
        )

    return tuple(columns)


def list_source_paths(
    directory: Path,
) -> list[Path]:
    """Localiza recursivamente os CSVs do período selecionado."""

    if not directory.exists():
        raise FileNotFoundError(
            "O diretório de staging da PGFN não foi encontrado:\n"
            f"{directory}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.ingestion.extract_pgfn"
        )

    selected_files: list[Path] = []

    for path in directory.rglob("*"):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.suffix.lower() != ".csv"
            or path.stat().st_size == 0
        ):
            continue

        try:
            year, _ = parse_partitions_from_path(
                path
            )

        except ValueError:
            continue

        if START_YEAR <= year <= END_YEAR:
            selected_files.append(
                path
            )

    selected_files.sort()

    if not selected_files:
        raise FileNotFoundError(
            "Nenhum CSV da PGFN foi encontrado para "
            f"o período {START_YEAR}–{END_YEAR} em:\n"
            f"{directory}"
        )

    return selected_files


def inspect_source_files(
    paths: list[Path],
) -> list[SourceFile]:
    """Inspeciona as partições, codificações e cabeçalhos."""

    sources: list[SourceFile] = []

    for path in paths:
        year, quarter = parse_partitions_from_path(
            path
        )

        separator = detect_separator(
            path
        )

        encoding = detect_encoding(
            path
        )

        columns = read_header(
            path=path,
            separator=separator,
            encoding=encoding,
        )

        sources.append(
            SourceFile(
                path=path,
                year=year,
                quarter=quarter,
                state=parse_state_from_filename(
                    path
                ),
                separator=separator,
                encoding=encoding,
                columns=columns,
            )
        )

    return sources


def build_unified_columns(
    sources: list[SourceFile],
) -> list[str]:
    """Constrói a união ordenada das variáveis encontradas."""

    unified: list[str] = []
    observed: set[str] = set()

    for source in sources:
        for column in source.columns:
            if column not in observed:
                observed.add(
                    column
                )

                unified.append(
                    column
                )

    metadata_columns = [
        "ano",
        "trimestre",
        "uf",
        "arquivo_origem",
    ]

    for column in metadata_columns:
        if column not in observed:
            unified.append(
                column
            )

    return unified


def build_arrow_schema(
    columns: list[str],
) -> pa.Schema:
    """Define um esquema estável para o Parquet final."""

    fields: list[pa.Field] = []

    for column in columns:
        if column == "valor_consolidado":
            data_type = pa.float64()

        elif column == "ano":
            data_type = pa.int32()

        elif column == "trimestre":
            data_type = pa.int8()

        else:
            data_type = pa.string()

        fields.append(
            pa.field(
                column,
                data_type,
            )
        )

    return pa.schema(
        fields
    )


def normalize_text(
    series: pd.Series,
) -> pd.Series:
    """Remove espaços e transforma textos vazios em nulos."""

    normalized = (
        series.astype("string")
        .str.strip()
    )

    return normalized.mask(
        normalized.isin(
            [
                "",
                "nan",
                "NaN",
                "<NA>",
            ]
        )
    )


def parse_monetary_values(
    series: pd.Series,
) -> pd.Series:
    """Converte formatos monetários brasileiros e internacionais."""

    text = normalize_text(
        series
    )

    cleaned = text.str.replace(
        r"[^0-9,.\-]",
        "",
        regex=True,
    )

    result = cleaned.copy()

    has_comma = cleaned.str.contains(
        ",",
        regex=False,
        na=False,
    )

    has_dot = cleaned.str.contains(
        ".",
        regex=False,
        na=False,
    )

    both_separators = (
        has_comma
        & has_dot
    )

    comma_is_decimal = (
        both_separators
        & (
            cleaned.str.rfind(",")
            > cleaned.str.rfind(".")
        )
    ).fillna(False)

    dot_is_decimal = (
        both_separators
        & ~comma_is_decimal
    ).fillna(False)

    result.loc[
        comma_is_decimal
    ] = (
        cleaned.loc[
            comma_is_decimal
        ]
        .str.replace(
            ".",
            "",
            regex=False,
        )
        .str.replace(
            ",",
            ".",
            regex=False,
        )
    )

    result.loc[
        dot_is_decimal
    ] = (
        cleaned.loc[
            dot_is_decimal
        ]
        .str.replace(
            ",",
            "",
            regex=False,
        )
    )

    only_comma = (
        has_comma
        & ~has_dot
    )

    multiple_commas = (
        only_comma
        & cleaned.str.count(",").gt(1)
    ).fillna(False)

    single_comma = (
        only_comma
        & ~multiple_commas
    ).fillna(False)

    result.loc[
        multiple_commas
    ] = cleaned.loc[
        multiple_commas
    ].str.replace(
        ",",
        "",
        regex=False,
    )

    result.loc[
        single_comma
    ] = cleaned.loc[
        single_comma
    ].str.replace(
        ",",
        ".",
        regex=False,
    )

    only_dot = (
        has_dot
        & ~has_comma
    )

    multiple_dots = (
        only_dot
        & cleaned.str.count(r"\.").gt(1)
    ).fillna(False)

    result.loc[
        multiple_dots
    ] = cleaned.loc[
        multiple_dots
    ].str.replace(
        ".",
        "",
        regex=False,
    )

    return pd.to_numeric(
        result,
        errors="coerce",
    )


def iterate_chunks(
    source: SourceFile,
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    """Lê um arquivo em lotes controlados."""

    reader = pd.read_csv(
        source.path,
        sep=source.separator,
        encoding=source.encoding,
        dtype="string",
        chunksize=chunk_size,
        engine="c",
        on_bad_lines="skip",
        low_memory=False,
    )

    yield from reader


def prepare_chunk(
    chunk: pd.DataFrame,
    source: SourceFile,
    unified_columns: list[str],
    statistics: ProcessingStatistics,
) -> pd.DataFrame:
    """Padroniza um lote antes da gravação."""

    statistics.processed_chunks += 1
    statistics.input_rows += len(
        chunk
    )

    if chunk.empty:
        statistics.empty_chunks += 1

        return pd.DataFrame(
            columns=unified_columns
        )

    normalized_columns = normalize_column_names(
        chunk.columns.tolist()
    )

    if tuple(normalized_columns) != source.columns:
        raise ValueError(
            "O cabeçalho mudou durante a leitura do arquivo: "
            f"{source.path}"
        )

    chunk.columns = normalized_columns

    for column in chunk.columns:
        chunk[column] = normalize_text(
            chunk[column]
        )

    if "valor_consolidado" in chunk.columns:
        raw_values = chunk[
            "valor_consolidado"
        ]

        informed_mask = raw_values.notna()

        parsed_values = parse_monetary_values(
            raw_values
        )

        statistics.invalid_value_rows += int(
            (
                informed_mask
                & parsed_values.isna()
            ).sum()
        )

        chunk[
            "valor_consolidado"
        ] = parsed_values

    chunk["ano"] = source.year
    chunk["trimestre"] = source.quarter
    chunk["uf"] = source.state
    chunk["arquivo_origem"] = source.path.name

    for column in unified_columns:
        if column not in chunk.columns:
            chunk[column] = pd.NA

    chunk = chunk[
        unified_columns
    ]

    text_columns = [
        column
        for column in unified_columns
        if column not in {
            "valor_consolidado",
            "ano",
            "trimestre",
        }
    ]

    for column in text_columns:
        chunk[column] = chunk[
            column
        ].astype("string")

    chunk["ano"] = pd.Series(
        source.year,
        index=chunk.index,
        dtype="int32",
    )

    chunk["trimestre"] = pd.Series(
        source.quarter,
        index=chunk.index,
        dtype="int8",
    )

    statistics.output_rows += len(
        chunk
    )

    return chunk


def validate_output_files(
    force: bool,
) -> None:
    """Evita a substituição acidental dos resultados."""

    output_files = (
        OUTPUT_PATH,
        SAMPLE_PATH,
        SCHEMA_PATH,
        SOURCES_PATH,
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
            "Os seguintes resultados já existem:\n"
            f"{formatted_files}\n\n"
            "Use --force para substituí-los."
        )

    if force:
        for path in existing_files:
            path.unlink()


def create_sample_report(
    sample_parts: list[pd.DataFrame],
) -> None:
    """Salva a amostra acumulada durante o processamento."""

    if sample_parts:
        sample = pd.concat(
            sample_parts,
            ignore_index=True,
        )

    else:
        sample = pd.DataFrame()

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


def create_sources_report(
    sources: list[SourceFile],
) -> None:
    """Documenta todos os arquivos utilizados na consolidação."""

    table = pd.DataFrame(
        [
            {
                "arquivo": str(
                    source.path
                ),
                "ano": source.year,
                "trimestre": source.quarter,
                "uf": source.state,
                "separador": source.separator,
                "encoding": source.encoding,
                "quantidade_colunas": len(
                    source.columns
                ),
            }
            for source in sources
        ]
    )

    table.to_csv(
        SOURCES_PATH,
        index=False,
        encoding="utf-8",
    )


def create_summary_report(
    parquet_file: pq.ParquetFile,
    sources: list[SourceFile],
    statistics: ProcessingStatistics,
    chunk_size: int,
    sample_rows: int,
) -> None:
    """Registra os principais controles da consolidação."""

    encoding_counts = Counter(
        source.encoding
        for source in sources
    )

    separator_counts = Counter(
        source.separator
        for source in sources
    )

    encoding_text = ", ".join(
        f"{encoding}={count}"
        for encoding, count in sorted(
            encoding_counts.items()
        )
    )

    separator_text = ", ".join(
        (
            "ponto_e_virgula"
            if separator == ";"
            else "virgula"
        )
        + f"={count}"
        for separator, count in sorted(
            separator_counts.items()
        )
    )

    content = (
        "Base consolidada da PGFN — período 2024–2025\n"
        "================================================\n\n"
        f"Diretório de entrada: {INPUT_DIR}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Período: {START_YEAR}–{END_YEAR}\n"
        f"Tamanho dos lotes: {chunk_size:,}\n"
        f"Limite da amostra: {sample_rows:,}\n\n"
        "Fontes\n"
        "------\n"
        f"Arquivos processados: {statistics.source_files:,}\n"
        f"Encodings: {encoding_text}\n"
        f"Separadores: {separator_text}\n\n"
        "Controles de processamento\n"
        "--------------------------\n"
        f"Lotes processados: {statistics.processed_chunks:,}\n"
        f"Lotes vazios: {statistics.empty_chunks:,}\n"
        f"Linhas lidas: {statistics.input_rows:,}\n"
        f"Linhas gravadas: {statistics.output_rows:,}\n"
        "Valores consolidados informados e não convertidos: "
        f"{statistics.invalid_value_rows:,}\n\n"
        "Metadados do Parquet\n"
        "--------------------\n"
        f"Linhas: {parquet_file.metadata.num_rows:,}\n"
        f"Colunas: {parquet_file.metadata.num_columns:,}\n"
        f"Row groups: {parquet_file.num_row_groups:,}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_pipeline(
    chunk_size: int = 200_000,
    sample_rows: int = 100_000,
    force: bool = False,
) -> None:
    """Executa a consolidação completa dos arquivos da PGFN."""

    validate_parameters(
        chunk_size=chunk_size,
        sample_rows=sample_rows,
    )

    ensure_project_directories()

    source_paths = list_source_paths(
        INPUT_DIR
    )

    sources = inspect_source_files(
        source_paths
    )

    unified_columns = build_unified_columns(
        sources
    )

    arrow_schema = build_arrow_schema(
        unified_columns
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
        source_files=len(sources)
    )

    sample_parts: list[pd.DataFrame] = []
    sampled_rows = 0

    writer: pq.ParquetWriter | None = None

    print(f"[INFO] Entrada: {INPUT_DIR}")
    print(f"[INFO] Período: {START_YEAR}–{END_YEAR}")
    print(f"[INFO] Arquivos: {len(sources)}")
    print(f"[INFO] Colunas consolidadas: {len(unified_columns)}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(f"[INFO] Tamanho do lote: {chunk_size:,}")

    try:
        for source_index, source in enumerate(
            sources,
            start=1,
        ):
            print(
                "[READ] "
                f"({source_index}/{len(sources)}) "
                f"ano={source.year} "
                f"trimestre={source.quarter} "
                f"uf={source.state} "
                f"arquivo={source.path.name}"
            )

            for chunk in iterate_chunks(
                source=source,
                chunk_size=chunk_size,
            ):
                output = prepare_chunk(
                    chunk=chunk,
                    source=source,
                    unified_columns=unified_columns,
                    statistics=statistics,
                )

                if output.empty:
                    continue

                table = pa.Table.from_pandas(
                    output,
                    schema=arrow_schema,
                    preserve_index=False,
                    safe=False,
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temporary_path,
                        arrow_schema,
                        compression="snappy",
                        use_dictionary=True,
                    )

                writer.write_table(
                    table
                )

                if sampled_rows < sample_rows:
                    remaining = (
                        sample_rows
                        - sampled_rows
                    )

                    sample_part = output.head(
                        remaining
                    ).copy()

                    sample_parts.append(
                        sample_part
                    )

                    sampled_rows += len(
                        sample_part
                    )

    except Exception:
        if writer is not None:
            writer.close()

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    if writer is None:
        raise RuntimeError(
            "Nenhuma linha foi produzida durante "
            "a consolidação da PGFN."
        )

    writer.close()

    temporary_path.replace(
        OUTPUT_PATH
    )

    parquet_file = pq.ParquetFile(
        OUTPUT_PATH
    )

    if (
        parquet_file.metadata.num_rows
        != statistics.output_rows
    ):
        raise RuntimeError(
            "A contagem do Parquet não corresponde à "
            "quantidade de linhas processadas."
        )

    create_sample_report(
        sample_parts
    )

    create_schema_report(
        parquet_file
    )

    create_sources_report(
        sources
    )

    create_summary_report(
        parquet_file=parquet_file,
        sources=sources,
        statistics=statistics,
        chunk_size=chunk_size,
        sample_rows=sample_rows,
    )

    print(
        "[OK] Base consolidada: "
        f"{parquet_file.metadata.num_rows:,} linhas."
    )

    print(
        f"[OK] Saída: {OUTPUT_PATH}"
    )

    print(
        f"[OK] Resumo: {SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos da linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Consolida os arquivos trimestrais da PGFN "
            "referentes a 2024 e 2025."
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
        "--sample-rows",
        type=int,
        default=100_000,
        help=(
            "Quantidade máxima de registros mantidos "
            "para a amostra de relatório."
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
        sample_rows=arguments.sample_rows,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()