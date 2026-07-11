"""Constrói a ABT da PGFN para o período de 2024 a 2025.

A ABT mantém somente pessoas jurídicas e cria chaves padronizadas de CNPJ
para integração com a base cadastral da Receita Federal.

O processamento é realizado em lotes com PyArrow, evitando o carregamento
integral da base na memória.

Entrada
-------
data/processed/pgfn/pgfn_sida_2024_2025.parquet

Saída principal
---------------
data/processed/pgfn/pgfn_abt_2024_2025.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from pgfn_cnpj.settings import (
    PGFN_PROCESSED_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_SAMPLES_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


INPUT_PATH = (
    PGFN_PROCESSED_DIR
    / "pgfn_sida_2024_2025.parquet"
)

OUTPUT_PATH = (
    PGFN_PROCESSED_DIR
    / "pgfn_abt_2024_2025.parquet"
)

SAMPLE_PATH = (
    REPORTS_SAMPLES_DIR
    / "pgfn_abt_2024_2025_sample.csv"
)

SCHEMA_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_abt_2024_2025_schema.csv"
)

DECISIONS_PATH = (
    REPORTS_TABLES_DIR
    / "pgfn_abt_2024_2025_decisions.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "pgfn_abt_2024_2025_summary.txt"
)


CORE_INPUT_COLUMNS = {
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

OPTIONAL_INPUT_COLUMNS = {
    "nome_devedor",
}

OUTPUT_COLUMNS = [
    "cnpj",
    "cnpj_raiz",
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
    "nome_devedor",
]


@dataclass
class ProcessingStatistics:
    """Acumula controles durante o processamento dos lotes."""

    input_rows: int = 0
    legal_entity_rows: int = 0
    output_rows: int = 0
    valid_cnpj_rows: int = 0
    processed_batches: int = 0
    empty_batches: int = 0


def validate_parameters(
    batch_size: int,
) -> None:
    """Valida os parâmetros da execução."""

    if batch_size < 1:
        raise ValueError(
            "batch_size deve ser maior ou igual a 1."
        )


def validate_input_file() -> ds.Dataset:
    """Verifica a existência e o esquema da base de entrada."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "O arquivo consolidado da PGFN não foi encontrado:\n"
            f"{INPUT_PATH}"
        )

    dataset = ds.dataset(
        INPUT_PATH,
        format="parquet",
    )

    available_columns = set(
        dataset.schema.names
    )

    missing_columns = sorted(
        CORE_INPUT_COLUMNS
        - available_columns
    )

    if missing_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise ValueError(
            "A base da PGFN não possui as variáveis necessárias:\n"
            f"{formatted_columns}"
        )

    return dataset


def validate_output_files(
    force: bool,
) -> None:
    """Evita a substituição acidental dos resultados."""

    output_files = (
        OUTPUT_PATH,
        SAMPLE_PATH,
        SCHEMA_PATH,
        DECISIONS_PATH,
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


def to_string(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Converte uma coluna para texto."""

    if (
        pa.types.is_string(array.type)
        or pa.types.is_large_string(array.type)
    ):
        return array

    return pc.cast(
        array,
        pa.string(),
    )


def clean_string(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Remove espaços e transforma textos vazios em nulos."""

    text = to_string(
        array
    )

    text = pc.utf8_trim_whitespace(
        text
    )

    empty_mask = pc.equal(
        text,
        "",
    )

    return pc.if_else(
        empty_mask,
        pa.scalar(
            None,
            pa.string(),
        ),
        text,
    )


def digits_only(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Mantém apenas os dígitos de uma coluna textual."""

    text = clean_string(
        array
    )

    return pc.replace_substring_regex(
        text,
        pattern=r"[^0-9]",
        replacement="",
    )


def normalize_cnpj(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Retorna somente CNPJs com exatamente 14 dígitos."""

    digits = digits_only(
        array
    )

    length = pc.utf8_length(
        digits
    )

    valid_mask = pc.equal(
        length,
        14,
    )

    return pc.if_else(
        valid_mask,
        digits,
        pa.scalar(
            None,
            pa.string(),
        ),
    )


def create_cnpj_root(
    cnpj: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Extrai os oito primeiros dígitos do CNPJ."""

    return pc.utf8_slice_codeunits(
        cnpj,
        start=0,
        stop=8,
    )


def normalize_legal_entity_mask(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Identifica registros referentes a pessoas jurídicas."""

    text = clean_string(
        array
    )

    upper_text = pc.utf8_upper(
        text
    )

    accepted_values = pa.array(
        [
            "PESSOA JURÍDICA",
            "PESSOA JURIDICA",
            "PJ",
        ],
        type=pa.string(),
    )

    return pc.is_in(
        upper_text,
        value_set=accepted_values,
    )


def normalize_yes_no(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Converte indicadores textuais para valores 0 e 1."""

    text = clean_string(
        array
    )

    upper_text = pc.utf8_upper(
        text
    )

    positive_mask = pc.is_in(
        upper_text,
        value_set=pa.array(
            [
                "SIM",
                "S",
                "1",
                "TRUE",
            ],
            type=pa.string(),
        ),
    )

    negative_mask = pc.is_in(
        upper_text,
        value_set=pa.array(
            [
                "NAO",
                "NÃO",
                "N",
                "0",
                "FALSE",
            ],
            type=pa.string(),
        ),
    )

    return pc.if_else(
        positive_mask,
        pa.scalar(
            1,
            pa.int8(),
        ),
        pc.if_else(
            negative_mask,
            pa.scalar(
                0,
                pa.int8(),
            ),
            pa.scalar(
                None,
                pa.int8(),
            ),
        ),
    )


def parse_date(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Converte datas conhecidas para timestamp em milissegundos."""

    if pa.types.is_timestamp(
        array.type
    ):
        return pc.cast(
            array,
            pa.timestamp("ms"),
        )

    if (
        pa.types.is_date32(array.type)
        or pa.types.is_date64(array.type)
    ):
        return pc.cast(
            array,
            pa.timestamp("ms"),
        )

    text = clean_string(
        array
    )

    brazilian_format = pc.strptime(
        text,
        format="%d/%m/%Y",
        unit="ms",
        error_is_null=True,
    )

    iso_format = pc.strptime(
        text,
        format="%Y-%m-%d",
        unit="ms",
        error_is_null=True,
    )

    return pc.coalesce(
        brazilian_format,
        iso_format,
    )


def normalize_situation_type(
    array: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    """Agrupa a situação da inscrição em cinco categorias."""

    text = clean_string(
        array
    )

    lower_text = pc.utf8_lower(
        text
    )

    result = pa.scalar(
        "outros",
        pa.string(),
    )

    result = pc.if_else(
        pc.match_substring(
            lower_text,
            "cobran",
        ),
        "irregular",
        result,
    )

    result = pc.if_else(
        pc.match_substring(
            lower_text,
            "benef",
        ),
        "beneficio_fiscal",
        result,
    )

    result = pc.if_else(
        pc.match_substring(
            lower_text,
            "negocia",
        ),
        "negociacao",
        result,
    )

    result = pc.if_else(
        pc.match_substring(
            lower_text,
            "suspens",
        ),
        "suspenso_judicial",
        result,
    )

    result = pc.if_else(
        pc.match_substring(
            lower_text,
            "garant",
        ),
        "garantia",
        result,
    )

    return pc.cast(
        result,
        pa.string(),
    )


def process_batch(
    record_batch: pa.RecordBatch,
    include_name: bool,
    statistics: ProcessingStatistics,
) -> pa.Table | None:
    """Filtra e transforma um lote da base da PGFN."""

    table = pa.Table.from_batches(
        [record_batch]
    )

    statistics.input_rows += (
        table.num_rows
    )

    statistics.processed_batches += 1

    legal_entity_mask = (
        normalize_legal_entity_mask(
            table["tipo_pessoa"]
        )
    )

    table = table.filter(
        legal_entity_mask
    )

    statistics.legal_entity_rows += (
        table.num_rows
    )

    if table.num_rows == 0:
        statistics.empty_batches += 1

        return None

    cnpj = normalize_cnpj(
        table["cpf_cnpj"]
    )

    cnpj_root = create_cnpj_root(
        cnpj
    )

    valid_cnpj_count = pc.sum(
        pc.invert(
            pc.is_null(
                cnpj
            )
        )
    ).as_py()

    statistics.valid_cnpj_rows += int(
        valid_cnpj_count or 0
    )

    columns: dict[
        str,
        pa.Array | pa.ChunkedArray,
    ] = {
        "cnpj": cnpj,
        "cnpj_raiz": cnpj_root,
        "tipo_devedor": clean_string(
            table["tipo_devedor"]
        ),
        "numero_inscricao": clean_string(
            table["numero_inscricao"]
        ),
        "situacao_inscricao": clean_string(
            table["situacao_inscricao"]
        ),
        "tipo_situacao_inscricao": (
            normalize_situation_type(
                table[
                    "tipo_situacao_inscricao"
                ]
            )
        ),
        "receita_principal": clean_string(
            table["receita_principal"]
        ),
        "data_inscricao": parse_date(
            table["data_inscricao"]
        ),
        "indicador_ajuizado": normalize_yes_no(
            table["indicador_ajuizado"]
        ),
        "valor_consolidado": pc.cast(
            table["valor_consolidado"],
            pa.float64(),
        ),
        "unidade_responsavel": clean_string(
            table["unidade_responsavel"]
        ),
        "ano": pc.cast(
            table["ano"],
            pa.int32(),
        ),
        "trimestre": pc.cast(
            table["trimestre"],
            pa.int8(),
        ),
    }

    if include_name:
        columns["nome_devedor"] = clean_string(
            table["nome_devedor"]
        )

    output_table = pa.table(
        columns
    )

    final_columns = [
        column
        for column in OUTPUT_COLUMNS
        if column in output_table.column_names
    ]

    output_table = output_table.select(
        final_columns
    )

    statistics.output_rows += (
        output_table.num_rows
    )

    return output_table


def create_decisions_report() -> None:
    """Documenta as principais decisões da construção da ABT."""

    decisions = pd.DataFrame(
        [
            {
                "elemento": "tipo_pessoa",
                "decisao": "Filtrar somente pessoas jurídicas",
                "justificativa": (
                    "A unidade de análise do projeto é a empresa."
                ),
            },
            {
                "elemento": "cnpj",
                "decisao": "Manter somente valores com 14 dígitos",
                "justificativa": (
                    "Padronização da chave cadastral completa."
                ),
            },
            {
                "elemento": "cnpj_raiz",
                "decisao": "Extrair os oito primeiros dígitos",
                "justificativa": (
                    "Permite agregação e integração por empresa."
                ),
            },
            {
                "elemento": "indicador_ajuizado",
                "decisao": "Converter para indicador binário",
                "justificativa": (
                    "Facilita agregações e modelagem estatística."
                ),
            },
            {
                "elemento": "tipo_situacao_inscricao",
                "decisao": "Agrupar em cinco categorias",
                "justificativa": (
                    "Reduz a dispersão do domínio original."
                ),
            },
            {
                "elemento": "uf e uf_devedor",
                "decisao": "Não incluir na ABT",
                "justificativa": (
                    "Não são necessárias para o recorte analítico atual."
                ),
            },
        ]
    )

    decisions.to_csv(
        DECISIONS_PATH,
        index=False,
        encoding="utf-8",
    )


def create_sample_report(
    parquet_file: pq.ParquetFile,
) -> None:
    """Salva uma pequena amostra da ABT."""

    if parquet_file.num_row_groups == 0:
        sample = pd.DataFrame()

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
    batch_size: int,
) -> None:
    """Registra controles da execução."""

    invalid_cnpj_rows = (
        statistics.output_rows
        - statistics.valid_cnpj_rows
    )

    legal_entity_rate = (
        statistics.legal_entity_rows
        / statistics.input_rows
        if statistics.input_rows
        else 0.0
    )

    valid_cnpj_rate = (
        statistics.valid_cnpj_rows
        / statistics.output_rows
        if statistics.output_rows
        else 0.0
    )

    content = (
        "ABT da PGFN — período 2024–2025\n"
        "================================\n\n"
        f"Entrada: {INPUT_PATH}\n"
        f"Saída: {OUTPUT_PATH}\n"
        f"Tamanho do lote: {batch_size:,}\n\n"
        "Controles de processamento\n"
        "--------------------------\n"
        f"Linhas de entrada: {statistics.input_rows:,}\n"
        "Linhas de pessoas jurídicas: "
        f"{statistics.legal_entity_rows:,}\n"
        f"Taxa de pessoas jurídicas: {legal_entity_rate:.2%}\n"
        f"Linhas gravadas: {statistics.output_rows:,}\n"
        f"CNPJs válidos: {statistics.valid_cnpj_rows:,}\n"
        f"CNPJs inválidos ou ausentes: {invalid_cnpj_rows:,}\n"
        f"Taxa de CNPJ válido: {valid_cnpj_rate:.2%}\n"
        f"Lotes processados: {statistics.processed_batches:,}\n"
        f"Lotes vazios após o filtro: {statistics.empty_batches:,}\n\n"
        "Metadados do Parquet de saída\n"
        "-----------------------------\n"
        f"Linhas: {parquet_file.metadata.num_rows:,}\n"
        f"Colunas: {parquet_file.metadata.num_columns:,}\n"
        f"Row groups: {parquet_file.num_row_groups:,}\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def run_pipeline(
    batch_size: int = 25_000,
    force: bool = False,
) -> None:
    """Executa a construção completa da ABT da PGFN."""

    validate_parameters(
        batch_size=batch_size
    )

    ensure_project_directories()

    dataset = validate_input_file()

    validate_output_files(
        force=force
    )

    include_name = (
        "nome_devedor"
        in dataset.schema.names
    )

    source_columns = sorted(
        CORE_INPUT_COLUMNS
        | (
            OPTIONAL_INPUT_COLUMNS
            if include_name
            else set()
        )
    )

    temporary_path = OUTPUT_PATH.with_suffix(
        ".temporary.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    scanner = dataset.scanner(
        columns=source_columns,
        batch_size=batch_size,
    )

    statistics = ProcessingStatistics()

    writer: pq.ParquetWriter | None = None

    print(f"[INFO] Entrada: {INPUT_PATH}")
    print(f"[INFO] Saída: {OUTPUT_PATH}")
    print(f"[INFO] Tamanho do lote: {batch_size:,}")
    print(
        "[INFO] Nome do devedor incluído: "
        f"{include_name}"
    )

    try:
        for record_batch in scanner.to_batches():
            output_table = process_batch(
                record_batch=record_batch,
                include_name=include_name,
                statistics=statistics,
            )

            if output_table is None:
                continue

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    output_table.schema,
                    compression="snappy",
                    use_dictionary=True,
                )

            writer.write_table(
                output_table
            )

            if (
                statistics.processed_batches
                % 50
                == 0
            ):
                print(
                    "[INFO] Lotes: "
                    f"{statistics.processed_batches:,} | "
                    "Linhas gravadas: "
                    f"{statistics.output_rows:,}"
                )

    except Exception:
        if writer is not None:
            writer.close()

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    if writer is None:
        raise RuntimeError(
            "Nenhuma linha foi produzida após o filtro "
            "de pessoas jurídicas."
        )

    writer.close()

    temporary_path.replace(
        OUTPUT_PATH
    )

    parquet_file = pq.ParquetFile(
        OUTPUT_PATH
    )

    create_decisions_report()

    create_sample_report(
        parquet_file
    )

    create_schema_report(
        parquet_file
    )

    create_summary_report(
        parquet_file=parquet_file,
        statistics=statistics,
        batch_size=batch_size,
    )

    print(
        "[OK] ABT da PGFN criada: "
        f"{parquet_file.metadata.num_rows:,} linhas."
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

    print(
        "[OK] Decisões: "
        f"{DECISIONS_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Constrói a ABT da PGFN para o período "
            "de 2024 a 2025."
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=25_000,
        help=(
            "Quantidade de registros processados por lote."
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
        batch_size=arguments.batch_size,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()