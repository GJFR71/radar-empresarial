"""Funções compartilhadas para validação de arquivos Parquet."""

from __future__ import annotations

import io
import math
from collections.abc import Collection

import pandas as pd
import pyarrow.parquet as pq


def read_row_group_dataframe(
    parquet_file: pq.ParquetFile,
    row_group_index: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Lê um row group e o converte para DataFrame."""

    table = parquet_file.read_row_group(
        row_group_index,
        columns=columns,
    )

    return table.to_pandas()


def evenly_spaced_indices(
    total: int,
    count: int,
) -> list[int]:
    """Seleciona índices aproximadamente equidistantes."""

    if total < 1 or count < 1:
        return []

    if count >= total:
        return list(range(total))

    if count == 1:
        return [0]

    indices = {
        round(
            position
            * (total - 1)
            / (count - 1)
        )
        for position in range(count)
    }

    return sorted(indices)


def build_systematic_sample(
    parquet_file: pq.ParquetFile,
    max_rows: int,
    max_row_groups: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Monta uma amostra distribuída entre os row groups."""

    selected_columns = (
        columns
        if columns is not None
        else parquet_file.schema_arrow.names
    )

    if max_rows < 1 or parquet_file.num_row_groups == 0:
        return pd.DataFrame(
            columns=selected_columns
        )

    row_group_indices = evenly_spaced_indices(
        total=parquet_file.num_row_groups,
        count=max_row_groups,
    )

    if not row_group_indices:
        return pd.DataFrame(
            columns=selected_columns
        )

    rows_per_group = math.ceil(
        max_rows / len(row_group_indices)
    )

    sample_parts: list[pd.DataFrame] = []

    for row_group_index in row_group_indices:
        part = read_row_group_dataframe(
            parquet_file=parquet_file,
            row_group_index=row_group_index,
            columns=columns,
        )

        if part.empty:
            continue

        sample_parts.append(
            part.head(
                rows_per_group
            )
        )

    if not sample_parts:
        return pd.DataFrame(
            columns=selected_columns
        )

    sample = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    return sample.head(
        max_rows
    )


def build_schema_table(
    parquet_file: pq.ParquetFile,
) -> pd.DataFrame:
    """Cria uma tabela com o esquema do Parquet."""

    schema = parquet_file.schema_arrow

    return pd.DataFrame(
        [
            {
                "variavel": field.name,
                "tipo_parquet": str(field.type),
                "permite_nulo": field.nullable,
            }
            for field in schema
        ]
    )


def build_null_profile(
    sample: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula nulos e cardinalidade na amostra."""

    sample_rows = len(sample)

    rows: list[dict[str, object]] = []

    for column in sample.columns:
        series = sample[column]

        null_rows = int(
            series.isna().sum()
        )

        null_rate = (
            null_rows / sample_rows
            if sample_rows
            else 0.0
        )

        rows.append(
            {
                "variavel": column,
                "tipo_amostra": str(series.dtype),
                "linhas_amostra": sample_rows,
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

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "percentual_nulos",
                "variavel",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


def suggested_type(
    column: str,
    series: pd.Series,
    distinct_values: int,
) -> str:
    """Sugere um tipo estrutural para uma variável."""

    if column == "valor_consolidado":
        return "float64"

    if column in {
        "ano",
        "trimestre",
    }:
        return "inteiro"

    if (
        "data" in column.lower()
        or pd.api.types.is_datetime64_any_dtype(
            series
        )
    ):
        return "datetime"

    if pd.api.types.is_numeric_dtype(
        series
    ):
        return str(series.dtype)

    if distinct_values <= 80:
        return "categoria"

    return "texto"


def build_decision_table(
    sample: pd.DataFrame,
    protected_columns: Collection[str],
) -> pd.DataFrame:
    """Cria recomendações diagnósticas sem excluir variáveis."""

    sample_rows = len(sample)

    rows: list[dict[str, object]] = []

    for column in sample.columns:
        series = sample[column]

        null_rows = int(
            series.isna().sum()
        )

        null_rate = (
            null_rows / sample_rows
            if sample_rows
            else 0.0
        )

        distinct_values = int(
            series.nunique(
                dropna=True
            )
        )

        is_constant = (
            distinct_values <= 1
        )

        if column in protected_columns:
            action = "manter"

            reason = (
                "Variável protegida por sua função "
                "estrutural ou analítica."
            )

        elif null_rate == 1.0:
            action = "revisar"

            reason = (
                "Variável totalmente nula na amostra; "
                "confirmar no conjunto completo."
            )

        elif is_constant:
            action = "revisar"

            reason = (
                "Variável constante na amostra; "
                "confirmar representatividade."
            )

        else:
            action = "manter"

            reason = (
                "A variável apresenta informação "
                "na amostra analisada."
            )

        rows.append(
            {
                "variavel": column,
                "tipo_atual": str(series.dtype),
                "tipo_sugerido": suggested_type(
                    column=column,
                    series=series,
                    distinct_values=distinct_values,
                ),
                "acao_sugerida": action,
                "percentual_nulos": round(
                    null_rate * 100,
                    2,
                ),
                "valores_distintos": distinct_values,
                "constante_na_amostra": is_constant,
                "justificativa": reason,
            }
        )

    action_order = {
        "revisar": 0,
        "manter": 1,
    }

    table = pd.DataFrame(rows)

    if table.empty:
        return table

    table["_ordem"] = table[
        "acao_sugerida"
    ].map(action_order)

    return (
        table.sort_values(
            [
                "_ordem",
                "percentual_nulos",
                "variavel",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .drop(
            columns="_ordem"
        )
        .reset_index(
            drop=True
        )
    )


def dataframe_info_text(
    dataframe: pd.DataFrame,
) -> str:
    """Captura a saída de DataFrame.info como texto."""

    buffer = io.StringIO()

    dataframe.info(
        buf=buffer,
        verbose=True,
        show_counts=True,
    )

    return buffer.getvalue()