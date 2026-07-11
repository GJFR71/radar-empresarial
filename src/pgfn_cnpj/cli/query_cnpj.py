"""Consulta os dois modelos do projeto a partir de um CNPJ.

A entrada pode ser um CNPJ completo, com 14 dígitos, ou somente a raiz,
com 8 dígitos. A consulta busca uma única empresa diretamente nos arquivos
Parquet e calcula os scores com os modelos treinados.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import duckdb
import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

from pgfn_cnpj.settings import (
    MODEL_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
)


IDENTIFIER_COLUMN = "cnpj_raiz"

CONSULTATIONS_DIR = (
    REPORTS_DIR
    / "consultations"
)


FEATURE_LABELS = {
    "qtd_inscricoes": (
        "Quantidade de inscrições"
    ),
    "qtd_periodos": (
        "Quantidade de períodos com registros"
    ),
    "qtd_irregular": (
        "Quantidade de inscrições irregulares"
    ),
    "qtd_beneficio_fiscal": (
        "Quantidade com benefício fiscal"
    ),
    "qtd_negociacao": (
        "Quantidade em negociação"
    ),
    "qtd_suspenso_judicial": (
        "Quantidade suspensa judicialmente"
    ),
    "qtd_garantia": (
        "Quantidade com garantia"
    ),
    "qtd_estabelecimentos": (
        "Quantidade de estabelecimentos"
    ),
    "qtd_ativos": (
        "Quantidade de estabelecimentos ativos"
    ),
    "qtd_inativos": (
        "Quantidade de estabelecimentos inativos"
    ),
    "idade_empresa_dias": (
        "Idade da empresa em dias"
    ),
    "idade_empresa_anos": (
        "Idade da empresa em anos"
    ),
    "inscricoes_por_periodo": (
        "Inscrições por período"
    ),
    "pgfn_qtd_inscricoes": (
        "Quantidade de inscrições na PGFN"
    ),
    "log1p_pgfn_divida_total": (
        "Dívida total em escala logarítmica"
    ),
    "pgfn_divida_media": (
        "Valor médio das dívidas"
    ),
    "pgfn_divida_max": (
        "Maior valor individual em dívida"
    ),
    "pgfn_qtd_ajuizadas": (
        "Quantidade de inscrições ajuizadas"
    ),
    "pgfn_pct_ajuizadas": (
        "Proporção de inscrições ajuizadas"
    ),
    "pgfn_qtd_receitas_distintas": (
        "Diversidade de tipos de receita"
    ),
    "pgfn_qtd_situacoes_distintas": (
        "Diversidade de situações"
    ),
    "pgfn_pct_irregular": (
        "Proporção de inscrições irregulares"
    ),
    "pgfn_pct_beneficio_fiscal": (
        "Proporção com benefício fiscal"
    ),
    "pgfn_pct_negociacao": (
        "Proporção em negociação"
    ),
    "pgfn_pct_suspenso_judicial": (
        "Proporção suspensa judicialmente"
    ),
    "pgfn_pct_garantia": (
        "Proporção com garantia"
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """Configuração necessária para consultar um modelo."""

    key: str
    name: str
    data_path: Path
    model_path: Path
    target: str
    explanation: Literal[
        "linear",
        "tree",
    ]
    positive_text: str
    negative_text: str
    limitation: str


MODEL_SPECS = {
    "fiscal": ModelSpec(
        key="fiscal",
        name="Priorização fiscal",
        data_path=(
            MODEL_DATA_DIR
            / "fiscal_risk_dataset.parquet"
        ),
        model_path=(
            MODELS_DIR
            / "fiscal_risk"
            / "logistic_model.joblib"
        ),
        target="y_risco_fiscal",
        explanation="linear",
        positive_text=(
            "Atende à regra operacional "
            "de priorização fiscal"
        ),
        negative_text=(
            "Não atende à regra operacional "
            "de priorização fiscal"
        ),
        limitation=(
            "O score classifica uma regra operacional "
            "contemporânea; não prevê inadimplência futura."
        ),
    ),
    "discontinuity": ModelSpec(
        key="discontinuity",
        name="Descontinuidade empresarial",
        data_path=(
            MODEL_DATA_DIR
            / "business_discontinuity_dataset.parquet"
        ),
        model_path=(
            MODELS_DIR
            / "business_discontinuity"
            / "random_forest.joblib"
        ),
        target="y_descontinuidade",
        explanation="tree",
        positive_text=(
            "Sem estabelecimento ativo no snapshot"
        ),
        negative_text=(
            "Com pelo menos um estabelecimento "
            "ativo no snapshot"
        ),
        limitation=(
            "O score classifica a condição cadastral "
            "observada; não prevê encerramento futuro."
        ),
    ),
}


def normalize_cnpj_root(
    value: str,
) -> str:
    """Retorna a raiz do CNPJ com oito dígitos."""

    digits = re.sub(
        r"\D+",
        "",
        value or "",
    )

    if len(digits) == 14:
        return digits[:8]

    if len(digits) == 8:
        return digits

    raise ValueError(
        "Informe um CNPJ completo com 14 dígitos "
        "ou uma raiz com 8 dígitos."
    )


def escape_sql_path(
    path: Path,
) -> str:
    """Prepara um caminho para utilização no DuckDB."""

    return path.as_posix().replace(
        "'",
        "''",
    )


def quote_identifier(
    value: str,
) -> str:
    """Protege um nome de coluna utilizado na consulta SQL."""

    escaped_value = value.replace(
        '"',
        '""',
    )

    return f'"{escaped_value}"'


def load_model(
    specification: ModelSpec,
) -> tuple[
    Pipeline,
    list[str],
]:
    """Carrega e valida o pipeline e suas variáveis."""

    if not specification.model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado: "
            f"{specification.model_path}"
        )

    payload = load(
        specification.model_path
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "O artefato do modelo não contém "
            "um dicionário válido."
        )

    model = payload.get(
        "model"
    )

    features = list(
        payload.get(
            "features",
            [],
        )
    )

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "O artefato não contém um Pipeline válido."
        )

    if not features:
        raise ValueError(
            "A lista de variáveis do modelo está vazia."
        )

    required_steps = {
        "imputer",
        "classifier",
    }

    if specification.explanation == "linear":
        required_steps.add(
            "scaler"
        )

    missing_steps = sorted(
        required_steps
        - set(
            model.named_steps
        )
    )

    if missing_steps:
        raise ValueError(
            "O pipeline não possui as etapas esperadas: "
            f"{', '.join(missing_steps)}."
        )

    return (
        model,
        features,
    )


def load_company_row(
    specification: ModelSpec,
    cnpj_root: str,
    features: list[str],
) -> pd.DataFrame:
    """Busca uma empresa diretamente no dataset Parquet."""

    if not specification.data_path.exists():
        raise FileNotFoundError(
            "Dataset não encontrado: "
            f"{specification.data_path}"
        )

    columns = [
        IDENTIFIER_COLUMN,
        specification.target,
        *features,
    ]

    select_columns = ", ".join(
        quote_identifier(column)
        for column in columns
    )

    data_path = escape_sql_path(
        specification.data_path
    )

    query = f"""
        SELECT
            {select_columns}

        FROM read_parquet('{data_path}')

        WHERE
            LPAD(
                CAST(cnpj_raiz AS VARCHAR),
                8,
                '0'
            ) = ?

        LIMIT 2
    """

    with duckdb.connect(
        database=":memory:"
    ) as connection:
        dataframe = connection.execute(
            query,
            [cnpj_root],
        ).fetch_df()

    if dataframe.empty:
        raise LookupError(
            f"CNPJ raiz {cnpj_root} não encontrado "
            f"em {specification.data_path.name}."
        )

    if len(dataframe) > 1:
        raise RuntimeError(
            "O dataset possui mais de uma linha "
            f"para a raiz {cnpj_root}."
        )

    return dataframe


def prepare_features(
    dataframe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Converte os preditores para formato numérico."""

    return (
        dataframe[features]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )


def linear_factors(
    row: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """Calcula contribuições na escala de log-odds."""

    feature_data = prepare_features(
        row,
        features,
    )

    imputed_values = (
        model.named_steps[
            "imputer"
        ]
        .transform(
            feature_data
        )
    )

    standardized_values = (
        model.named_steps[
            "scaler"
        ]
        .transform(
            imputed_values
        )
    )

    coefficients = (
        model.named_steps[
            "classifier"
        ]
        .coef_[0]
    )

    contributions = (
        standardized_values[0]
        * coefficients
    )

    selected_indices = np.argsort(
        np.abs(
            contributions
        )
    )[::-1][:top_n]

    factors: list[
        dict[str, Any]
    ] = []

    for position, index in enumerate(
        selected_indices,
        start=1,
    ):
        contribution = float(
            contributions[index]
        )

        feature = features[
            index
        ]

        if contribution > 0:
            direction = "aumenta o score"

        elif contribution < 0:
            direction = "reduz o score"

        else:
            direction = "efeito nulo"

        factors.append(
            {
                "position": position,
                "feature": feature,
                "description": (
                    FEATURE_LABELS.get(
                        feature,
                        feature,
                    )
                ),
                "value": float(
                    imputed_values[
                        0
                    ][
                        index
                    ]
                ),
                "detail": direction,
            }
        )

    return factors


def tree_factors(
    row: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    top_n: int,
) -> list[dict[str, Any]]:
    """Cria fatores aproximados para árvore ou Random Forest."""

    feature_data = prepare_features(
        row,
        features,
    )

    imputer = model.named_steps[
        "imputer"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    values = np.asarray(
        imputer.transform(
            feature_data
        ),
        dtype="float64",
    )[0]

    reference = np.asarray(
        imputer.statistics_,
        dtype="float64",
    )

    importance = np.asarray(
        classifier.feature_importances_,
        dtype="float64",
    )

    scale = np.maximum(
        np.abs(
            reference
        ),
        1.0,
    )

    relative_deviation = (
        values
        - reference
    ) / scale

    relevance = np.abs(
        relative_deviation
        * importance
    )

    selected_indices = np.argsort(
        relevance
    )[::-1][:top_n]

    factors: list[
        dict[str, Any]
    ] = []

    for position, index in enumerate(
        selected_indices,
        start=1,
    ):
        feature = features[
            index
        ]

        if values[index] > reference[index]:
            detail = (
                "acima da mediana de referência"
            )

        elif values[index] < reference[index]:
            detail = (
                "abaixo da mediana de referência"
            )

        else:
            detail = (
                "igual à mediana de referência"
            )

        factors.append(
            {
                "position": position,
                "feature": feature,
                "description": (
                    FEATURE_LABELS.get(
                        feature,
                        feature,
                    )
                ),
                "value": float(
                    values[index]
                ),
                "detail": detail,
            }
        )

    return factors


def query_model(
    specification: ModelSpec,
    cnpj_root: str,
    top_factors: int,
) -> dict[str, Any]:
    """Calcula o score e os fatores de um modelo."""

    (
        model,
        features,
    ) = load_model(
        specification
    )

    row = load_company_row(
        specification=specification,
        cnpj_root=cnpj_root,
        features=features,
    )

    feature_data = prepare_features(
        row,
        features,
    )

    score = float(
        model.predict_proba(
            feature_data
        )[0, 1]
    )

    observed_target = int(
        row.iloc[0][
            specification.target
        ]
    )

    if specification.explanation == "linear":
        factors = linear_factors(
            row=row,
            model=model,
            features=features,
            top_n=top_factors,
        )

    else:
        factors = tree_factors(
            row=row,
            model=model,
            features=features,
            top_n=top_factors,
        )

    observed_condition = (
        specification.positive_text
        if observed_target == 1
        else specification.negative_text
    )

    return {
        "status": "ok",
        "model": specification.name,
        "score": score,
        "observed_condition": (
            observed_condition
        ),
        "limitation": (
            specification.limitation
        ),
        "factors": factors,
    }


def query_cnpj(
    cnpj_input: str,
    model_option: str = "both",
    top_factors: int = 5,
) -> dict[str, Any]:
    """Consulta um ou os dois modelos do projeto."""

    if top_factors < 1:
        raise ValueError(
            "top_factors deve ser maior ou igual a 1."
        )

    cnpj_root = normalize_cnpj_root(
        cnpj_input
    )

    model_keys = (
        [
            "fiscal",
            "discontinuity",
        ]
        if model_option == "both"
        else [
            model_option
        ]
    )

    results: list[
        dict[str, Any]
    ] = []

    for model_key in model_keys:
        specification = MODEL_SPECS[
            model_key
        ]

        try:
            result = query_model(
                specification=specification,
                cnpj_root=cnpj_root,
                top_factors=top_factors,
            )

        except (
            FileNotFoundError,
            LookupError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            result = {
                "status": "unavailable",
                "model": specification.name,
                "message": str(error),
            }

        results.append(
            result
        )

    return {
        "cnpj_input": cnpj_input,
        "cnpj_root": cnpj_root,
        "results": results,
    }


def format_number(
    value: float,
) -> str:
    """Formata números para leitura no terminal."""

    if abs(value) >= 1_000:
        text = f"{value:,.2f}"

        return (
            text.replace(
                ",",
                "X",
            )
            .replace(
                ".",
                ",",
            )
            .replace(
                "X",
                ".",
            )
        )

    return f"{value:.4f}".replace(
        ".",
        ",",
    )


def render_text(
    result: dict[str, Any],
) -> str:
    """Gera a apresentação textual da consulta."""

    lines = [
        "=" * 72,
        "Consulta analítica por CNPJ",
        f"Entrada: {result['cnpj_input']}",
        f"CNPJ raiz: {result['cnpj_root']}",
        "=" * 72,
    ]

    for model_result in result[
        "results"
    ]:
        lines.extend(
            [
                "",
                f"[{model_result['model']}]",
            ]
        )

        if model_result[
            "status"
        ] != "ok":
            lines.append(
                "Resultado indisponível: "
                f"{model_result['message']}"
            )

            continue

        score = model_result[
            "score"
        ]

        lines.append(
            f"Score: {score:.6f} ({score:.2%})"
        )

        lines.append(
            "Condição observada: "
            f"{model_result['observed_condition']}"
        )

        lines.append(
            "Principais fatores:"
        )

        for factor in model_result[
            "factors"
        ]:
            lines.append(
                f"  {factor['position']}. "
                f"{factor['description']}: "
                f"{format_number(factor['value'])} "
                f"({factor['detail']})"
            )

        lines.append(
            "Limitação: "
            f"{model_result['limitation']}"
        )

    lines.extend(
        [
            "",
            "Observação geral:",
            (
                "Os scores são instrumentos analíticos "
                "baseados em dados públicos."
            ),
            (
                "Eles não representam certeza "
                "sobre eventos futuros."
            ),
            "=" * 72,
        ]
    )

    return "\n".join(
        lines
    )


def save_text(
    result: dict[str, Any],
    text: str,
) -> Path:
    """Salva a consulta em reports/consultations."""

    CONSULTATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        CONSULTATIONS_DIR
        / (
            "consulta_cnpj_"
            f"{result['cnpj_root']}.txt"
        )
    )

    output_path.write_text(
        text,
        encoding="utf-8",
    )

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Consulta os modelos do projeto "
            "a partir de um CNPJ."
        )
    )

    parser.add_argument(
        "--cnpj",
        required=True,
        help=(
            "CNPJ completo com 14 dígitos "
            "ou raiz com 8 dígitos."
        ),
    )

    parser.add_argument(
        "--model",
        choices=[
            "fiscal",
            "discontinuity",
            "both",
        ],
        default="both",
        help="Modelo consultado.",
    )

    parser.add_argument(
        "--top-factors",
        type=int,
        default=5,
        help=(
            "Quantidade de fatores "
            "apresentados por modelo."
        ),
    )

    parser.add_argument(
        "--save-txt",
        action="store_true",
        help=(
            "Salva a consulta em "
            "reports/consultations."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    result = query_cnpj(
        cnpj_input=arguments.cnpj,
        model_option=arguments.model,
        top_factors=arguments.top_factors,
    )

    text = render_text(
        result
    )

    print(
        text
    )

    if arguments.save_txt:
        output_path = save_text(
            result,
            text,
        )

        print(
            "[OK] Consulta salva: "
            f"{output_path}"
        )


if __name__ == "__main__":
    main()