"""Gera scores e explicações para o modelo de priorização fiscal.

O processamento é realizado em lotes para evitar o carregamento integral
do dataset na memória.

Entradas
--------
data/processed/model/fiscal_risk_dataset.parquet
models/fiscal_risk/logistic_model.joblib

Saídas
------
data/processed/model/scores/fiscal_risk_scores.parquet
reports/tables/fiscal_risk_top_scores.csv
reports/tables/fiscal_risk_global_factors.csv
reports/tables/fiscal_risk_local_factors.csv
reports/metrics/fiscal_risk_scoring_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import load
from sklearn.pipeline import Pipeline

from pgfn_cnpj.settings import (
    MODEL_DATA_DIR,
    MODELS_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


IDENTIFIER_COLUMN = "cnpj_raiz"
SCORE_COLUMN = "score_risco_fiscal"

DEFAULT_DATA_PATH = (
    MODEL_DATA_DIR
    / "fiscal_risk_dataset.parquet"
)

DEFAULT_MODEL_PATH = (
    MODELS_DIR
    / "fiscal_risk"
    / "logistic_model.joblib"
)

SCORES_DIR = MODEL_DATA_DIR / "scores"

SCORES_PATH = (
    SCORES_DIR
    / "fiscal_risk_scores.parquet"
)

TOP_SCORES_PATH = (
    REPORTS_TABLES_DIR
    / "fiscal_risk_top_scores.csv"
)

GLOBAL_FACTORS_PATH = (
    REPORTS_TABLES_DIR
    / "fiscal_risk_global_factors.csv"
)

LOCAL_FACTORS_PATH = (
    REPORTS_TABLES_DIR
    / "fiscal_risk_local_factors.csv"
)

SUMMARY_PATH = (
    REPORTS_METRICS_DIR
    / "fiscal_risk_scoring_summary.txt"
)


def validate_arguments(
    batch_size: int,
    top_n: int,
    local_factors: int,
) -> None:
    """Valida os parâmetros da execução."""

    if batch_size < 1:
        raise ValueError(
            "batch_size deve ser maior ou igual a 1."
        )

    if top_n < 1:
        raise ValueError(
            "top_n deve ser maior ou igual a 1."
        )

    if local_factors < 1:
        raise ValueError(
            "local_factors deve ser maior ou igual a 1."
        )


def validate_input_files(
    data_path: Path,
    model_path: Path,
) -> None:
    """Verifica se dataset e modelo estão disponíveis."""

    missing_files = [
        path
        for path in (data_path, model_path)
        if not path.exists()
    ]

    if missing_files:
        formatted_files = "\n".join(
            f"- {path}"
            for path in missing_files
        )

        raise FileNotFoundError(
            "Os seguintes arquivos não foram encontrados:\n"
            f"{formatted_files}\n\n"
            "Execute primeiro a construção do dataset e "
            "o treinamento do modelo."
        )


def load_model_payload(
    model_path: Path,
) -> tuple[dict[str, Any], Pipeline, list[str]]:
    """Carrega e valida o artefato salvo no treinamento."""

    payload = load(model_path)

    if not isinstance(payload, dict):
        raise TypeError(
            "O arquivo do modelo não contém um payload válido."
        )

    if "model" not in payload:
        raise KeyError(
            "O payload não contém a chave 'model'."
        )

    if "features" not in payload:
        raise KeyError(
            "O payload não contém a chave 'features'."
        )

    model = payload["model"]
    features = list(payload["features"])

    if not isinstance(model, Pipeline):
        raise TypeError(
            "O objeto salvo não é um Pipeline do scikit-learn."
        )

    required_steps = {
        "imputer",
        "scaler",
        "classifier",
    }

    available_steps = set(
        model.named_steps.keys()
    )

    missing_steps = sorted(
        required_steps - available_steps
    )

    if missing_steps:
        formatted_steps = ", ".join(missing_steps)

        raise ValueError(
            "O pipeline não possui as etapas esperadas: "
            f"{formatted_steps}. "
            "Treine novamente o modelo com o módulo refatorado."
        )

    if not features:
        raise ValueError(
            "A lista de variáveis do modelo está vazia."
        )

    return payload, model, features


def validate_dataset_schema(
    data_path: Path,
    features: list[str],
) -> pq.ParquetFile:
    """Confirma se identificador e variáveis estão no Parquet."""

    parquet_file = pq.ParquetFile(data_path)

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    required_columns = {
        IDENTIFIER_COLUMN,
        *features,
    }

    missing_columns = sorted(
        required_columns - available_columns
    )

    if missing_columns:
        formatted_columns = "\n".join(
            f"- {column}"
            for column in missing_columns
        )

        raise ValueError(
            "O dataset não possui as variáveis necessárias:\n"
            f"{formatted_columns}"
        )

    return parquet_file


def validate_output_files(
    force: bool,
) -> None:
    """Evita a substituição acidental de resultados."""

    output_files = (
        SCORES_PATH,
        TOP_SCORES_PATH,
        GLOBAL_FACTORS_PATH,
        LOCAL_FACTORS_PATH,
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


def prepare_features(
    dataframe: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Prepara as variáveis na mesma ordem usada no treino."""

    return (
        dataframe[features]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
    )


def update_top_candidates(
    current_top: pd.DataFrame,
    batch_candidates: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Mantém apenas os maiores scores observados até o momento."""

    combined = pd.concat(
        [
            current_top,
            batch_candidates,
        ],
        ignore_index=True,
    )

    return (
        combined.nlargest(
            top_n,
            SCORE_COLUMN,
        )
        .reset_index(drop=True)
    )


def score_dataset(
    parquet_file: pq.ParquetFile,
    model: Pipeline,
    features: list[str],
    batch_size: int,
    top_n: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calcula scores em lotes e grava o resultado em Parquet."""

    columns = [
        IDENTIFIER_COLUMN,
        *features,
    ]

    temporary_path = SCORES_PATH.with_suffix(
        ".temporary.parquet"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    writer: pq.ParquetWriter | None = None

    top_candidates = pd.DataFrame(
        columns=[
            IDENTIFIER_COLUMN,
            *features,
            SCORE_COLUMN,
        ]
    )

    observation_count = 0
    score_sum = 0.0
    score_min = np.inf
    score_max = -np.inf

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            dataframe = batch.to_pandas()

            feature_data = prepare_features(
                dataframe=dataframe,
                features=features,
            )

            probabilities = model.predict_proba(
                feature_data
            )[:, 1]

            score_batch = pd.DataFrame(
                {
                    IDENTIFIER_COLUMN: (
                        dataframe[IDENTIFIER_COLUMN]
                        .astype("string")
                    ),
                    SCORE_COLUMN: probabilities.astype(
                        "float64"
                    ),
                }
            )

            score_table = pa.Table.from_pandas(
                score_batch,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    score_table.schema,
                    compression="snappy",
                )

            writer.write_table(score_table)

            batch_candidates = dataframe[
                [
                    IDENTIFIER_COLUMN,
                    *features,
                ]
            ].copy()

            batch_candidates[
                IDENTIFIER_COLUMN
            ] = (
                batch_candidates[
                    IDENTIFIER_COLUMN
                ]
                .astype("string")
            )

            batch_candidates[
                SCORE_COLUMN
            ] = probabilities

            top_candidates = update_top_candidates(
                current_top=top_candidates,
                batch_candidates=batch_candidates,
                top_n=top_n,
            )

            observation_count += len(
                score_batch
            )

            score_sum += float(
                np.sum(probabilities)
            )

            score_min = min(
                score_min,
                float(np.min(probabilities)),
            )

            score_max = max(
                score_max,
                float(np.max(probabilities)),
            )

    except Exception:
        if writer is not None:
            writer.close()

        if temporary_path.exists():
            temporary_path.unlink()

        raise

    if writer is None:
        raise RuntimeError(
            "O dataset não possui observações para pontuação."
        )

    writer.close()

    temporary_path.replace(SCORES_PATH)

    top_candidates = (
        top_candidates.sort_values(
            SCORE_COLUMN,
            ascending=False,
        )
        .reset_index(drop=True)
    )

    score_mean = (
        score_sum / observation_count
        if observation_count
        else 0.0
    )

    statistics = {
        "observation_count": observation_count,
        "score_mean": score_mean,
        "score_min": float(score_min),
        "score_max": float(score_max),
    }

    return top_candidates, statistics


def create_top_scores_table(
    top_candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Cria a tabela compacta das empresas mais priorizadas."""

    table = top_candidates[
        [
            IDENTIFIER_COLUMN,
            SCORE_COLUMN,
        ]
    ].copy()

    table.insert(
        0,
        "posicao",
        np.arange(
            1,
            len(table) + 1,
        ),
    )

    table.to_csv(
        TOP_SCORES_PATH,
        index=False,
        encoding="utf-8",
    )

    return table


def create_global_factors_table(
    model: Pipeline,
    features: list[str],
) -> pd.DataFrame:
    """Salva os coeficientes padronizados da regressão."""

    classifier = model.named_steps[
        "classifier"
    ]

    coefficients = classifier.coef_[0]

    table = pd.DataFrame(
        {
            "variavel": features,
            "coeficiente_padronizado": coefficients,
        }
    )

    table["valor_absoluto"] = (
        table["coeficiente_padronizado"]
        .abs()
    )

    table["razao_de_chances"] = np.exp(
        np.clip(
            table["coeficiente_padronizado"],
            -50,
            50,
        )
    )

    table["direcao"] = np.where(
        table["coeficiente_padronizado"] > 0,
        "aumenta o score",
        np.where(
            table["coeficiente_padronizado"] < 0,
            "reduz o score",
            "efeito nulo",
        ),
    )

    table = (
        table.sort_values(
            "valor_absoluto",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    table.to_csv(
        GLOBAL_FACTORS_PATH,
        index=False,
        encoding="utf-8",
    )

    return table


def create_local_factors_table(
    top_candidates: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    local_factors: int,
) -> pd.DataFrame:
    """Calcula contribuições das variáveis para os maiores scores."""

    feature_data = prepare_features(
        dataframe=top_candidates,
        features=features,
    )

    imputed_data = model.named_steps[
        "imputer"
    ].transform(feature_data)

    standardized_data = model.named_steps[
        "scaler"
    ].transform(imputed_data)

    coefficients = model.named_steps[
        "classifier"
    ].coef_[0]

    contributions = (
        standardized_data * coefficients
    )

    factor_count = min(
        local_factors,
        len(features),
    )

    records: list[dict[str, Any]] = []

    for row_index in range(
        len(top_candidates)
    ):
        row_contributions = contributions[
            row_index
        ]

        selected_indices = np.argsort(
            np.abs(row_contributions)
        )[::-1][:factor_count]

        for position, feature_index in enumerate(
            selected_indices,
            start=1,
        ):
            contribution = float(
                row_contributions[
                    feature_index
                ]
            )

            records.append(
                {
                    IDENTIFIER_COLUMN: str(
                        top_candidates.iloc[
                            row_index
                        ][IDENTIFIER_COLUMN]
                    ),
                    SCORE_COLUMN: float(
                        top_candidates.iloc[
                            row_index
                        ][SCORE_COLUMN]
                    ),
                    "posicao_fator": position,
                    "variavel": features[
                        feature_index
                    ],
                    "contribuicao": contribution,
                    "direcao": (
                        "aumenta o score"
                        if contribution > 0
                        else (
                            "reduz o score"
                            if contribution < 0
                            else "efeito nulo"
                        )
                    ),
                }
            )

    table = pd.DataFrame(records)

    table.to_csv(
        LOCAL_FACTORS_PATH,
        index=False,
        encoding="utf-8",
    )

    return table


def create_summary_report(
    data_path: Path,
    model_path: Path,
    statistics: dict[str, float | int],
    top_n: int,
    local_factors: int,
    features: list[str],
) -> None:
    """Registra um resumo da execução do scoring."""

    content = (
        "Scoring do modelo de priorização fiscal\n"
        "========================================\n\n"
        f"Dataset: {data_path}\n"
        f"Modelo: {model_path}\n"
        f"Scores completos: {SCORES_PATH}\n\n"
        "Resultados\n"
        "----------\n"
        "Empresas pontuadas: "
        f"{statistics['observation_count']:,}\n"
        "Score médio: "
        f"{statistics['score_mean']:.6f}\n"
        "Score mínimo: "
        f"{statistics['score_min']:.6f}\n"
        "Score máximo: "
        f"{statistics['score_max']:.6f}\n\n"
        "Relatórios para inspeção\n"
        "------------------------\n"
        f"Maiores scores apresentados: {top_n}\n"
        "Fatores locais por empresa: "
        f"{local_factors}\n"
        "Quantidade de variáveis do modelo: "
        f"{len(features)}\n\n"
        "Interpretação\n"
        "-------------\n"
        "O score representa a probabilidade estimada de uma "
        "empresa atender à regra operacional de priorização "
        "fiscal definida no projeto. Ele não representa uma "
        "previsão de inadimplência futura.\n"
    )

    SUMMARY_PATH.write_text(
        content,
        encoding="utf-8",
    )


def generate_scores(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    batch_size: int = 100_000,
    top_n: int = 100,
    local_factors: int = 3,
    force: bool = False,
) -> None:
    """Executa o scoring e gera explicações globais e locais."""

    validate_arguments(
        batch_size=batch_size,
        top_n=top_n,
        local_factors=local_factors,
    )

    ensure_project_directories()

    SCORES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_files(
        data_path=data_path,
        model_path=model_path,
    )

    (
        _,
        model,
        features,
    ) = load_model_payload(model_path)

    parquet_file = validate_dataset_schema(
        data_path=data_path,
        features=features,
    )

    validate_output_files(force=force)

    print(f"[INFO] Dataset: {data_path}")
    print(f"[INFO] Modelo: {model_path}")
    print(
        "[INFO] Variáveis utilizadas: "
        f"{len(features)}"
    )

    top_candidates, statistics = score_dataset(
        parquet_file=parquet_file,
        model=model,
        features=features,
        batch_size=batch_size,
        top_n=top_n,
    )

    create_top_scores_table(
        top_candidates
    )

    create_global_factors_table(
        model=model,
        features=features,
    )

    create_local_factors_table(
        top_candidates=top_candidates,
        model=model,
        features=features,
        local_factors=local_factors,
    )

    create_summary_report(
        data_path=data_path,
        model_path=model_path,
        statistics=statistics,
        top_n=min(
            top_n,
            len(top_candidates),
        ),
        local_factors=min(
            local_factors,
            len(features),
        ),
        features=features,
    )

    print(
        "[OK] Scores completos: "
        f"{SCORES_PATH}"
    )

    print(
        "[OK] Maiores scores: "
        f"{TOP_SCORES_PATH}"
    )

    print(
        "[OK] Fatores globais: "
        f"{GLOBAL_FACTORS_PATH}"
    )

    print(
        "[OK] Fatores locais: "
        f"{LOCAL_FACTORS_PATH}"
    )

    print(
        "[OK] Resumo: "
        f"{SUMMARY_PATH}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Gera scores e explicações para o modelo "
            "de priorização fiscal."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Caminho do dataset utilizado no scoring.",
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Caminho do modelo treinado.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Quantidade de registros processados por lote.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Quantidade de maiores scores no relatório.",
    )

    parser.add_argument(
        "--local-factors",
        type=int,
        default=3,
        help=(
            "Quantidade de fatores locais apresentados "
            "por empresa."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Substitui resultados existentes.",
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    generate_scores(
        data_path=arguments.data_path,
        model_path=arguments.model_path,
        batch_size=arguments.batch_size,
        top_n=arguments.top_n,
        local_factors=arguments.local_factors,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()