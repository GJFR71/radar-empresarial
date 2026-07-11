"""Gera scores e explicações do modelo de descontinuidade empresarial.

O processamento é realizado em lotes para evitar o carregamento integral
do dataset na memória.

As explicações locais apresentadas neste módulo são aproximações baseadas
na distância entre o valor observado e a mediana utilizada pelo imputador,
ponderada pela importância global da variável. Elas não representam uma
decomposição exata da predição da árvore.

Entradas
--------
data/processed/model/business_discontinuity_dataset.parquet

models/business_discontinuity/decision_tree.joblib
ou
models/business_discontinuity/random_forest.joblib

Saídas
------
data/processed/model/scores/business_discontinuity_<modelo>_scores.parquet

reports/tables/business_discontinuity_<modelo>_top_scores.csv
reports/tables/business_discontinuity_<modelo>_global_importance.csv
reports/tables/business_discontinuity_<modelo>_local_factors.csv

reports/metrics/business_discontinuity_<modelo>_scoring_summary.txt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
TARGET_COLUMN = "y_descontinuidade"
SCORE_COLUMN = "score_descontinuidade"

DEFAULT_DATA_PATH = (
    MODEL_DATA_DIR
    / "business_discontinuity_dataset.parquet"
)

DEFAULT_RF_MODEL_PATH = (
    MODELS_DIR
    / "business_discontinuity"
    / "random_forest.joblib"
)

SCORES_DIR = (
    MODEL_DATA_DIR
    / "scores"
)


FEATURE_LABELS = {
    "qtd_estabelecimentos": (
        "Quantidade de estabelecimentos"
    ),
    "idade_empresa_anos": (
        "Idade da empresa em anos"
    ),
    "pgfn_qtd_inscricoes": (
        "Quantidade de inscrições na PGFN"
    ),
    "log1p_pgfn_divida_total": (
        "Valor total da dívida em escala logarítmica"
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
        "Diversidade de situações das inscrições"
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
class OutputPaths:
    """Agrupa os arquivos produzidos por modelo."""

    scores: Path
    top_scores: Path
    global_importance: Path
    local_factors: Path
    summary: Path


def build_output_paths(
    model_key: str,
) -> OutputPaths:
    """Define os nomes dos arquivos conforme o algoritmo."""

    prefix = (
        f"business_discontinuity_{model_key}"
    )

    return OutputPaths(
        scores=(
            SCORES_DIR
            / f"{prefix}_scores.parquet"
        ),
        top_scores=(
            REPORTS_TABLES_DIR
            / f"{prefix}_top_scores.csv"
        ),
        global_importance=(
            REPORTS_TABLES_DIR
            / f"{prefix}_global_importance.csv"
        ),
        local_factors=(
            REPORTS_TABLES_DIR
            / f"{prefix}_local_factors.csv"
        ),
        summary=(
            REPORTS_METRICS_DIR
            / f"{prefix}_scoring_summary.txt"
        ),
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
    """Verifica a existência do dataset e do modelo."""

    missing_files = [
        path
        for path in (
            data_path,
            model_path,
        )
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
) -> tuple[
    dict[str, Any],
    Pipeline,
    list[str],
    str,
    str,
]:
    """Carrega e valida o artefato produzido no treinamento."""

    payload = load(
        model_path
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "O arquivo do modelo não contém um payload válido."
        )

    required_keys = {
        "model",
        "features",
        "model_key",
        "model_name",
    }

    missing_keys = sorted(
        required_keys
        - set(payload.keys())
    )

    if missing_keys:
        formatted_keys = ", ".join(
            missing_keys
        )

        raise KeyError(
            "O payload não contém as chaves necessárias: "
            f"{formatted_keys}. "
            "Treine novamente o modelo com o módulo refatorado."
        )

    model = payload[
        "model"
    ]

    features = list(
        payload["features"]
    )

    model_key = str(
        payload["model_key"]
    )

    model_name = str(
        payload["model_name"]
    )

    if not isinstance(
        model,
        Pipeline,
    ):
        raise TypeError(
            "O objeto salvo não é um Pipeline do scikit-learn."
        )

    required_steps = {
        "imputer",
        "classifier",
    }

    available_steps = set(
        model.named_steps.keys()
    )

    missing_steps = sorted(
        required_steps
        - available_steps
    )

    if missing_steps:
        formatted_steps = ", ".join(
            missing_steps
        )

        raise ValueError(
            "O pipeline não possui as etapas esperadas: "
            f"{formatted_steps}."
        )

    classifier = model.named_steps[
        "classifier"
    ]

    if not hasattr(
        classifier,
        "feature_importances_",
    ):
        raise ValueError(
            "O classificador não possui feature_importances_."
        )

    if not features:
        raise ValueError(
            "A lista de variáveis do modelo está vazia."
        )

    return (
        payload,
        model,
        features,
        model_key,
        model_name,
    )


def validate_dataset_schema(
    data_path: Path,
    features: list[str],
) -> tuple[
    pq.ParquetFile,
    bool,
]:
    """Confirma se identificador e preditores existem no Parquet."""

    parquet_file = pq.ParquetFile(
        data_path
    )

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    required_columns = {
        IDENTIFIER_COLUMN,
        *features,
    }

    missing_columns = sorted(
        required_columns
        - available_columns
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

    has_target = (
        TARGET_COLUMN
        in available_columns
    )

    return (
        parquet_file,
        has_target,
    )


def validate_output_files(
    paths: OutputPaths,
    force: bool,
) -> None:
    """Evita substituição acidental dos resultados."""

    output_files = (
        paths.scores,
        paths.top_scores,
        paths.global_importance,
        paths.local_factors,
        paths.summary,
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
    """Converte os preditores para formato numérico."""

    feature_data = (
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

    return feature_data


def update_top_candidates(
    current_top: pd.DataFrame,
    batch_candidates: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Mantém apenas os maiores scores observados."""

    if current_top.empty:
        return (
            batch_candidates.nlargest(
                top_n,
                SCORE_COLUMN,
            )
            .reset_index(
                drop=True
            )
        )

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
        .reset_index(
            drop=True
        )
    )


def score_dataset(
    parquet_file: pq.ParquetFile,
    model: Pipeline,
    features: list[str],
    has_target: bool,
    batch_size: int,
    top_n: int,
    scores_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, float | int],
]:
    """Calcula os scores em lotes e grava o Parquet completo."""

    columns = [
        IDENTIFIER_COLUMN,
        *features,
    ]

    if has_target:
        columns.append(
            TARGET_COLUMN
        )

    temporary_path = (
        scores_path.with_suffix(
            ".temporary.parquet"
        )
    )

    if temporary_path.exists():
        temporary_path.unlink()

    writer: pq.ParquetWriter | None = None

    top_candidates = pd.DataFrame()

    observation_count = 0
    score_sum = 0.0
    score_min = np.inf
    score_max = -np.inf

    try:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=columns,
        ):
            dataframe = (
                batch.to_pandas()
            )

            feature_data = prepare_features(
                dataframe=dataframe,
                features=features,
            )

            probabilities = (
                model.predict_proba(
                    feature_data
                )[:, 1]
            )

            score_data: dict[
                str,
                Any,
            ] = {
                IDENTIFIER_COLUMN: (
                    dataframe[
                        IDENTIFIER_COLUMN
                    ]
                    .astype("string")
                ),
                SCORE_COLUMN: (
                    probabilities.astype(
                        "float64"
                    )
                ),
            }

            if has_target:
                score_data[
                    TARGET_COLUMN
                ] = (
                    dataframe[
                        TARGET_COLUMN
                    ]
                    .astype(
                        "int8"
                    )
                )

            score_batch = pd.DataFrame(
                score_data
            )

            score_table = (
                pa.Table.from_pandas(
                    score_batch,
                    preserve_index=False,
                )
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_path,
                    score_table.schema,
                    compression="snappy",
                )

            writer.write_table(
                score_table
            )

            candidate_columns = [
                IDENTIFIER_COLUMN,
                *features,
            ]

            if has_target:
                candidate_columns.append(
                    TARGET_COLUMN
                )

            batch_candidates = dataframe[
                candidate_columns
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

            top_candidates = (
                update_top_candidates(
                    current_top=top_candidates,
                    batch_candidates=(
                        batch_candidates
                    ),
                    top_n=top_n,
                )
            )

            observation_count += len(
                score_batch
            )

            score_sum += float(
                np.sum(
                    probabilities
                )
            )

            score_min = min(
                score_min,
                float(
                    np.min(
                        probabilities
                    )
                ),
            )

            score_max = max(
                score_max,
                float(
                    np.max(
                        probabilities
                    )
                ),
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

    temporary_path.replace(
        scores_path
    )

    top_candidates = (
        top_candidates.sort_values(
            SCORE_COLUMN,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    score_mean = (
        score_sum
        / observation_count
        if observation_count
        else 0.0
    )

    statistics = {
        "observation_count": (
            observation_count
        ),
        "score_mean": (
            float(
                score_mean
            )
        ),
        "score_min": (
            float(
                score_min
            )
        ),
        "score_max": (
            float(
                score_max
            )
        ),
    }

    return (
        top_candidates,
        statistics,
    )


def create_top_scores_table(
    top_candidates: pd.DataFrame,
    has_target: bool,
    output_path: Path,
) -> pd.DataFrame:
    """Cria a tabela compacta das empresas priorizadas."""

    columns = [
        IDENTIFIER_COLUMN,
        SCORE_COLUMN,
    ]

    if has_target:
        columns.append(
            TARGET_COLUMN
        )

    table = top_candidates[
        columns
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
        output_path,
        index=False,
        encoding="utf-8",
    )

    return table


def create_global_importance_table(
    model: Pipeline,
    features: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Cria a tabela de importância global das variáveis."""

    classifier = model.named_steps[
        "classifier"
    ]

    importances = np.asarray(
        classifier.feature_importances_,
        dtype="float64",
    )

    table = pd.DataFrame(
        {
            "variavel": features,
            "descricao": [
                FEATURE_LABELS.get(
                    feature,
                    feature,
                )
                for feature in features
            ],
            "importancia": importances,
        }
    )

    table[
        "importancia_percentual"
    ] = (
        table["importancia"]
        * 100
    )

    table = (
        table.sort_values(
            "importancia",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return table


def create_local_factors_table(
    top_candidates: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    local_factors: int,
    output_path: Path,
) -> pd.DataFrame:
    """Cria fatores locais aproximados para os maiores scores."""

    feature_data = prepare_features(
        dataframe=top_candidates,
        features=features,
    )

    imputer = model.named_steps[
        "imputer"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    imputed_values = np.asarray(
        imputer.transform(
            feature_data
        ),
        dtype="float64",
    )

    reference_values = np.asarray(
        imputer.statistics_,
        dtype="float64",
    )

    importances = np.asarray(
        classifier.feature_importances_,
        dtype="float64",
    )

    reference_scale = np.maximum(
        np.abs(
            reference_values
        ),
        1.0,
    )

    relative_deviation = (
        imputed_values
        - reference_values
    ) / reference_scale

    approximate_contribution = (
        relative_deviation
        * importances
    )

    factor_count = min(
        local_factors,
        len(features),
    )

    records: list[
        dict[str, Any]
    ] = []

    for row_index in range(
        len(top_candidates)
    ):
        row_contributions = (
            approximate_contribution[
                row_index
            ]
        )

        selected_indices = np.argsort(
            np.abs(
                row_contributions
            )
        )[::-1][
            :factor_count
        ]

        for position, feature_index in enumerate(
            selected_indices,
            start=1,
        ):
            contribution = float(
                row_contributions[
                    feature_index
                ]
            )

            feature_name = features[
                feature_index
            ]

            records.append(
                {
                    IDENTIFIER_COLUMN: str(
                        top_candidates.iloc[
                            row_index
                        ][
                            IDENTIFIER_COLUMN
                        ]
                    ),
                    SCORE_COLUMN: float(
                        top_candidates.iloc[
                            row_index
                        ][
                            SCORE_COLUMN
                        ]
                    ),
                    "posicao_fator": (
                        position
                    ),
                    "variavel": (
                        feature_name
                    ),
                    "descricao": (
                        FEATURE_LABELS.get(
                            feature_name,
                            feature_name,
                        )
                    ),
                    "valor_observado": float(
                        imputed_values[
                            row_index,
                            feature_index,
                        ]
                    ),
                    "valor_referencia": float(
                        reference_values[
                            feature_index
                        ]
                    ),
                    "importancia_global": float(
                        importances[
                            feature_index
                        ]
                    ),
                    "contribuicao_aproximada": (
                        contribution
                    ),
                    "direcao_aproximada": (
                        "acima da referência"
                        if contribution > 0
                        else (
                            "abaixo da referência"
                            if contribution < 0
                            else "igual à referência"
                        )
                    ),
                }
            )

    table = pd.DataFrame(
        records
    )

    table.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    return table


def create_summary_report(
    data_path: Path,
    model_path: Path,
    model_name: str,
    statistics: dict[str, float | int],
    top_n: int,
    local_factors: int,
    features: list[str],
    output_paths: OutputPaths,
) -> None:
    """Registra um resumo da execução do scoring."""

    content = (
        "Scoring de descontinuidade empresarial\n"
        "=======================================\n\n"
        f"Algoritmo: {model_name}\n"
        f"Dataset: {data_path}\n"
        f"Modelo: {model_path}\n"
        f"Scores completos: {output_paths.scores}\n\n"
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
        "Fatores aproximados por empresa: "
        f"{local_factors}\n"
        "Quantidade de variáveis do modelo: "
        f"{len(features)}\n\n"
        "Interpretação\n"
        "-------------\n"
        "O score representa a probabilidade estimada de a "
        "empresa atender à proxy de descontinuidade definida "
        "no snapshot cadastral. Não representa uma previsão "
        "temporal de encerramento futuro.\n\n"
        "Explicabilidade\n"
        "---------------\n"
        "Os fatores locais são aproximações calculadas pela "
        "distância do valor observado em relação à mediana "
        "utilizada pelo imputador, ponderada pela importância "
        "global da variável. Não constituem uma decomposição "
        "exata da predição individual.\n"
    )

    output_paths.summary.write_text(
        content,
        encoding="utf-8",
    )


def generate_scores(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_RF_MODEL_PATH,
    batch_size: int = 100_000,
    top_n: int = 100,
    local_factors: int = 5,
    force: bool = False,
) -> None:
    """Executa o scoring e gera os relatórios do modelo."""

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
        model_key,
        model_name,
    ) = load_model_payload(
        model_path
    )

    output_paths = build_output_paths(
        model_key
    )

    (
        parquet_file,
        has_target,
    ) = validate_dataset_schema(
        data_path=data_path,
        features=features,
    )

    validate_output_files(
        paths=output_paths,
        force=force,
    )

    print(f"[INFO] Dataset: {data_path}")
    print(f"[INFO] Modelo: {model_path}")
    print(f"[INFO] Algoritmo: {model_name}")
    print(
        "[INFO] Variáveis utilizadas: "
        f"{len(features)}"
    )

    (
        top_candidates,
        statistics,
    ) = score_dataset(
        parquet_file=parquet_file,
        model=model,
        features=features,
        has_target=has_target,
        batch_size=batch_size,
        top_n=top_n,
        scores_path=output_paths.scores,
    )

    create_top_scores_table(
        top_candidates=top_candidates,
        has_target=has_target,
        output_path=output_paths.top_scores,
    )

    create_global_importance_table(
        model=model,
        features=features,
        output_path=(
            output_paths.global_importance
        ),
    )

    create_local_factors_table(
        top_candidates=top_candidates,
        model=model,
        features=features,
        local_factors=local_factors,
        output_path=(
            output_paths.local_factors
        ),
    )

    create_summary_report(
        data_path=data_path,
        model_path=model_path,
        model_name=model_name,
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
        output_paths=output_paths,
    )

    print(
        "[OK] Scores completos: "
        f"{output_paths.scores}"
    )

    print(
        "[OK] Maiores scores: "
        f"{output_paths.top_scores}"
    )

    print(
        "[OK] Importância global: "
        f"{output_paths.global_importance}"
    )

    print(
        "[OK] Fatores locais aproximados: "
        f"{output_paths.local_factors}"
    )

    print(
        "[OK] Resumo: "
        f"{output_paths.summary}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Gera scores e explicações para o modelo "
            "de descontinuidade empresarial."
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
        default=DEFAULT_RF_MODEL_PATH,
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
        default=5,
        help=(
            "Quantidade de fatores aproximados apresentados "
            "para cada empresa."
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