"""Treina modelos para classificação da descontinuidade empresarial.

São avaliados dois algoritmos:

- Árvore de Decisão, priorizando interpretabilidade;
- Random Forest, priorizando estabilidade e desempenho preditivo.

O alvo ``y_descontinuidade`` é definido pela ausência de estabelecimentos
ativos no snapshot cadastral:

- y = 1 quando qtd_ativos = 0;
- y = 0 quando qtd_ativos > 0.

As variáveis ``qtd_ativos``, ``qtd_inativos`` e ``pct_ativos`` são
deliberadamente excluídas porque estão diretamente ligadas à definição
do alvo.

O resultado deve ser interpretado como classificação da condição cadastral
observada, e não como previsão temporal de encerramento futuro.

Entrada
-------
data/processed/model/business_discontinuity_dataset.parquet

Saídas
------
models/business_discontinuity/decision_tree.joblib
models/business_discontinuity/random_forest.joblib

reports/metrics/business_discontinuity_decision_tree_metrics.txt
reports/metrics/business_discontinuity_random_forest_metrics.txt

reports/tables/business_discontinuity_decision_tree_importance.csv
reports/tables/business_discontinuity_random_forest_importance.csv
reports/tables/business_discontinuity_model_comparison.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from pgfn_cnpj.settings import (
    MODEL_DATA_DIR,
    MODELS_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


TARGET_COLUMN = "y_descontinuidade"

DEFAULT_INPUT_PATH = (
    MODEL_DATA_DIR
    / "business_discontinuity_dataset.parquet"
)

MODEL_OUTPUT_DIR = (
    MODELS_DIR
    / "business_discontinuity"
)

COMPARISON_PATH = (
    REPORTS_TABLES_DIR
    / "business_discontinuity_model_comparison.csv"
)

RANDOM_STATE = 42


# Lista explícita das variáveis admitidas no treinamento.
#
# Foram escolhidas variáveis empresariais e fiscais que não participam
# diretamente da definição do alvo.
MODEL_FEATURES = [
    "qtd_estabelecimentos",
    "idade_empresa_anos",
    "pgfn_qtd_inscricoes",
    "log1p_pgfn_divida_total",
    "pgfn_divida_media",
    "pgfn_divida_max",
    "pgfn_qtd_ajuizadas",
    "pgfn_pct_ajuizadas",
    "pgfn_qtd_receitas_distintas",
    "pgfn_qtd_situacoes_distintas",
    "pgfn_pct_irregular",
    "pgfn_pct_beneficio_fiscal",
    "pgfn_pct_negociacao",
    "pgfn_pct_suspenso_judicial",
    "pgfn_pct_garantia",
]


# Variáveis excluídas por estarem ligadas diretamente à definição do alvo.
LEAKAGE_FEATURES = [
    "qtd_ativos",
    "qtd_inativos",
    "pct_ativos",
]


@dataclass(frozen=True)
class ModelSpecification:
    """Agrupa nome, arquivo e configuração de cada algoritmo."""

    key: str
    display_name: str
    model_path: Path
    metrics_path: Path
    importance_path: Path


MODEL_SPECIFICATIONS = {
    "tree": ModelSpecification(
        key="tree",
        display_name="Árvore de Decisão",
        model_path=(
            MODEL_OUTPUT_DIR
            / "decision_tree.joblib"
        ),
        metrics_path=(
            REPORTS_METRICS_DIR
            / "business_discontinuity_decision_tree_metrics.txt"
        ),
        importance_path=(
            REPORTS_TABLES_DIR
            / "business_discontinuity_decision_tree_importance.csv"
        ),
    ),
    "rf": ModelSpecification(
        key="rf",
        display_name="Random Forest",
        model_path=(
            MODEL_OUTPUT_DIR
            / "random_forest.joblib"
        ),
        metrics_path=(
            REPORTS_METRICS_DIR
            / "business_discontinuity_random_forest_metrics.txt"
        ),
        importance_path=(
            REPORTS_TABLES_DIR
            / "business_discontinuity_random_forest_importance.csv"
        ),
    ),
}


def validate_arguments(
    sample_n: int,
    test_size: float,
    top_fraction: float,
    n_estimators: int,
    max_depth: int,
    min_leaf_tree: int,
    min_leaf_rf: int,
) -> None:
    """Valida os parâmetros fornecidos pela linha de comando."""

    if sample_n < 0:
        raise ValueError(
            "sample_n não pode ser negativo."
        )

    if sample_n == 1:
        raise ValueError(
            "sample_n deve ser 0 ou maior que 1."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size deve estar entre 0 e 1."
        )

    if not 0 < top_fraction <= 1:
        raise ValueError(
            "top_fraction deve estar entre 0 e 1."
        )

    if n_estimators < 1:
        raise ValueError(
            "n_estimators deve ser maior ou igual a 1."
        )

    if max_depth < 1:
        raise ValueError(
            "max_depth deve ser maior ou igual a 1."
        )

    if min_leaf_tree < 1:
        raise ValueError(
            "min_leaf_tree deve ser maior ou igual a 1."
        )

    if min_leaf_rf < 1:
        raise ValueError(
            "min_leaf_rf deve ser maior ou igual a 1."
        )


def validate_input_file(
    input_path: Path,
) -> None:
    """Verifica se o dataset de treinamento está disponível."""

    if not input_path.exists():
        raise FileNotFoundError(
            "O dataset de descontinuidade não foi encontrado:\n"
            f"{input_path}\n\n"
            "Execute primeiro:\n"
            "python -m "
            "pgfn_cnpj.modeling.business_discontinuity."
            "build_dataset"
        )


def validate_dataset_schema(
    input_path: Path,
) -> None:
    """Confirma a existência do alvo e dos preditores."""

    parquet_file = pq.ParquetFile(
        input_path
    )

    available_columns = set(
        parquet_file.schema_arrow.names
    )

    required_columns = {
        TARGET_COLUMN,
        *MODEL_FEATURES,
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


def load_dataset(
    input_path: Path,
    sample_n: int,
    random_state: int,
) -> pd.DataFrame:
    """Carrega somente as colunas necessárias ao treinamento."""

    required_columns = [
        TARGET_COLUMN,
        *MODEL_FEATURES,
    ]

    dataframe = pd.read_parquet(
        input_path,
        columns=required_columns,
    )

    if (
        sample_n > 0
        and sample_n < len(dataframe)
    ):
        dataframe, _ = train_test_split(
            dataframe,
            train_size=sample_n,
            random_state=random_state,
            stratify=dataframe[TARGET_COLUMN],
        )

    return dataframe.reset_index(
        drop=True
    )


def prepare_xy(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separa os preditores e o alvo binário."""

    target = (
        dataframe[TARGET_COLUMN]
        .astype("int8")
        .to_numpy()
    )

    unique_classes = np.unique(
        target
    )

    if not np.array_equal(
        unique_classes,
        np.array(
            [0, 1],
            dtype="int8",
        ),
    ):
        raise ValueError(
            "O alvo deve conter as duas classes binárias: "
            "0 e 1."
        )

    features = (
        dataframe[MODEL_FEATURES]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .copy()
    )

    return features, target


def select_model_keys(
    model_type: str,
) -> list[str]:
    """Converte a opção da linha de comando em modelos a executar."""

    if model_type == "both":
        return [
            "tree",
            "rf",
        ]

    if model_type in MODEL_SPECIFICATIONS:
        return [model_type]

    raise ValueError(
        "model_type deve ser tree, rf ou both."
    )


def build_classifier(
    model_key: str,
    n_estimators: int,
    max_depth: int,
    min_leaf_tree: int,
    min_leaf_rf: int,
    class_weight: str | None,
    random_state: int,
) -> DecisionTreeClassifier | RandomForestClassifier:
    """Cria o classificador solicitado."""

    if model_key == "tree":
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_leaf_tree,
            class_weight=class_weight,
            random_state=random_state,
        )

    if model_key == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_leaf_rf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        f"Modelo desconhecido: {model_key}"
    )


def build_pipeline(
    classifier: (
        DecisionTreeClassifier
        | RandomForestClassifier
    ),
) -> Pipeline:
    """Cria o pipeline de imputação e classificação."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_top_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    """Calcula métricas para a fração superior do ranking."""

    observation_count = len(
        y_true
    )

    cutoff = max(
        1,
        int(
            np.ceil(
                observation_count
                * fraction
            )
        ),
    )

    ranking = np.argsort(
        -y_score
    )

    selected_indices = ranking[
        :cutoff
    ]

    selected_targets = y_true[
        selected_indices
    ]

    base_rate = float(
        np.mean(y_true)
    )

    selected_rate = float(
        np.mean(selected_targets)
    )

    positive_count = int(
        np.sum(y_true)
    )

    selected_positive_count = int(
        np.sum(selected_targets)
    )

    lift = (
        selected_rate / base_rate
        if base_rate > 0
        else np.nan
    )

    recall = (
        selected_positive_count
        / positive_count
        if positive_count > 0
        else 0.0
    )

    return {
        "fraction": fraction,
        "cutoff": cutoff,
        "selected_positives": (
            selected_positive_count
        ),
        "lift": float(lift),
        "precision": selected_rate,
        "recall": float(recall),
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_fraction: float,
) -> dict[str, Any]:
    """Calcula métricas discriminatórias e de priorização."""

    top_metrics = calculate_top_metrics(
        y_true=y_true,
        y_score=y_score,
        fraction=top_fraction,
    )

    return {
        "test_observations": int(
            len(y_true)
        ),
        "test_positives": int(
            np.sum(y_true)
        ),
        "test_prevalence": float(
            np.mean(y_true)
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_score,
            )
        ),
        "top": top_metrics,
    }


def create_importance_table(
    model: Pipeline,
) -> pd.DataFrame:
    """Cria a tabela de importância das variáveis."""

    classifier = model.named_steps[
        "classifier"
    ]

    importances = (
        classifier.feature_importances_
    )

    table = pd.DataFrame(
        {
            "variavel": MODEL_FEATURES,
            "importancia": importances,
        }
    )

    table["importancia_percentual"] = (
        table["importancia"]
        * 100
    )

    return (
        table.sort_values(
            "importancia",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def output_paths_for_models(
    model_keys: list[str],
) -> list[Path]:
    """Retorna todos os arquivos que serão produzidos."""

    paths: list[Path] = []

    for model_key in model_keys:
        specification = (
            MODEL_SPECIFICATIONS[
                model_key
            ]
        )

        paths.extend(
            [
                specification.model_path,
                specification.metrics_path,
                specification.importance_path,
            ]
        )

    if len(model_keys) > 1:
        paths.append(
            COMPARISON_PATH
        )

    return paths


def validate_output_files(
    model_keys: list[str],
    force: bool,
) -> None:
    """Evita substituição acidental dos resultados."""

    output_paths = output_paths_for_models(
        model_keys
    )

    existing_files = [
        path
        for path in output_paths
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


def save_model_payload(
    specification: ModelSpecification,
    model: Pipeline,
    metrics: dict[str, Any],
    input_path: Path,
    test_size: float,
    class_weight: str | None,
    random_state: int,
    parameters: dict[str, Any],
) -> None:
    """Salva o modelo acompanhado de seus metadados."""

    payload = {
        "model": model,
        "model_key": specification.key,
        "model_name": specification.display_name,
        "target": TARGET_COLUMN,
        "features": MODEL_FEATURES,
        "excluded_leakage_features": (
            LEAKAGE_FEATURES
        ),
        "input_path": str(
            input_path
        ),
        "test_size": test_size,
        "class_weight": class_weight,
        "random_state": random_state,
        "parameters": parameters,
        "metrics": metrics,
        "interpretation": (
            "Classificação da condição cadastral observada "
            "no snapshot. Não representa previsão temporal "
            "de encerramento futuro."
        ),
    }

    dump(
        payload,
        specification.model_path,
    )


def save_metrics_report(
    specification: ModelSpecification,
    metrics: dict[str, Any],
    input_path: Path,
    dataset_size: int,
    train_size: int,
    test_size_count: int,
    parameters: dict[str, Any],
    class_weight: str | None,
) -> None:
    """Salva um relatório textual das métricas."""

    top_metrics = metrics[
        "top"
    ]

    model_features = "\n".join(
        f"- {feature}"
        for feature in MODEL_FEATURES
    )

    leakage_features = "\n".join(
        f"- {feature}"
        for feature in LEAKAGE_FEATURES
    )

    parameter_lines = "\n".join(
        f"{key}: {value}"
        for key, value in parameters.items()
    )

    top_percentage = (
        top_metrics["fraction"]
        * 100
    )

    content = (
        "Modelo de descontinuidade empresarial\n"
        "======================================\n\n"
        f"Algoritmo: {specification.display_name}\n"
        f"Dataset: {input_path}\n"
        f"Observações utilizadas: {dataset_size:,}\n"
        f"Treino: {train_size:,}\n"
        f"Teste: {test_size_count:,}\n\n"
        "Configuração\n"
        "------------\n"
        f"{parameter_lines}\n"
        f"class_weight: {class_weight}\n\n"
        "Métricas no conjunto de teste\n"
        "-----------------------------\n"
        f"Prevalência: {metrics['test_prevalence']:.4f}\n"
        f"ROC AUC: {metrics['roc_auc']:.4f}\n"
        "Average Precision: "
        f"{metrics['average_precision']:.4f}\n"
        f"Brier Score: {metrics['brier_score']:.4f}\n\n"
        f"Priorização no top {top_percentage:.0f}%\n"
        "------------------------\n"
        "Empresas selecionadas: "
        f"{top_metrics['cutoff']:,}\n"
        "Casos positivos selecionados: "
        f"{top_metrics['selected_positives']:,}\n"
        f"Lift: {top_metrics['lift']:.3f}\n"
        f"Precision: {top_metrics['precision']:.3f}\n"
        f"Recall: {top_metrics['recall']:.3f}\n\n"
        "Variáveis utilizadas\n"
        "--------------------\n"
        f"{model_features}\n\n"
        "Variáveis excluídas por vazamento\n"
        "---------------------------------\n"
        f"{leakage_features}\n\n"
        "Limitação metodológica\n"
        "----------------------\n"
        "O alvo representa a ausência de estabelecimentos "
        "ativos no mesmo snapshot cadastral das variáveis "
        "explicativas. Portanto, o resultado deve ser "
        "interpretado como classificação da condição "
        "cadastral observada, e não como previsão de "
        "encerramento empresarial futuro.\n"
    )

    specification.metrics_path.write_text(
        content,
        encoding="utf-8",
    )


def train_single_model(
    model_key: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    input_path: Path,
    dataset_size: int,
    test_size: float,
    top_fraction: float,
    n_estimators: int,
    max_depth: int,
    min_leaf_tree: int,
    min_leaf_rf: int,
    class_weight: str | None,
    random_state: int,
) -> dict[str, Any]:
    """Treina, avalia e salva um único algoritmo."""

    specification = (
        MODEL_SPECIFICATIONS[
            model_key
        ]
    )

    classifier = build_classifier(
        model_key=model_key,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_leaf_tree=min_leaf_tree,
        min_leaf_rf=min_leaf_rf,
        class_weight=class_weight,
        random_state=random_state,
    )

    model = build_pipeline(
        classifier
    )

    print(
        "[INFO] Treinando: "
        f"{specification.display_name}"
    )

    model.fit(
        x_train,
        y_train,
    )

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    metrics = calculate_metrics(
        y_true=y_test,
        y_score=probabilities,
        top_fraction=top_fraction,
    )

    importance_table = (
        create_importance_table(
            model
        )
    )

    importance_table.to_csv(
        specification.importance_path,
        index=False,
        encoding="utf-8",
    )

    if model_key == "tree":
        parameters = {
            "max_depth": max_depth,
            "min_samples_leaf": (
                min_leaf_tree
            ),
        }
    else:
        parameters = {
            "n_estimators": (
                n_estimators
            ),
            "max_depth": max_depth,
            "min_samples_leaf": (
                min_leaf_rf
            ),
        }

    save_model_payload(
        specification=specification,
        model=model,
        metrics=metrics,
        input_path=input_path,
        test_size=test_size,
        class_weight=class_weight,
        random_state=random_state,
        parameters=parameters,
    )

    save_metrics_report(
        specification=specification,
        metrics=metrics,
        input_path=input_path,
        dataset_size=dataset_size,
        train_size=len(
            x_train
        ),
        test_size_count=len(
            x_test
        ),
        parameters=parameters,
        class_weight=class_weight,
    )

    top_metrics = metrics[
        "top"
    ]

    print(
        "[RESULTADO] "
        f"{specification.display_name}: "
        f"ROC AUC={metrics['roc_auc']:.4f} | "
        "Average Precision="
        f"{metrics['average_precision']:.4f} | "
        f"Lift={top_metrics['lift']:.3f}"
    )

    return {
        "modelo": (
            specification.display_name
        ),
        "roc_auc": metrics[
            "roc_auc"
        ],
        "average_precision": metrics[
            "average_precision"
        ],
        "brier_score": metrics[
            "brier_score"
        ],
        "lift_top": top_metrics[
            "lift"
        ],
        "precision_top": top_metrics[
            "precision"
        ],
        "recall_top": top_metrics[
            "recall"
        ],
    }


def create_comparison_table(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Cria a tabela comparativa dos algoritmos."""

    comparison = pd.DataFrame(
        results
    )

    comparison = (
        comparison.sort_values(
            [
                "average_precision",
                "roc_auc",
            ],
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    comparison.insert(
        0,
        "posicao",
        np.arange(
            1,
            len(comparison) + 1,
        ),
    )

    comparison.to_csv(
        COMPARISON_PATH,
        index=False,
        encoding="utf-8",
    )

    return comparison


def train_models(
    input_path: Path = DEFAULT_INPUT_PATH,
    model_type: str = "both",
    sample_n: int = 0,
    test_size: float = 0.20,
    top_fraction: float = 0.10,
    n_estimators: int = 300,
    max_depth: int = 12,
    min_leaf_tree: int = 200,
    min_leaf_rf: int = 50,
    class_weight: str | None = None,
    random_state: int = RANDOM_STATE,
    force: bool = False,
) -> None:
    """Executa o treinamento dos modelos selecionados."""

    validate_arguments(
        sample_n=sample_n,
        test_size=test_size,
        top_fraction=top_fraction,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_leaf_tree=min_leaf_tree,
        min_leaf_rf=min_leaf_rf,
    )

    model_keys = select_model_keys(
        model_type
    )

    ensure_project_directories()

    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_file(
        input_path
    )

    validate_dataset_schema(
        input_path
    )

    validate_output_files(
        model_keys=model_keys,
        force=force,
    )

    dataframe = load_dataset(
        input_path=input_path,
        sample_n=sample_n,
        random_state=random_state,
    )

    features, target = prepare_xy(
        dataframe
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    print(f"[INFO] Dataset: {input_path}")
    print(
        "[INFO] Observações: "
        f"{len(dataframe):,}"
    )
    print(
        "[INFO] Variáveis utilizadas: "
        f"{len(MODEL_FEATURES)}"
    )
    print(
        "[INFO] Modelos selecionados: "
        f"{', '.join(model_keys)}"
    )

    results: list[
        dict[str, Any]
    ] = []

    for model_key in model_keys:
        result = train_single_model(
            model_key=model_key,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
            input_path=input_path,
            dataset_size=len(
                dataframe
            ),
            test_size=test_size,
            top_fraction=top_fraction,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_leaf_tree=min_leaf_tree,
            min_leaf_rf=min_leaf_rf,
            class_weight=class_weight,
            random_state=random_state,
        )

        results.append(
            result
        )

    if len(results) > 1:
        comparison = (
            create_comparison_table(
                results
            )
        )

        best_model = comparison.iloc[
            0
        ]["modelo"]

        print(
            "[OK] Comparação salva: "
            f"{COMPARISON_PATH}"
        )

        print(
            "[RESULTADO] Melhor modelo por "
            "Average Precision: "
            f"{best_model}"
        )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Treina Árvore de Decisão e Random Forest "
            "para classificação da descontinuidade "
            "empresarial."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Caminho do dataset de treinamento.",
    )

    parser.add_argument(
        "--model",
        dest="model_type",
        choices=[
            "tree",
            "rf",
            "both",
        ],
        default="both",
        help="Modelo a treinar.",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help=(
            "Quantidade de observações utilizadas. "
            "Use 0 para a base completa."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporção destinada ao conjunto de teste.",
    )

    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fração superior usada nas métricas de ranking.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Quantidade de árvores da Random Forest.",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=12,
        help="Profundidade máxima das árvores.",
    )

    parser.add_argument(
        "--min-leaf-tree",
        type=int,
        default=200,
        help=(
            "Quantidade mínima de observações por folha "
            "na Árvore de Decisão."
        ),
    )

    parser.add_argument(
        "--min-leaf-rf",
        type=int,
        default=50,
        help=(
            "Quantidade mínima de observações por folha "
            "na Random Forest."
        ),
    )

    parser.add_argument(
        "--class-weight",
        choices=[
            "none",
            "balanced",
        ],
        default="none",
        help="Estratégia de ponderação das classes.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Semente utilizada na divisão e treinamento.",
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

    class_weight = (
        None
        if arguments.class_weight == "none"
        else "balanced"
    )

    train_models(
        input_path=arguments.input_path,
        model_type=arguments.model_type,
        sample_n=arguments.sample_n,
        test_size=arguments.test_size,
        top_fraction=arguments.top_fraction,
        n_estimators=arguments.n_estimators,
        max_depth=arguments.max_depth,
        min_leaf_tree=arguments.min_leaf_tree,
        min_leaf_rf=arguments.min_leaf_rf,
        class_weight=class_weight,
        random_state=arguments.random_state,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()