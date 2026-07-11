"""Treina a regressão logística para priorização fiscal.

O alvo ``y_risco_fiscal`` é uma regra operacional construída a partir de:

- existência de inscrição ajuizada; ou
- dívida total igual ou superior ao percentil 90.

Por esse motivo, variáveis diretamente utilizadas nessa definição, ou
derivadas delas, são deliberadamente excluídas do treinamento.

O modelo deve ser interpretado como uma análise exploratória de priorização
fiscal com dados contemporâneos, e não como previsão de um evento futuro.

Entrada
-------
data/processed/model/fiscal_risk_dataset.parquet

Saídas
------
models/fiscal_risk/logistic_model.joblib
reports/metrics/fiscal_risk_logistic_metrics.txt
reports/tables/fiscal_risk_logistic_coefficients.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pgfn_cnpj.settings import (
    MODEL_DATA_DIR,
    MODELS_DIR,
    REPORTS_METRICS_DIR,
    REPORTS_TABLES_DIR,
    ensure_project_directories,
)


TARGET_COLUMN = "y_risco_fiscal"

DEFAULT_INPUT_PATH = (
    MODEL_DATA_DIR
    / "fiscal_risk_dataset.parquet"
)

MODEL_OUTPUT_DIR = MODELS_DIR / "fiscal_risk"

MODEL_PATH = (
    MODEL_OUTPUT_DIR
    / "logistic_model.joblib"
)

METRICS_PATH = (
    REPORTS_METRICS_DIR
    / "fiscal_risk_logistic_metrics.txt"
)

COEFFICIENTS_PATH = (
    REPORTS_TABLES_DIR
    / "fiscal_risk_logistic_coefficients.csv"
)

RANDOM_STATE = 42


# Lista explícita de variáveis admitidas no modelo.
#
# A seleção explícita é mais segura do que utilizar automaticamente todas
# as colunas disponíveis no dataset.
MODEL_FEATURES = [
    "qtd_inscricoes",
    "qtd_periodos",
    "qtd_irregular",
    "qtd_beneficio_fiscal",
    "qtd_negociacao",
    "qtd_suspenso_judicial",
    "qtd_garantia",
    "qtd_estabelecimentos",
    "qtd_ativos",
    "qtd_inativos",
    "idade_empresa_dias",
    "inscricoes_por_periodo",
]


# Variáveis excluídas por participarem diretamente da construção do alvo
# ou por serem derivadas das mesmas informações.
LEAKAGE_FEATURES = [
    "divida_total",
    "divida_media",
    "divida_max",
    "qtd_ajuizado",
    "pct_ajuizado",
    "divida_por_estabelecimento",
    "divida_por_estabelecimento_ativo",
    "proporcao_ajuizada_calculada",
]


def validate_arguments(
    sample_n: int,
    test_size: float,
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
) -> None:
    """Valida os parâmetros fornecidos pela linha de comando."""

    if sample_n < 0:
        raise ValueError("sample_n não pode ser negativo.")

    if sample_n == 1:
        raise ValueError(
            "sample_n deve ser 0 ou maior que 1."
        )

    if not 0 < test_size < 1:
        raise ValueError(
            "test_size deve estar entre 0 e 1."
        )

    if c_value <= 0:
        raise ValueError("C deve ser maior que zero.")

    if not 0 <= l1_ratio <= 1:
        raise ValueError(
            "l1_ratio deve estar entre 0 e 1."
        )

    if max_iter < 1:
        raise ValueError(
            "max_iter deve ser maior ou igual a 1."
        )

    if tolerance <= 0:
        raise ValueError(
            "tol deve ser maior que zero."
        )


def validate_input_file(input_path: Path) -> None:
    """Verifica a existência do dataset de treinamento."""

    if not input_path.exists():
        raise FileNotFoundError(
            "O dataset de priorização fiscal não foi encontrado:\n"
            f"{input_path}\n\n"
            "Execute primeiro:\n"
            "python -m "
            "pgfn_cnpj.modeling.fiscal_risk.build_dataset"
        )


def validate_dataset_schema(input_path: Path) -> None:
    """Confirma se alvo e preditores existem no arquivo Parquet."""

    parquet_file = pq.ParquetFile(input_path)
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
    """Carrega somente as colunas necessárias para o modelo."""

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

    return dataframe.reset_index(drop=True)


def prepare_xy(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Separa os preditores e o alvo."""

    target = (
        dataframe[TARGET_COLUMN]
        .astype("int8")
        .to_numpy()
    )

    unique_classes = np.unique(target)

    if not np.array_equal(
        unique_classes,
        np.array([0, 1], dtype="int8"),
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


def build_model(
    c_value: float,
    l1_ratio: float,
    max_iter: int,
    tolerance: float,
    random_state: int,
    class_weight: str | None,
) -> Pipeline:
    """Cria o pipeline de imputação, padronização e modelagem."""

    classifier = LogisticRegression(
        solver="saga",
        C=c_value,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
        class_weight=class_weight,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_top_k_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = 0.10,
) -> dict[str, float | int]:
    """Calcula métricas de priorização para a fração superior."""

    if not 0 < fraction <= 1:
        raise ValueError(
            "fraction deve estar entre 0 e 1."
        )

    observation_count = len(y_true)

    cutoff = max(
        1,
        int(np.ceil(
            observation_count * fraction
        )),
    )

    ranking = np.argsort(-y_score)
    selected_indices = ranking[:cutoff]

    selected_targets = y_true[selected_indices]

    base_rate = float(np.mean(y_true))
    selected_rate = float(
        np.mean(selected_targets)
    )

    positive_count = int(np.sum(y_true))
    selected_positive_count = int(
        np.sum(selected_targets)
    )

    lift = (
        selected_rate / base_rate
        if base_rate > 0
        else np.nan
    )

    precision = selected_rate

    recall = (
        selected_positive_count / positive_count
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
        "precision": float(precision),
        "recall": float(recall),
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    """Calcula métricas discriminatórias e de priorização."""

    top_10_metrics = calculate_top_k_metrics(
        y_true=y_true,
        y_score=y_score,
        fraction=0.10,
    )

    prevalence = float(np.mean(y_true))

    return {
        "test_observations": int(len(y_true)),
        "test_positives": int(np.sum(y_true)),
        "test_prevalence": prevalence,
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
        "top_10": top_10_metrics,
    }


def create_coefficients_table(
    model: Pipeline,
) -> pd.DataFrame:
    """Cria tabela interpretável dos coeficientes padronizados."""

    classifier = model.named_steps["classifier"]

    coefficients = classifier.coef_[0]

    table = pd.DataFrame(
        {
            "variavel": MODEL_FEATURES,
            "coeficiente_padronizado": coefficients,
        }
    )

    table["valor_absoluto"] = (
        table["coeficiente_padronizado"]
        .abs()
    )

    clipped_coefficients = np.clip(
        table["coeficiente_padronizado"],
        -50,
        50,
    )

    table["razao_de_chances"] = np.exp(
        clipped_coefficients
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

    return (
        table.sort_values(
            "valor_absoluto",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def validate_output_files(force: bool) -> None:
    """Evita substituição acidental dos resultados oficiais."""

    output_files = (
        MODEL_PATH,
        METRICS_PATH,
        COEFFICIENTS_PATH,
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


def save_model(
    model: Pipeline,
    metrics: dict[str, Any],
    input_path: Path,
    test_size: float,
    c_value: float,
    l1_ratio: float,
    class_weight: str | None,
    random_state: int,
) -> None:
    """Salva o pipeline e os metadados da execução."""

    payload = {
        "model": model,
        "target": TARGET_COLUMN,
        "features": MODEL_FEATURES,
        "excluded_leakage_features": (
            LEAKAGE_FEATURES
        ),
        "input_path": str(input_path),
        "test_size": test_size,
        "c_value": c_value,
        "l1_ratio": l1_ratio,
        "class_weight": class_weight,
        "random_state": random_state,
        "metrics": metrics,
        "trained_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "interpretation": (
            "Modelo exploratório de priorização fiscal "
            "com dados contemporâneos."
        ),
    }

    dump(
        payload,
        MODEL_PATH,
    )


def save_metrics_report(
    metrics: dict[str, Any],
    input_path: Path,
    dataset_size: int,
    train_size: int,
    test_size_count: int,
    c_value: float,
    l1_ratio: float,
    class_weight: str | None,
    max_iter: int,
    tolerance: float,
    iterations: int,
) -> None:
    """Salva um relatório textual das métricas."""

    top_10 = metrics["top_10"]

    excluded_features = "\n".join(
        f"- {feature}"
        for feature in LEAKAGE_FEATURES
    )

    model_features = "\n".join(
        f"- {feature}"
        for feature in MODEL_FEATURES
    )

    content = (
        "Modelo de priorização fiscal — Regressão logística\n"
        "==================================================\n\n"
        f"Dataset: {input_path}\n"
        f"Observações utilizadas: {dataset_size:,}\n"
        f"Treino: {train_size:,}\n"
        f"Teste: {test_size_count:,}\n\n"
        "Configuração\n"
        "------------\n"
        "Solver: saga\n"
        f"C: {c_value}\n"
        f"l1_ratio: {l1_ratio}\n"
        f"class_weight: {class_weight}\n"
        f"max_iter: {max_iter}\n"
        f"tol: {tolerance}\n"
        f"iterações realizadas: {iterations}\n\n"
        "Métricas no conjunto de teste\n"
        "-----------------------------\n"
        f"Prevalência: {metrics['test_prevalence']:.4f}\n"
        f"ROC AUC: {metrics['roc_auc']:.4f}\n"
        "Average Precision: "
        f"{metrics['average_precision']:.4f}\n"
        f"Brier Score: {metrics['brier_score']:.4f}\n\n"
        "Priorização no top 10%\n"
        "----------------------\n"
        f"Empresas selecionadas: {top_10['cutoff']:,}\n"
        "Casos positivos selecionados: "
        f"{top_10['selected_positives']:,}\n"
        f"Lift@10%: {top_10['lift']:.3f}\n"
        f"Precision@10%: {top_10['precision']:.3f}\n"
        f"Recall@10%: {top_10['recall']:.3f}\n\n"
        "Variáveis utilizadas\n"
        "--------------------\n"
        f"{model_features}\n\n"
        "Variáveis excluídas por risco de vazamento\n"
        "------------------------------------------\n"
        f"{excluded_features}\n\n"
        "Limitação metodológica\n"
        "----------------------\n"
        "O alvo representa uma regra operacional construída "
        "com informações do mesmo período das variáveis "
        "explicativas. Portanto, este resultado não deve ser "
        "interpretado como previsão de inadimplência futura. "
        "O modelo avalia a capacidade de outras características "
        "empresariais discriminarem a regra de priorização fiscal.\n"
    )

    METRICS_PATH.write_text(
        content,
        encoding="utf-8",
    )


def train_model(
    input_path: Path = DEFAULT_INPUT_PATH,
    sample_n: int = 0,
    test_size: float = 0.20,
    c_value: float = 1.0,
    l1_ratio: float = 0.50,
    class_weight: str | None = None,
    max_iter: int = 5000,
    tolerance: float = 1e-4,
    random_state: int = RANDOM_STATE,
    force: bool = False,
) -> None:
    """Executa o treinamento e salva os artefatos oficiais."""

    validate_arguments(
        sample_n=sample_n,
        test_size=test_size,
        c_value=c_value,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
    )

    ensure_project_directories()
    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_input_file(input_path)
    validate_dataset_schema(input_path)
    validate_output_files(force=force)

    dataframe = load_dataset(
        input_path=input_path,
        sample_n=sample_n,
        random_state=random_state,
    )

    features, target = prepare_xy(dataframe)

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

    model = build_model(
        c_value=c_value,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tolerance=tolerance,
        random_state=random_state,
        class_weight=class_weight,
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

    model.fit(
        x_train,
        y_train,
    )

    probability = model.predict_proba(
        x_test
    )[:, 1]

    metrics = calculate_metrics(
        y_true=y_test,
        y_score=probability,
    )

    coefficients_table = (
        create_coefficients_table(model)
    )

    coefficients_table.to_csv(
        COEFFICIENTS_PATH,
        index=False,
        encoding="utf-8",
    )

    classifier = model.named_steps["classifier"]
    iterations = int(classifier.n_iter_[0])

    save_model(
        model=model,
        metrics=metrics,
        input_path=input_path,
        test_size=test_size,
        c_value=c_value,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        random_state=random_state,
    )

    save_metrics_report(
        metrics=metrics,
        input_path=input_path,
        dataset_size=len(dataframe),
        train_size=len(x_train),
        test_size_count=len(x_test),
        c_value=c_value,
        l1_ratio=l1_ratio,
        class_weight=class_weight,
        max_iter=max_iter,
        tolerance=tolerance,
        iterations=iterations,
    )

    top_10 = metrics["top_10"]

    print(f"[OK] Modelo salvo: {MODEL_PATH}")
    print(f"[OK] Métricas: {METRICS_PATH}")
    print(
        "[OK] Coeficientes: "
        f"{COEFFICIENTS_PATH}"
    )
    print(
        "[RESULTADO] "
        f"ROC AUC={metrics['roc_auc']:.4f} | "
        "Average Precision="
        f"{metrics['average_precision']:.4f} | "
        f"Lift@10%={top_10['lift']:.3f}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos de execução."""

    parser = argparse.ArgumentParser(
        description=(
            "Treina a regressão logística de "
            "priorização fiscal."
        )
    )

    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Caminho do dataset de treinamento.",
    )

    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help=(
            "Quantidade de observações da amostra. "
            "Use 0 para utilizar a base completa."
        ),
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporção destinada ao conjunto de teste.",
    )

    parser.add_argument(
        "--c",
        dest="c_value",
        type=float,
        default=1.0,
        help=(
            "Inverso da intensidade de regularização."
        ),
    )

    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.50,
        help=(
            "0 corresponde a L2; 1 corresponde a L1; "
            "valores intermediários usam Elastic Net."
        ),
    )

    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="Estratégia de ponderação das classes.",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Quantidade máxima de iterações.",
    )

    parser.add_argument(
        "--tol",
        dest="tolerance",
        type=float,
        default=1e-4,
        help="Tolerância para convergência.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Semente utilizada na amostragem e divisão.",
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

    class_weight = (
        None
        if arguments.class_weight == "none"
        else "balanced"
    )

    train_model(
        input_path=arguments.input_path,
        sample_n=arguments.sample_n,
        test_size=arguments.test_size,
        c_value=arguments.c_value,
        l1_ratio=arguments.l1_ratio,
        class_weight=class_weight,
        max_iter=arguments.max_iter,
        tolerance=arguments.tolerance,
        random_state=arguments.random_state,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()