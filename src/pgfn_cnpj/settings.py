"""Configurações compartilhadas de caminhos do projeto."""

from pathlib import Path


# Este arquivo está em:
# <raiz_do_projeto>/src/pgfn_cnpj/settings.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_STAGING_DIR = DATA_DIR / "staging"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

PGFN_PROCESSED_DIR = DATA_PROCESSED_DIR / "pgfn"
CNPJ_PROCESSED_DIR = DATA_PROCESSED_DIR / "cnpj"
ABT_PROCESSED_DIR = DATA_PROCESSED_DIR / "abt"
MODEL_DATA_DIR = DATA_PROCESSED_DIR / "model"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_METRICS_DIR = REPORTS_DIR / "metrics"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"
REPORTS_SAMPLES_DIR = REPORTS_DIR / "samples"

TEMP_DIR = PROJECT_ROOT / ".tmp"


def ensure_project_directories() -> None:
    """Cria os diretórios necessários para dados, modelos e relatórios."""

    directories = (
        DATA_RAW_DIR,
        DATA_STAGING_DIR,
        PGFN_PROCESSED_DIR,
        CNPJ_PROCESSED_DIR,
        ABT_PROCESSED_DIR,
        MODEL_DATA_DIR,
        MODELS_DIR,
        REPORTS_FIGURES_DIR,
        REPORTS_METRICS_DIR,
        REPORTS_TABLES_DIR,
        REPORTS_SAMPLES_DIR,
        TEMP_DIR,
    )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
