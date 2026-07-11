"""Realiza a ingestão dos arquivos públicos da PGFN.

As fontes são definidas em ``config/pgfn_periodos.csv``. Por padrão, o
módulo baixa somente os períodos de 2024 e 2025 utilizados no projeto.

Esta etapa realiza apenas o download. A extração e a transformação ficam
sob responsabilidade dos módulos posteriores do pipeline.

Saídas
------
data/raw/pgfn/
reports/manifests/manifest_pgfn_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests

from pgfn_cnpj.ingestion.http import (
    append_manifest_row,
    build_http_session,
    download_file,
    file_metadata,
)
from pgfn_cnpj.settings import (
    CONFIG_DIR,
    PGFN_RAW_DIR,
    REPORTS_MANIFESTS_DIR,
    ensure_project_directories,
)


DEFAULT_CONFIG_PATH = CONFIG_DIR / "pgfn_periodos.csv"

DEFAULT_START_YEAR = 2024
DEFAULT_END_YEAR = 2025

DICTIONARY_URL = (
    "https://www.gov.br/pgfn/pt-br/assuntos/"
    "divida-ativa-da-uniao/transparencia-fiscal-1/"
    "arquivos-dados-abertos/dicionario_de_campos.xlsx"
)

DATASET_NAME = "pgfn_nao_previdenciario"

MANIFEST_HEADERS = [
    "run_date",
    "run_timestamp",
    "dataset",
    "ano",
    "trimestre",
    "url",
    "final_url",
    "output_path",
    "status",
    "http_status",
    "size_mb",
    "sha256_16",
    "message",
]


@dataclass(frozen=True)
class PeriodSource:
    """Representa um período disponível para download."""

    year: int
    quarter: int
    url: str


def validate_arguments(
    start_year: int,
    end_year: int,
    timeout: int,
) -> None:
    """Valida os parâmetros da execução."""

    if start_year < 2000:
        raise ValueError(
            "start_year deve ser maior ou igual a 2000."
        )

    if end_year < start_year:
        raise ValueError(
            "end_year deve ser maior ou igual a start_year."
        )

    if timeout < 1:
        raise ValueError(
            "timeout deve ser maior ou igual a 1."
        )


def read_periods(
    config_path: Path,
    start_year: int,
    end_year: int,
) -> list[PeriodSource]:
    """Lê e valida os períodos cadastrados no arquivo CSV."""

    if not config_path.exists():
        raise FileNotFoundError(
            "Arquivo de configuração não encontrado:\n"
            f"{config_path}"
        )

    periods: list[PeriodSource] = []
    observed_keys: set[tuple[int, int]] = set()

    with config_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "ano",
            "trimestre",
            "url_zip",
        }

        available_columns = set(
            reader.fieldnames or []
        )

        missing_columns = sorted(
            required_columns - available_columns
        )

        if missing_columns:
            formatted_columns = ", ".join(
                missing_columns
            )

            raise ValueError(
                "O arquivo de configuração não possui "
                "as colunas necessárias: "
                f"{formatted_columns}."
            )

        for line_number, row in enumerate(
            reader,
            start=2,
        ):
            year_text = (
                row.get("ano") or ""
            ).strip()

            quarter_text = (
                row.get("trimestre") or ""
            ).strip()

            url = (
                row.get("url_zip") or ""
            ).strip()

            if not (
                year_text
                and quarter_text
                and url
            ):
                continue

            try:
                year = int(year_text)
                quarter = int(quarter_text)

            except ValueError as error:
                raise ValueError(
                    "Ano ou trimestre inválido na linha "
                    f"{line_number}."
                ) from error

            if quarter not in {
                1,
                2,
                3,
                4,
            }:
                raise ValueError(
                    "Trimestre inválido na linha "
                    f"{line_number}: {quarter}."
                )

            key = (
                year,
                quarter,
            )

            if key in observed_keys:
                raise ValueError(
                    "Período duplicado no arquivo de configuração: "
                    f"{year} T{quarter}."
                )

            observed_keys.add(
                key
            )

            if (
                start_year
                <= year
                <= end_year
            ):
                periods.append(
                    PeriodSource(
                        year=year,
                        quarter=quarter,
                        url=url,
                    )
                )

    periods.sort(
        key=lambda period: (
            period.year,
            period.quarter,
        )
    )

    if not periods:
        raise ValueError(
            "Nenhum período foi encontrado para o intervalo "
            f"{start_year}–{end_year}."
        )

    return periods


def build_manifest_row(
    *,
    run_date: str,
    run_timestamp: str,
    dataset: str,
    year: int | str,
    quarter: int | str,
    url: str,
    final_url: str,
    output_path: Path | str,
    status: str,
    http_status: int | str,
    size_mb: float | str,
    sha256_16: str,
    message: str,
) -> dict[str, object]:
    """Cria uma linha padronizada para o manifesto."""

    formatted_size = (
        f"{size_mb:.2f}"
        if isinstance(
            size_mb,
            float,
        )
        else size_mb
    )

    return {
        "run_date": run_date,
        "run_timestamp": run_timestamp,
        "dataset": dataset,
        "ano": year,
        "trimestre": quarter,
        "url": url,
        "final_url": final_url,
        "output_path": str(output_path),
        "status": status,
        "http_status": http_status,
        "size_mb": formatted_size,
        "sha256_16": sha256_16,
        "message": message,
    }


def register_existing_file(
    *,
    manifest_path: Path,
    output_path: Path,
    dataset: str,
    year: int | str,
    quarter: int | str,
    url: str,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Registra um arquivo que já estava disponível localmente."""

    size_mb, hash_16 = file_metadata(
        output_path
    )

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            dataset=dataset,
            year=year,
            quarter=quarter,
            url=url,
            final_url=url,
            output_path=output_path,
            status="SKIPPED_EXISTS",
            http_status=200,
            size_mb=size_mb,
            sha256_16=hash_16,
            message=(
                "Arquivo já existente; "
                "download não executado."
            ),
        ),
    )

    return "SKIPPED_EXISTS"


def register_failed_download(
    *,
    manifest_path: Path,
    dataset: str,
    year: int | str,
    quarter: int | str,
    url: str,
    final_url: str,
    status: str,
    http_status: int | str,
    message: str,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Registra uma tentativa de download sem sucesso."""

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            dataset=dataset,
            year=year,
            quarter=quarter,
            url=url,
            final_url=final_url,
            output_path="",
            status=status,
            http_status=http_status,
            size_mb="",
            sha256_16="",
            message=message,
        ),
    )

    return status


def download_period(
    *,
    period: PeriodSource,
    session: requests.Session,
    manifest_path: Path,
    timeout: int,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Baixa um arquivo trimestral da PGFN."""

    filename = (
        f"{DATASET_NAME}_"
        f"ano={period.year}_"
        f"tri={period.quarter}.zip"
    )

    output_path = (
        PGFN_RAW_DIR
        / filename
    )

    if output_path.exists():
        print(
            "[SKIP] "
            f"{period.year} T{period.quarter}: "
            "arquivo existente."
        )

        return register_existing_file(
            manifest_path=manifest_path,
            output_path=output_path,
            dataset=DATASET_NAME,
            year=period.year,
            quarter=period.quarter,
            url=period.url,
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    print(
        "[DOWN] "
        f"{period.year} T{period.quarter}"
    )

    try:
        http_status, final_url = download_file(
            session=session,
            url=period.url,
            destination=output_path,
            timeout=timeout,
            attempts=3,
            user_agent=(
                "Mozilla/5.0 "
                "(PGFN-CNPJ Portfolio)"
            ),
        )

    except requests.RequestException as error:
        print(
            "[ERROR] "
            f"{period.year} T{period.quarter}: "
            f"{type(error).__name__}"
        )

        return register_failed_download(
            manifest_path=manifest_path,
            dataset=DATASET_NAME,
            year=period.year,
            quarter=period.quarter,
            url=period.url,
            final_url="",
            status="FAILED_REQUEST",
            http_status="",
            message=(
                "Erro de rede ou timeout: "
                f"{type(error).__name__}."
            ),
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    if http_status != 200:
        print(
            "[WARN] "
            f"{period.year} T{period.quarter}: "
            f"HTTP {http_status}."
        )

        return register_failed_download(
            manifest_path=manifest_path,
            dataset=DATASET_NAME,
            year=period.year,
            quarter=period.quarter,
            url=period.url,
            final_url=final_url,
            status="FAILED_HTTP",
            http_status=http_status,
            message=(
                "Servidor retornou código HTTP "
                "diferente de 200."
            ),
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    size_mb, hash_16 = file_metadata(
        output_path
    )

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            dataset=DATASET_NAME,
            year=period.year,
            quarter=period.quarter,
            url=period.url,
            final_url=final_url,
            output_path=output_path,
            status="OK",
            http_status=200,
            size_mb=size_mb,
            sha256_16=hash_16,
            message="Download concluído.",
        ),
    )

    print(
        "[OK] "
        f"{period.year} T{period.quarter}: "
        f"{size_mb:.2f} MB."
    )

    return "OK"


def download_dictionary(
    *,
    session: requests.Session,
    manifest_path: Path,
    timeout: int,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Baixa o dicionário oficial de campos da PGFN."""

    dataset = "pgfn_dicionario_campos"

    output_path = (
        PGFN_RAW_DIR
        / "pgfn_dicionario_campos.xlsx"
    )

    if output_path.exists():
        print(
            "[SKIP] Dicionário já existente."
        )

        return register_existing_file(
            manifest_path=manifest_path,
            output_path=output_path,
            dataset=dataset,
            year="",
            quarter="",
            url=DICTIONARY_URL,
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    print(
        "[DOWN] Dicionário de campos."
    )

    try:
        http_status, final_url = download_file(
            session=session,
            url=DICTIONARY_URL,
            destination=output_path,
            timeout=min(
                timeout,
                300,
            ),
            attempts=3,
            user_agent=(
                "Mozilla/5.0 "
                "(PGFN-CNPJ Portfolio)"
            ),
        )

    except requests.RequestException as error:
        print(
            "[ERROR] Dicionário: "
            f"{type(error).__name__}"
        )

        return register_failed_download(
            manifest_path=manifest_path,
            dataset=dataset,
            year="",
            quarter="",
            url=DICTIONARY_URL,
            final_url="",
            status="FAILED_REQUEST",
            http_status="",
            message=(
                "Erro de rede ou timeout: "
                f"{type(error).__name__}."
            ),
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    if http_status != 200:
        print(
            "[WARN] Dicionário: "
            f"HTTP {http_status}."
        )

        return register_failed_download(
            manifest_path=manifest_path,
            dataset=dataset,
            year="",
            quarter="",
            url=DICTIONARY_URL,
            final_url=final_url,
            status="FAILED_HTTP",
            http_status=http_status,
            message=(
                "Servidor retornou código HTTP "
                "diferente de 200."
            ),
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    size_mb, hash_16 = file_metadata(
        output_path
    )

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            dataset=dataset,
            year="",
            quarter="",
            url=DICTIONARY_URL,
            final_url=final_url,
            output_path=output_path,
            status="OK",
            http_status=200,
            size_mb=size_mb,
            sha256_16=hash_16,
            message=(
                "Download do dicionário concluído."
            ),
        ),
    )

    print(
        "[OK] Dicionário: "
        f"{size_mb:.2f} MB."
    )

    return "OK"


def run_ingestion(
    config_path: Path = DEFAULT_CONFIG_PATH,
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
    timeout: int = 300,
    include_dictionary: bool = True,
) -> None:
    """Executa a ingestão dos períodos selecionados."""

    validate_arguments(
        start_year=start_year,
        end_year=end_year,
        timeout=timeout,
    )

    ensure_project_directories()

    periods = read_periods(
        config_path=config_path,
        start_year=start_year,
        end_year=end_year,
    )

    now = datetime.now()

    run_date = now.strftime(
        "%Y%m%d"
    )

    run_timestamp = now.isoformat(
        timespec="seconds"
    )

    manifest_path = (
        REPORTS_MANIFESTS_DIR
        / f"manifest_pgfn_{run_date}.csv"
    )

    session = build_http_session(
        total_retries=5,
        backoff_factor=2.0,
        pool_size=10,
    )

    statuses: list[str] = []

    print(
        f"[INFO] Configuração: {config_path}"
    )

    print(
        f"[INFO] Período: {start_year}–{end_year}"
    )

    print(
        f"[INFO] Arquivos selecionados: {len(periods)}"
    )

    print(
        f"[INFO] Diretório: {PGFN_RAW_DIR}"
    )

    print(
        f"[INFO] Manifesto: {manifest_path}"
    )

    if include_dictionary:
        dictionary_status = download_dictionary(
            session=session,
            manifest_path=manifest_path,
            timeout=timeout,
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

        statuses.append(
            dictionary_status
        )

    for period in periods:
        status = download_period(
            period=period,
            session=session,
            manifest_path=manifest_path,
            timeout=timeout,
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

        statuses.append(
            status
        )

    session.close()

    completed = statuses.count(
        "OK"
    )

    skipped = statuses.count(
        "SKIPPED_EXISTS"
    )

    failed = len(statuses) - completed - skipped

    print(
        "[DONE] Ingestão da PGFN concluída."
    )

    print(
        "[RESUMO] "
        f"Baixados={completed} | "
        f"Existentes={skipped} | "
        f"Falhas={failed}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Baixa os arquivos públicos da PGFN "
            "definidos no arquivo de configuração."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Caminho do arquivo pgfn_periodos.csv."
        ),
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
        help="Primeiro ano incluído.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
        help="Último ano incluído.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help=(
            "Tempo máximo por requisição, em segundos."
        ),
    )

    parser.add_argument(
        "--skip-dictionary",
        action="store_true",
        help=(
            "Não baixa o dicionário de campos."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_ingestion(
        config_path=arguments.config,
        start_year=arguments.start_year,
        end_year=arguments.end_year,
        timeout=arguments.timeout,
        include_dictionary=(
            not arguments.skip_dictionary
        ),
    )


if __name__ == "__main__":
    main()