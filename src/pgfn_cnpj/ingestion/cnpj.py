"""Realiza a ingestão dos Dados Abertos do CNPJ.

As fontes são definidas em ``config/cnpj_arquivos.csv``. O arquivo deve
conter as colunas:

- grupo;
- nome;
- url.

Os grupos utilizados neste projeto são:

- empresas;
- estabelecimentos.

Esta etapa apenas baixa os arquivos compactados. Extração, transformação e
construção das tabelas ficam sob responsabilidade dos módulos posteriores.

Saídas
------
data/raw/cnpj/
reports/manifests/manifest_cnpj_YYYYMMDD.csv
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
    CNPJ_RAW_DIR,
    CONFIG_DIR,
    REPORTS_MANIFESTS_DIR,
    ensure_project_directories,
)


DEFAULT_CONFIG_PATH = CONFIG_DIR / "cnpj_arquivos.csv"

SNAPSHOT = "2025-12"

DATASET_NAME = "cnpj_dados_abertos"

ALLOWED_GROUPS = {
    "empresas",
    "estabelecimentos",
}

MANIFEST_HEADERS = [
    "run_date",
    "run_timestamp",
    "dataset",
    "snapshot",
    "grupo",
    "nome",
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
class Resource:
    """Representa um arquivo público disponível para download."""

    group: str
    name: str
    url: str


def validate_arguments(
    timeout: int,
) -> None:
    """Valida os parâmetros de execução."""

    if timeout < 1:
        raise ValueError(
            "timeout deve ser maior ou igual a 1."
        )


def normalize_groups(
    groups_text: str,
) -> set[str]:
    """Converte a opção textual em um conjunto de grupos."""

    groups = {
        value.strip().lower()
        for value in groups_text.split(",")
        if value.strip()
    }

    if not groups:
        raise ValueError(
            "Informe pelo menos um grupo."
        )

    if "all" in groups:
        return set(ALLOWED_GROUPS)

    invalid_groups = sorted(
        groups - ALLOWED_GROUPS
    )

    if invalid_groups:
        formatted_groups = ", ".join(
            invalid_groups
        )

        raise ValueError(
            "Grupos desconhecidos: "
            f"{formatted_groups}. "
            "Use empresas, estabelecimentos ou all."
        )

    return groups


def read_resources(
    config_path: Path,
) -> list[Resource]:
    """Lê e valida os recursos cadastrados no arquivo CSV."""

    if not config_path.exists():
        raise FileNotFoundError(
            "Arquivo de configuração não encontrado:\n"
            f"{config_path}"
        )

    resources: list[Resource] = []

    observed_names: set[str] = set()
    observed_urls: set[str] = set()

    with config_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "grupo",
            "nome",
            "url",
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
            group = (
                row.get("grupo") or ""
            ).strip().lower()

            name = (
                row.get("nome") or ""
            ).strip()

            url = (
                row.get("url") or ""
            ).strip()

            if not (
                group
                or name
                or url
            ):
                continue

            if not (
                group
                and name
                and url
            ):
                raise ValueError(
                    "Registro incompleto na linha "
                    f"{line_number}."
                )

            if group not in ALLOWED_GROUPS:
                raise ValueError(
                    "Grupo inválido na linha "
                    f"{line_number}: {group}."
                )

            if not name.lower().endswith(
                ".zip"
            ):
                raise ValueError(
                    "O recurso da linha "
                    f"{line_number} não possui extensão .zip."
                )

            if not url.startswith(
                "https://"
            ):
                raise ValueError(
                    "A URL da linha "
                    f"{line_number} não utiliza HTTPS."
                )

            normalized_name = name.lower()

            if normalized_name in observed_names:
                raise ValueError(
                    "Nome de arquivo duplicado: "
                    f"{name}."
                )

            if url in observed_urls:
                raise ValueError(
                    "URL duplicada no arquivo de configuração: "
                    f"{url}."
                )

            observed_names.add(
                normalized_name
            )

            observed_urls.add(
                url
            )

            resources.append(
                Resource(
                    group=group,
                    name=name,
                    url=url,
                )
            )

    resources.sort(
        key=lambda resource: (
            resource.group,
            resource.name.lower(),
        )
    )

    if not resources:
        raise ValueError(
            "Nenhum recurso foi encontrado no arquivo "
            "de configuração."
        )

    return resources


def select_resources(
    resources: list[Resource],
    groups: set[str],
) -> list[Resource]:
    """Seleciona somente os recursos dos grupos solicitados."""

    selected = [
        resource
        for resource in resources
        if resource.group in groups
    ]

    if not selected:
        formatted_groups = ", ".join(
            sorted(groups)
        )

        raise ValueError(
            "Nenhum recurso encontrado para os grupos: "
            f"{formatted_groups}."
        )

    return selected


def build_manifest_row(
    *,
    run_date: str,
    run_timestamp: str,
    resource: Resource,
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
        "dataset": DATASET_NAME,
        "snapshot": SNAPSHOT,
        "grupo": resource.group,
        "nome": resource.name,
        "url": resource.url,
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
    resource: Resource,
    output_path: Path,
    manifest_path: Path,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Registra um arquivo que já estava disponível."""

    size_mb, hash_16 = file_metadata(
        output_path
    )

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
            final_url=resource.url,
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
    resource: Resource,
    manifest_path: Path,
    run_date: str,
    run_timestamp: str,
    final_url: str,
    status: str,
    http_status: int | str,
    message: str,
) -> str:
    """Registra uma tentativa de download sem sucesso."""

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
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


def download_resource(
    *,
    resource: Resource,
    session: requests.Session,
    manifest_path: Path,
    timeout: int,
    run_date: str,
    run_timestamp: str,
) -> str:
    """Baixa um recurso individual do CNPJ."""

    output_path = (
        CNPJ_RAW_DIR
        / resource.name
    )

    if output_path.exists():
        print(
            f"[SKIP] {resource.name}: "
            "arquivo existente."
        )

        return register_existing_file(
            resource=resource,
            output_path=output_path,
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
        )

    print(
        f"[DOWN] {resource.name} "
        f"({resource.group})"
    )

    try:
        http_status, final_url = download_file(
            session=session,
            url=resource.url,
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
            f"[ERROR] {resource.name}: "
            f"{type(error).__name__}"
        )

        return register_failed_download(
            resource=resource,
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            final_url="",
            status="FAILED_REQUEST",
            http_status="",
            message=(
                "Erro de rede ou timeout: "
                f"{type(error).__name__}."
            ),
        )

    if http_status != 200:
        print(
            f"[WARN] {resource.name}: "
            f"HTTP {http_status}."
        )

        return register_failed_download(
            resource=resource,
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            final_url=final_url,
            status="FAILED_HTTP",
            http_status=http_status,
            message=(
                "Servidor retornou código HTTP "
                "diferente de 200."
            ),
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
            resource=resource,
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
        f"[OK] {resource.name}: "
        f"{size_mb:.2f} MB."
    )

    return "OK"


def run_ingestion(
    config_path: Path = DEFAULT_CONFIG_PATH,
    groups_text: str = "all",
    timeout: int = 900,
) -> None:
    """Executa a ingestão dos grupos selecionados."""

    validate_arguments(
        timeout=timeout
    )

    groups = normalize_groups(
        groups_text
    )

    ensure_project_directories()

    resources = read_resources(
        config_path
    )

    selected_resources = select_resources(
        resources=resources,
        groups=groups,
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
        / f"manifest_cnpj_{run_date}.csv"
    )

    statuses: list[str] = []

    print(
        f"[INFO] Configuração: {config_path}"
    )

    print(
        f"[INFO] Snapshot: {SNAPSHOT}"
    )

    print(
        "[INFO] Grupos: "
        f"{', '.join(sorted(groups))}"
    )

    print(
        "[INFO] Recursos selecionados: "
        f"{len(selected_resources)}"
    )

    print(
        f"[INFO] Diretório: {CNPJ_RAW_DIR}"
    )

    print(
        f"[INFO] Manifesto: {manifest_path}"
    )

    with build_http_session(
        total_retries=5,
        backoff_factor=2.0,
        pool_size=10,
    ) as session:
        for resource in selected_resources:
            status = download_resource(
                resource=resource,
                session=session,
                manifest_path=manifest_path,
                timeout=timeout,
                run_date=run_date,
                run_timestamp=run_timestamp,
            )

            statuses.append(
                status
            )

    completed = statuses.count(
        "OK"
    )

    skipped = statuses.count(
        "SKIPPED_EXISTS"
    )

    failed = (
        len(statuses)
        - completed
        - skipped
    )

    print(
        "[DONE] Ingestão do CNPJ concluída."
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
            "Baixa os arquivos públicos do CNPJ "
            "definidos no arquivo de configuração."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Caminho do arquivo cnpj_arquivos.csv."
        ),
    )

    parser.add_argument(
        "--groups",
        type=str,
        default="all",
        help=(
            "Grupos separados por vírgula. "
            "Opções: empresas, estabelecimentos ou all."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help=(
            "Tempo máximo por requisição, em segundos."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_ingestion(
        config_path=arguments.config,
        groups_text=arguments.groups,
        timeout=arguments.timeout,
    )


if __name__ == "__main__":
    main()