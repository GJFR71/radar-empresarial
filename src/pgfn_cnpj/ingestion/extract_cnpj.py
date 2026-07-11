"""Extrai os arquivos dos Dados Abertos do CNPJ.

Os recursos são definidos em ``config/cnpj_arquivos.csv`` e organizados
em dois grupos:

- empresas;
- estabelecimentos.

Cada arquivo ZIP é extraído em uma subpasta própria, evitando colisões
entre arquivos e permitindo substituição controlada.

Entradas
--------
data/raw/cnpj/Empresas0.zip ... Empresas9.zip
data/raw/cnpj/Estabelecimentos0.zip ... Estabelecimentos9.zip

Saídas
------
data/staging/cnpj/2025-12/empresas/<arquivo_zip>/
data/staging/cnpj/2025-12/estabelecimentos/<arquivo_zip>/
reports/manifests/manifest_extract_cnpj_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import zipfile
from datetime import datetime
from pathlib import Path

from pgfn_cnpj.ingestion.archive import (
    UnsafeArchiveError,
    directory_has_files,
    extract_zip_atomic,
)
from pgfn_cnpj.ingestion.cnpj import (
    DEFAULT_CONFIG_PATH,
    Resource,
    normalize_groups,
    read_resources,
    select_resources,
)
from pgfn_cnpj.ingestion.http import append_manifest_row
from pgfn_cnpj.settings import (
    CNPJ_RAW_DIR,
    CNPJ_STAGING_DIR,
    REPORTS_MANIFESTS_DIR,
    ensure_project_directories,
)


SNAPSHOT = "2025-12"

MANIFEST_HEADERS = [
    "run_date",
    "run_timestamp",
    "snapshot",
    "grupo",
    "nome",
    "zip_path",
    "output_dir",
    "status",
    "files_extracted",
    "uncompressed_mb",
    "message",
]


def output_directory_for(resource: Resource) -> Path:
    """Define o diretório de extração de um recurso."""

    archive_name = Path(resource.name).stem

    return (
        CNPJ_STAGING_DIR
        / SNAPSHOT
        / resource.group
        / archive_name
    )


def build_manifest_row(
    *,
    run_date: str,
    run_timestamp: str,
    resource: Resource,
    zip_path: Path,
    output_directory: Path,
    status: str,
    files_extracted: int | str,
    uncompressed_mb: float | str,
    message: str,
) -> dict[str, object]:
    """Cria uma linha padronizada para o manifesto."""

    formatted_size = (
        f"{uncompressed_mb:.2f}"
        if isinstance(uncompressed_mb, float)
        else uncompressed_mb
    )

    return {
        "run_date": run_date,
        "run_timestamp": run_timestamp,
        "snapshot": SNAPSHOT,
        "grupo": resource.group,
        "nome": resource.name,
        "zip_path": str(zip_path),
        "output_dir": str(output_directory),
        "status": status,
        "files_extracted": files_extracted,
        "uncompressed_mb": formatted_size,
        "message": message,
    }


def register_result(
    *,
    manifest_path: Path,
    run_date: str,
    run_timestamp: str,
    resource: Resource,
    zip_path: Path,
    output_directory: Path,
    status: str,
    files_extracted: int | str = "",
    uncompressed_mb: float | str = "",
    message: str = "",
) -> None:
    """Adiciona o resultado da extração ao manifesto."""

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=build_manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
            zip_path=zip_path,
            output_directory=output_directory,
            status=status,
            files_extracted=files_extracted,
            uncompressed_mb=uncompressed_mb,
            message=message,
        ),
    )


def extract_resource(
    *,
    resource: Resource,
    manifest_path: Path,
    run_date: str,
    run_timestamp: str,
    force: bool,
) -> str:
    """Extrai um recurso e registra o resultado."""

    zip_path = CNPJ_RAW_DIR / resource.name

    output_directory = output_directory_for(
        resource
    )

    if not zip_path.exists():
        message = (
            "Arquivo ZIP não encontrado. "
            "Execute primeiro a ingestão do CNPJ."
        )

        register_result(
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
            zip_path=zip_path,
            output_directory=output_directory,
            status="MISSING_ZIP",
            message=message,
        )

        print(
            f"[MISSING] {resource.name}: "
            "arquivo não encontrado."
        )

        return "MISSING_ZIP"

    if (
        directory_has_files(output_directory)
        and not force
    ):
        register_result(
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
            zip_path=zip_path,
            output_directory=output_directory,
            status="SKIPPED_EXISTS",
            message=(
                "O diretório já possuía arquivos; "
                "extração não executada."
            ),
        )

        print(
            f"[SKIP] {resource.name}: "
            "destino existente."
        )

        return "SKIPPED_EXISTS"

    print(
        f"[EXTRACT] {resource.name} "
        f"({resource.group})"
    )

    try:
        result = extract_zip_atomic(
            zip_path=zip_path,
            destination=output_directory,
            force=force,
        )

    except zipfile.BadZipFile:
        status = "FAILED_BAD_ZIP"
        message = "Arquivo ZIP inválido ou corrompido."

    except UnsafeArchiveError as error:
        status = "FAILED_UNSAFE_ARCHIVE"
        message = str(error)

    except Exception as error:
        status = "FAILED_EXCEPTION"
        message = f"{type(error).__name__}: {error}"

    else:
        uncompressed_mb = (
            result.uncompressed_bytes
            / (1024 * 1024)
        )

        register_result(
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            resource=resource,
            zip_path=zip_path,
            output_directory=output_directory,
            status="OK",
            files_extracted=result.files_extracted,
            uncompressed_mb=uncompressed_mb,
            message="Extração concluída.",
        )

        print(
            f"[OK] {resource.name}: "
            f"{result.files_extracted} arquivo(s), "
            f"{uncompressed_mb:.2f} MB."
        )

        return "OK"

    register_result(
        manifest_path=manifest_path,
        run_date=run_date,
        run_timestamp=run_timestamp,
        resource=resource,
        zip_path=zip_path,
        output_directory=output_directory,
        status=status,
        message=message,
    )

    print(
        f"[ERROR] {resource.name}: "
        f"{message}"
    )

    return status


def run_extraction(
    config_path: Path = DEFAULT_CONFIG_PATH,
    groups_text: str = "all",
    force: bool = False,
    strict: bool = False,
) -> None:
    """Executa a extração dos grupos selecionados."""

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
        / f"manifest_extract_cnpj_{run_date}.csv"
    )

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
        f"[INFO] Origem: {CNPJ_RAW_DIR}"
    )

    print(
        f"[INFO] Destino: {CNPJ_STAGING_DIR / SNAPSHOT}"
    )

    print(
        f"[INFO] Manifesto: {manifest_path}"
    )

    statuses: list[str] = []

    for resource in selected_resources:
        status = extract_resource(
            resource=resource,
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            force=force,
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

    missing = statuses.count(
        "MISSING_ZIP"
    )

    failed = (
        len(statuses)
        - completed
        - skipped
        - missing
    )

    print(
        "[DONE] Extração do CNPJ concluída."
    )

    print(
        "[RESUMO] "
        f"Extraídos={completed} | "
        f"Existentes={skipped} | "
        f"Ausentes={missing} | "
        f"Falhas={failed}"
    )

    if strict and (
        missing > 0
        or failed > 0
    ):
        raise RuntimeError(
            "A extração terminou com arquivos ausentes "
            "ou falhas."
        )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Extrai os arquivos dos Dados Abertos do CNPJ."
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
        "--force",
        action="store_true",
        help="Substitui diretórios já extraídos.",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Encerra com erro quando houver ZIPs "
            "ausentes ou falhas de extração."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_extraction(
        config_path=arguments.config,
        groups_text=arguments.groups,
        force=arguments.force,
        strict=arguments.strict,
    )


if __name__ == "__main__":
    main()