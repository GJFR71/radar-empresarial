"""Extrai os arquivos trimestrais da PGFN.

Entradas
--------
data/raw/pgfn/pgfn_nao_previdenciario_ano=YYYY_tri=T.zip

Saídas
------
data/staging/pgfn/ano=YYYY/trimestre=T/
reports/manifests/manifest_extract_pgfn_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pgfn_cnpj.ingestion.archive import (
    UnsafeArchiveError,
    directory_has_files,
    extract_zip_atomic,
)
from pgfn_cnpj.ingestion.http import append_manifest_row
from pgfn_cnpj.settings import (
    PGFN_RAW_DIR,
    PGFN_STAGING_DIR,
    REPORTS_MANIFESTS_DIR,
    ensure_project_directories,
)


ARCHIVE_PATTERN = re.compile(
    r"^pgfn_nao_previdenciario_ano=(\d{4})_tri=([1-4])\.zip$",
    re.IGNORECASE,
)

MANIFEST_HEADERS = [
    "run_date",
    "run_timestamp",
    "ano",
    "trimestre",
    "zip_path",
    "output_dir",
    "status",
    "files_extracted",
    "uncompressed_mb",
    "message",
]


@dataclass(frozen=True)
class PeriodArchive:
    """Representa um arquivo trimestral da PGFN."""

    year: int
    quarter: int
    zip_path: Path


def validate_arguments(
    start_year: int,
    end_year: int,
) -> None:
    """Valida o intervalo de anos solicitado."""

    if start_year < 2000:
        raise ValueError(
            "start_year deve ser maior ou igual a 2000."
        )

    if end_year < start_year:
        raise ValueError(
            "end_year deve ser maior ou igual a start_year."
        )


def list_archives(
    raw_directory: Path,
    start_year: int,
    end_year: int,
) -> list[PeriodArchive]:
    """Localiza e ordena os arquivos trimestrais disponíveis."""

    archives: list[PeriodArchive] = []

    for zip_path in raw_directory.glob(
        "pgfn_nao_previdenciario_ano=*_tri=*.zip"
    ):
        match = ARCHIVE_PATTERN.fullmatch(
            zip_path.name
        )

        if match is None:
            continue

        year = int(match.group(1))
        quarter = int(match.group(2))

        if start_year <= year <= end_year:
            archives.append(
                PeriodArchive(
                    year=year,
                    quarter=quarter,
                    zip_path=zip_path,
                )
            )

    archives.sort(
        key=lambda archive: (
            archive.year,
            archive.quarter,
        )
    )

    return archives


def output_directory_for(
    archive: PeriodArchive,
) -> Path:
    """Define o destino de um período trimestral."""

    return (
        PGFN_STAGING_DIR
        / f"ano={archive.year}"
        / f"trimestre={archive.quarter}"
    )


def manifest_row(
    *,
    run_date: str,
    run_timestamp: str,
    archive: PeriodArchive,
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
        "ano": archive.year,
        "trimestre": archive.quarter,
        "zip_path": str(archive.zip_path),
        "output_dir": str(output_directory),
        "status": status,
        "files_extracted": files_extracted,
        "uncompressed_mb": formatted_size,
        "message": message,
    }


def extract_archive(
    *,
    archive: PeriodArchive,
    manifest_path: Path,
    run_date: str,
    run_timestamp: str,
    force: bool,
) -> str:
    """Extrai um período e registra o resultado."""

    output_directory = output_directory_for(
        archive
    )

    if (
        directory_has_files(output_directory)
        and not force
    ):
        append_manifest_row(
            manifest_path=manifest_path,
            headers=MANIFEST_HEADERS,
            row=manifest_row(
                run_date=run_date,
                run_timestamp=run_timestamp,
                archive=archive,
                output_directory=output_directory,
                status="SKIPPED_EXISTS",
                files_extracted="",
                uncompressed_mb="",
                message=(
                    "O diretório já possuía arquivos; "
                    "extração não executada."
                ),
            ),
        )

        print(
            f"[SKIP] {archive.year} T{archive.quarter}: "
            "destino existente."
        )

        return "SKIPPED_EXISTS"

    print(
        f"[EXTRACT] {archive.year} T{archive.quarter}: "
        f"{archive.zip_path.name}"
    )

    try:
        result = extract_zip_atomic(
            zip_path=archive.zip_path,
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

        append_manifest_row(
            manifest_path=manifest_path,
            headers=MANIFEST_HEADERS,
            row=manifest_row(
                run_date=run_date,
                run_timestamp=run_timestamp,
                archive=archive,
                output_directory=output_directory,
                status="OK",
                files_extracted=result.files_extracted,
                uncompressed_mb=uncompressed_mb,
                message="Extração concluída.",
            ),
        )

        print(
            f"[OK] {archive.year} T{archive.quarter}: "
            f"{result.files_extracted} arquivo(s), "
            f"{uncompressed_mb:.2f} MB."
        )

        return "OK"

    append_manifest_row(
        manifest_path=manifest_path,
        headers=MANIFEST_HEADERS,
        row=manifest_row(
            run_date=run_date,
            run_timestamp=run_timestamp,
            archive=archive,
            output_directory=output_directory,
            status=status,
            files_extracted="",
            uncompressed_mb="",
            message=message,
        ),
    )

    print(
        f"[ERROR] {archive.year} T{archive.quarter}: "
        f"{message}"
    )

    return status


def run_extraction(
    start_year: int = 2024,
    end_year: int = 2025,
    force: bool = False,
) -> None:
    """Executa a extração dos períodos selecionados."""

    validate_arguments(
        start_year=start_year,
        end_year=end_year,
    )

    ensure_project_directories()

    archives = list_archives(
        raw_directory=PGFN_RAW_DIR,
        start_year=start_year,
        end_year=end_year,
    )

    if not archives:
        raise FileNotFoundError(
            "Nenhum arquivo da PGFN foi encontrado para "
            f"o período {start_year}–{end_year} em:\n"
            f"{PGFN_RAW_DIR}\n\n"
            "Execute primeiro:\n"
            "python -m pgfn_cnpj.ingestion.pgfn"
        )

    now = datetime.now()

    run_date = now.strftime("%Y%m%d")
    run_timestamp = now.isoformat(
        timespec="seconds"
    )

    manifest_path = (
        REPORTS_MANIFESTS_DIR
        / f"manifest_extract_pgfn_{run_date}.csv"
    )

    print(f"[INFO] Origem: {PGFN_RAW_DIR}")
    print(f"[INFO] Destino: {PGFN_STAGING_DIR}")
    print(f"[INFO] Período: {start_year}–{end_year}")
    print(f"[INFO] Arquivos encontrados: {len(archives)}")
    print(f"[INFO] Manifesto: {manifest_path}")

    statuses: list[str] = []

    for archive in archives:
        status = extract_archive(
            archive=archive,
            manifest_path=manifest_path,
            run_date=run_date,
            run_timestamp=run_timestamp,
            force=force,
        )

        statuses.append(status)

    completed = statuses.count("OK")
    skipped = statuses.count("SKIPPED_EXISTS")
    failed = len(statuses) - completed - skipped

    print("[DONE] Extração da PGFN concluída.")
    print(
        "[RESUMO] "
        f"Extraídos={completed} | "
        f"Existentes={skipped} | "
        f"Falhas={failed}"
    )


def parse_arguments() -> argparse.Namespace:
    """Define os argumentos disponíveis na linha de comando."""

    parser = argparse.ArgumentParser(
        description=(
            "Extrai os arquivos trimestrais da PGFN."
        )
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2024,
        help="Primeiro ano incluído.",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Último ano incluído.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Substitui diretórios já extraídos.",
    )

    return parser.parse_args()


def main() -> None:
    """Ponto de entrada do módulo."""

    arguments = parse_arguments()

    run_extraction(
        start_year=arguments.start_year,
        end_year=arguments.end_year,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()
