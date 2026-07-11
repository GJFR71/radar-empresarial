"""Utilitários seguros para extração de arquivos ZIP."""

from __future__ import annotations

import shutil
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class UnsafeArchiveError(ValueError):
    """Indica que o ZIP possui um caminho ou membro inseguro."""


@dataclass(frozen=True)
class ExtractionResult:
    """Resume uma extração concluída."""

    files_extracted: int
    uncompressed_bytes: int


def directory_has_files(directory: Path) -> bool:
    """Verifica se um diretório contém pelo menos um arquivo."""

    return (
        directory.exists()
        and any(path.is_file() for path in directory.rglob("*"))
    )


def normalize_member_path(member: zipfile.ZipInfo) -> Path | None:
    """Valida e normaliza o caminho de um membro do ZIP."""

    member_path = PurePosixPath(member.filename)

    if member_path.is_absolute():
        raise UnsafeArchiveError(
            f"Caminho absoluto encontrado no ZIP: {member.filename}"
        )

    if ".." in member_path.parts:
        raise UnsafeArchiveError(
            f"Tentativa de saída do diretório: {member.filename}"
        )

    unix_mode = member.external_attr >> 16

    if stat.S_ISLNK(unix_mode):
        raise UnsafeArchiveError(
            f"Link simbólico não permitido: {member.filename}"
        )

    clean_parts = [
        part
        for part in member_path.parts
        if part not in {"", "."}
    ]

    if not clean_parts:
        return None

    return Path(*clean_parts)


def validate_target_path(
    extraction_root: Path,
    relative_path: Path,
) -> Path:
    """Confirma que o destino permanece dentro da pasta de extração."""

    root = extraction_root.resolve()
    target = (extraction_root / relative_path).resolve()

    if target != root and root not in target.parents:
        raise UnsafeArchiveError(
            f"Destino inseguro detectado: {relative_path}"
        )

    return target


def extract_zip_atomic(
    zip_path: Path,
    destination: Path,
    force: bool = False,
) -> ExtractionResult:
    """Extrai um ZIP de forma segura e com substituição controlada.

    A extração ocorre primeiro em um diretório temporário. O destino
    definitivo somente é alterado após a conclusão de todos os arquivos.
    """

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Arquivo ZIP não encontrado: {zip_path}"
        )

    if (
        destination.exists()
        and directory_has_files(destination)
        and not force
    ):
        raise FileExistsError(
            f"O diretório de destino já possui arquivos: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_directory = (
        destination.parent
        / f".{destination.name}.extracting"
    )

    if temporary_directory.exists():
        shutil.rmtree(temporary_directory)

    temporary_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_extracted = 0
    uncompressed_bytes = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                relative_path = normalize_member_path(member)

                if relative_path is None:
                    continue

                target_path = validate_target_path(
                    extraction_root=temporary_directory,
                    relative_path=relative_path,
                )

                if member.is_dir():
                    target_path.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                target_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                with (
                    archive.open(member, "r") as source,
                    target_path.open("wb") as destination_file,
                ):
                    shutil.copyfileobj(
                        source,
                        destination_file,
                        length=1024 * 1024,
                    )

                files_extracted += 1
                uncompressed_bytes += member.file_size

        if destination.exists():
            shutil.rmtree(destination)

        temporary_directory.replace(destination)

    except Exception:
        if temporary_directory.exists():
            shutil.rmtree(temporary_directory)

        raise

    return ExtractionResult(
        files_extracted=files_extracted,
        uncompressed_bytes=uncompressed_bytes,
    )
