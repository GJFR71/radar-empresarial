"""Utilitários compartilhados para download de arquivos públicos."""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_CHUNK_SIZE = 1024 * 1024

DEFAULT_STATUS_FORCELIST = (
    429,
    500,
    502,
    503,
    504,
)


def build_http_session(
    total_retries: int = 5,
    backoff_factor: float = 2.0,
    pool_size: int = 10,
) -> requests.Session:
    """Cria uma sessão HTTP com repetição para erros transitórios."""

    if total_retries < 0:
        raise ValueError(
            "total_retries não pode ser negativo."
        )

    if backoff_factor < 0:
        raise ValueError(
            "backoff_factor não pode ser negativo."
        )

    if pool_size < 1:
        raise ValueError(
            "pool_size deve ser maior ou igual a 1."
        )

    retries = Retry(
        total=total_retries,
        connect=total_retries,
        read=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=DEFAULT_STATUS_FORCELIST,
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_connections=pool_size,
        pool_maxsize=pool_size,
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.mount(
        "http://",
        adapter,
    )

    return session


def sha256_file(
    path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Calcula o hash SHA-256 de um arquivo."""

    if chunk_size < 1:
        raise ValueError(
            "chunk_size deve ser maior ou igual a 1."
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(chunk_size),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def file_metadata(
    path: Path,
) -> tuple[float, str]:
    """Retorna tamanho em megabytes e hash abreviado."""

    size_mb = (
        path.stat().st_size
        / (1024 * 1024)
    )

    hash_16 = sha256_file(path)[:16]

    return size_mb, hash_16


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: int = 300,
    attempts: int = 3,
    user_agent: str = "Mozilla/5.0 (PGFN-CNPJ Portfolio)",
) -> tuple[int, str]:
    """Baixa um arquivo por streaming e grava de forma atômica.

    Retorna
    -------
    tuple[int, str]
        Código HTTP e URL final após redirecionamentos.
    """

    if timeout < 1:
        raise ValueError(
            "timeout deve ser maior ou igual a 1."
        )

    if attempts < 1:
        raise ValueError(
            "attempts deve ser maior ou igual a 1."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    headers = {
        "User-Agent": user_agent,
    }

    for attempt in range(
        1,
        attempts + 1,
    ):
        try:
            if temporary_path.exists():
                temporary_path.unlink()

            with session.get(
                url,
                stream=True,
                timeout=timeout,
                headers=headers,
                allow_redirects=True,
            ) as response:
                status_code = response.status_code
                final_url = str(response.url)

                if status_code != 200:
                    return (
                        status_code,
                        final_url,
                    )

                with temporary_path.open(
                    "wb"
                ) as output_file:
                    for chunk in response.iter_content(
                        chunk_size=DEFAULT_CHUNK_SIZE
                    ):
                        if chunk:
                            output_file.write(
                                chunk
                            )

            temporary_path.replace(
                destination
            )

            return (
                200,
                final_url,
            )

        except requests.RequestException:
            if temporary_path.exists():
                temporary_path.unlink()

            if attempt == attempts:
                raise

            time.sleep(
                5 * attempt
            )

    return 0, url


def append_manifest_row(
    manifest_path: Path,
    headers: list[str],
    row: dict[str, Any],
) -> None:
    """Adiciona uma linha a um manifesto CSV."""

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_file = not manifest_path.exists()

    normalized_row = {
        header: row.get(
            header,
            "",
        )
        for header in headers
    }

    with manifest_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
        )

        if new_file:
            writer.writeheader()

        writer.writerow(
            normalized_row
        )
