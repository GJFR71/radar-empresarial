"""Interface unificada de linha de comando do projeto PGFN + CNPJ.

A interface encaminha cada comando para o módulo responsável, preservando
os argumentos específicos já definidos em cada etapa do projeto.

Exemplos
--------
pgfn-cnpj --help
pgfn-cnpj ingest-pgfn --help
pgfn-cnpj transform-pgfn --help
pgfn-cnpj train-fiscal-risk --help
pgfn-cnpj query-cnpj --help
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TextIO


PACKAGE_NAME = "pgfn-cnpj"


@dataclass(frozen=True)
class Command:
    """Representa um comando disponível na interface."""

    module: str
    function: str
    description: str
    group: str


COMMANDS: dict[str, Command] = {
    "ingest-pgfn": Command(
        module="pgfn_cnpj.ingestion.pgfn",
        function="main",
        description="Baixa os arquivos trimestrais da PGFN.",
        group="ingestion",
    ),
    "ingest-cnpj": Command(
        module="pgfn_cnpj.ingestion.cnpj",
        function="main",
        description="Baixa os arquivos públicos do CNPJ.",
        group="ingestion",
    ),
    "extract-pgfn": Command(
        module="pgfn_cnpj.ingestion.extract_pgfn",
        function="main",
        description="Extrai os arquivos trimestrais da PGFN.",
        group="ingestion",
    ),
    "extract-cnpj": Command(
        module="pgfn_cnpj.ingestion.extract_cnpj",
        function="main",
        description="Extrai os arquivos públicos do CNPJ.",
        group="ingestion",
    ),
    "transform-pgfn": Command(
        module="pgfn_cnpj.pipeline.transform_pgfn",
        function="main",
        description="Consolida os CSVs da PGFN em Parquet.",
        group="pipeline",
    ),
    "build-empresas-lookup": Command(
        module="pgfn_cnpj.pipeline.build_empresas_lookup",
        function="main",
        description="Constrói o lookup cadastral de empresas.",
        group="pipeline",
    ),
    "build-estabelecimentos-core": Command(
        module="pgfn_cnpj.pipeline.build_estabelecimentos_core",
        function="main",
        description="Constrói o núcleo cadastral de estabelecimentos.",
        group="pipeline",
    ),
    "build-cnpj-core": Command(
        module="pgfn_cnpj.pipeline.build_cnpj_core",
        function="main",
        description="Integra empresas e estabelecimentos.",
        group="pipeline",
    ),
    "validate-pgfn": Command(
        module="pgfn_cnpj.validation.validate_pgfn",
        function="main",
        description="Valida a base consolidada da PGFN.",
        group="validation",
    ),
    "validate-cnpj": Command(
        module="pgfn_cnpj.validation.validate_cnpj",
        function="main",
        description="Valida o núcleo cadastral do CNPJ.",
        group="validation",
    ),
    "build-pgfn-abt": Command(
        module="pgfn_cnpj.pipeline.build_pgfn_abt",
        function="main",
        description="Constrói a ABT fiscal da PGFN.",
        group="pipeline",
    ),
    "build-cnpj-abt": Command(
        module="pgfn_cnpj.pipeline.build_cnpj_abt",
        function="main",
        description="Constrói a ABT cadastral do CNPJ.",
        group="pipeline",
    ),
    "join-abts": Command(
        module="pgfn_cnpj.pipeline.join_pgfn_cnpj",
        function="main",
        description="Integra as ABTs da PGFN e do CNPJ.",
        group="pipeline",
    ),
    "build-fiscal-risk-dataset": Command(
        module=(
            "pgfn_cnpj.modeling.fiscal_risk."
            "build_dataset"
        ),
        function="main",
        description="Prepara o dataset de priorização fiscal.",
        group="modeling",
    ),
    "train-fiscal-risk": Command(
        module=(
            "pgfn_cnpj.modeling.fiscal_risk."
            "train_logistic"
        ),
        function="main",
        description="Treina o modelo de priorização fiscal.",
        group="modeling",
    ),
    "score-fiscal-risk": Command(
        module=(
            "pgfn_cnpj.modeling.fiscal_risk."
            "score_explain"
        ),
        function="main",
        description="Calcula e explica o score fiscal.",
        group="modeling",
    ),
    "build-discontinuity-dataset": Command(
        module=(
            "pgfn_cnpj.modeling.business_discontinuity."
            "build_dataset"
        ),
        function="main",
        description="Prepara o dataset de situação cadastral.",
        group="modeling",
    ),
    "train-discontinuity": Command(
        module=(
            "pgfn_cnpj.modeling.business_discontinuity."
            "train_models"
        ),
        function="main",
        description="Treina os modelos de situação cadastral.",
        group="modeling",
    ),
    "score-discontinuity": Command(
        module=(
            "pgfn_cnpj.modeling.business_discontinuity."
            "score_models"
        ),
        function="main",
        description="Calcula os scores de situação cadastral.",
        group="modeling",
    ),
    "query-cnpj": Command(
        module="pgfn_cnpj.cli.query_cnpj",
        function="main",
        description="Consulta os resultados dos modelos por CNPJ.",
        group="consultation",
    ),
}


GROUP_TITLES = {
    "ingestion": "Ingestão e extração",
    "pipeline": "Construção das bases",
    "validation": "Validação",
    "modeling": "Modelagem",
    "consultation": "Consulta",
}


def get_version() -> str:
    """Retorna a versão instalada do projeto."""

    try:
        return package_version(
            PACKAGE_NAME
        )

    except PackageNotFoundError:
        return "0.1.0"


def print_help(
    stream: TextIO = sys.stdout,
) -> None:
    """Exibe os comandos disponíveis."""

    print(
        "PGFN + CNPJ — pipeline de dados e modelos",
        file=stream,
    )

    print(
        "\nUso:",
        file=stream,
    )

    print(
        "  pgfn-cnpj <comando> [opções]",
        file=stream,
    )

    print(
        "\nAjuda de um comando:",
        file=stream,
    )

    print(
        "  pgfn-cnpj <comando> --help",
        file=stream,
    )

    command_width = max(
        len(name)
        for name in COMMANDS
    )

    for group, title in GROUP_TITLES.items():
        group_commands = [
            (
                name,
                command,
            )
            for name, command in COMMANDS.items()
            if command.group == group
        ]

        if not group_commands:
            continue

        print(
            f"\n{title}:",
            file=stream,
        )

        for name, command in group_commands:
            print(
                f"  {name:<{command_width}}  "
                f"{command.description}",
                file=stream,
            )

    print(
        "\nOpções gerais:",
        file=stream,
    )

    print(
        "  -h, --help     Exibe esta ajuda.",
        file=stream,
    )

    print(
        "  --version      Exibe a versão instalada.",
        file=stream,
    )


def load_command(
    command: Command,
) -> Callable[[], None]:
    """Importa a função de entrada de um comando."""

    module = import_module(
        command.module
    )

    function = getattr(
        module,
        command.function,
        None,
    )

    if function is None:
        raise AttributeError(
            "A função de entrada não foi encontrada: "
            f"{command.module}:{command.function}"
        )

    if not callable(function):
        raise TypeError(
            "O ponto de entrada não é executável: "
            f"{command.module}:{command.function}"
        )

    return function


def run_command(
    command_name: str,
    arguments: Sequence[str],
) -> None:
    """Encaminha a execução para o módulo selecionado."""

    command = COMMANDS[
        command_name
    ]

    function = load_command(
        command
    )

    original_argv = sys.argv.copy()

    sys.argv = [
        f"pgfn-cnpj {command_name}",
        *arguments,
    ]

    try:
        function()

    finally:
        sys.argv = original_argv


def main(
    argv: Sequence[str] | None = None,
) -> None:
    """Executa a interface unificada."""

    arguments = list(
        sys.argv[1:]
        if argv is None
        else argv
    )

    if not arguments:
        print_help()
        return

    first_argument = arguments.pop(
        0
    )

    if first_argument in {
        "-h",
        "--help",
        "help",
    }:
        print_help()
        return

    if first_argument == "--version":
        print(
            get_version()
        )
        return

    if first_argument not in COMMANDS:
        print(
            f"[ERRO] Comando desconhecido: "
            f"{first_argument}",
            file=sys.stderr,
        )

        print_help(
            stream=sys.stderr
        )

        raise SystemExit(
            2
        )

    run_command(
        command_name=first_argument,
        arguments=arguments,
    )


if __name__ == "__main__":
    main()