"""``slkit`` command dispatch.

The one job of this module is to be the single place that catches a
:class:`DiagnosticError` and turns it into rendered stderr output plus a non-zero exit
code. Every other layer raises well-formed diagnostics and stays out of the presentation
business. Subcommands grow here phase by phase; for now the scaffold ships ``config``,
which is enough to exercise config resolution and the diagnostic renderer end to end.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from ..config import load_config, require_network_identity
from ..diagnostics import DiagnosticError, render

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slkit",
        description="setlistkit — a setlist prediction toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"slkit {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="path to a config file (overrides SLKIT_CONFIG and the default search order)",
    )

    sub = parser.add_subparsers(dest="command")

    config_cmd = sub.add_parser("config", help="inspect the resolved configuration")
    config_sub = config_cmd.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="print the resolved configuration")
    config_sub.add_parser("check", help="validate the config, including network identity")

    return parser


def _cmd_config_show(config) -> int:
    print(f"config file : {config.source_path}")
    print(f"data_root   : {config.data_root}")
    print(f"user_agent  : {config.user_agent}")
    if config.user_agent_is_sentinel:
        print("            : (placeholder — network operations will refuse to run)")
    return EXIT_OK


def _cmd_config_check(config) -> int:
    require_network_identity(config)
    print(f"ok: {config.source_path} is valid and identifies itself to the network")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises past this boundary."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        if args.command == "config":
            config = load_config(args.config)
            if args.config_action == "check":
                return _cmd_config_check(config)
            # default action for `slkit config` is to show
            return _cmd_config_show(config)
    except DiagnosticError as exc:
        print(render(exc.diagnostic), file=sys.stderr)
        return EXIT_DIAGNOSTIC

    parser.print_help()
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
