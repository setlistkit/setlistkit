# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit`` command dispatch.

The one job of this module is to be the single place that catches a
:class:`DiagnosticError` and turns it into rendered stderr output plus a non-zero exit
code. Every other layer raises well-formed diagnostics and stays out of the presentation
business. Subcommands grow here phase by phase; for now the scaffold ships ``config``,
which is enough to exercise config resolution and the diagnostic renderer end to end.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import __version__
from ..catalog.lint import lint
from ..config import load_config, require_network_identity
from ..diagnostics import ERROR, Diagnostic, DiagnosticError, render
from ..sources.archive_org import ArchiveOrgClient
from ..sources.client import SourceError
from ..store import Store
from ..store.raw_cache import RawCache
from .common import min_year, required_setting, resolve_pack_dir
from .ingest import ingest

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2

# How often a long pull says it is still going. One metadata request per item at roughly a
# request a second means a first pull of a large collection runs for a quarter of an hour, and
# silence for that long is indistinguishable from a hang.
_PROGRESS_EVERY = 25

# Sources `slkit pull` knows how to fetch. Named here rather than derived from the config's
# [sources.*] tables: a typo'd table name would otherwise become a source that silently does
# nothing, and argparse can reject an unknown name with a usage line instead.
_SOURCES = ("archive_org",)


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

    store_cmd = sub.add_parser("store", help="create and inspect the state store")
    store_sub = store_cmd.add_subparsers(dest="store_action")
    store_sub.add_parser("init", help="create data_root and apply schema migrations")
    store_sub.add_parser("status", help="show schema version and table counts")

    sub.add_parser("dump", help="print a plain-text view of derived state")

    pull_cmd = sub.add_parser("pull", help="fetch raw source data into the cache")
    pull_cmd.add_argument("source", choices=_SOURCES, help="which source to fetch from")
    pull_cmd.add_argument(
        "--force-rescan", action="store_true",
        help="re-ask for data already cached (still rate-limited, and conditional: an "
             "unchanged item answers 304 and no bytes move)",
    )
    pull_cmd.add_argument(
        "--min-year", type=int, metavar="YEAR",
        help="ignore shows played before YEAR (overrides min_year in config)",
    )

    ingest_cmd = sub.add_parser(
        "ingest", help="parse the cached raw data and publish the merged corpus")
    ingest_cmd.add_argument("source", choices=_SOURCES, nargs="?", default=_SOURCES[0],
                            help="which cached source to parse (default: %(default)s)")
    ingest_cmd.add_argument("--pack", metavar="PATH",
                            help="pack directory to parse with (overrides [catalog] pack)")
    ingest_cmd.add_argument("--min-year", type=int, metavar="YEAR",
                            help="the min_year the cache was pulled under, if not the configured "
                                 "one (a listing is cached per collection and min_year)")
    ingest_cmd.add_argument("--dry-run", action="store_true",
                            help="parse, merge and report, but write nothing to the database")
    ingest_cmd.add_argument(
        "--force", action="store_true",
        help="publish even when the merge produced far fewer shows than are stored (see the "
             "no-shrink guard)",
    )

    pack_cmd = sub.add_parser("pack", help="work with band packs")
    pack_sub = pack_cmd.add_subparsers(dest="pack_action")
    lint_cmd = pack_sub.add_parser("lint", help="validate a pack and run conformance checks")
    lint_cmd.add_argument(
        "--pack", metavar="PATH",
        help="pack directory to lint (overrides [catalog] pack in config)",
    )
    lint_cmd.add_argument(
        "--format", choices=("human", "json"), default="human",
        help="output format (default: human)",
    )

    return parser


def _cmd_config(config, args) -> int:
    """`slkit config [show|check]`, defaulting to show."""
    if args.config_action == "check":
        return _cmd_config_check(config)
    return _cmd_config_show(config)


def _cmd_store(config, args) -> int:
    """`slkit store [init|status]`, defaulting to init."""
    if args.store_action == "status":
        return _cmd_store_status(config)
    return _cmd_store_init(config)


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


def _cmd_store_init(config) -> int:
    with Store(config.data_root) as store:
        applied = store.init()
        if applied:
            print(f"applied migrations: {', '.join(str(v) for v in applied)}")
        else:
            print("already current, nothing to apply")
        print(f"schema version {store.schema_version()} at {store.db_path}")
    return EXIT_OK


def _cmd_store_status(config) -> int:
    store = Store(config.data_root)
    print(f"db          : {store.db_path}")
    print(f"raw cache   : {store.raw.root}")
    # Don't open (and thereby create) the DB just to report it doesn't exist yet.
    if not store.db_path.is_file():
        print("schema      : not initialized — run `slkit store init`")
        return EXIT_OK
    with store:
        print(f"schema      : version {store.schema_version()}")
        for table, count in store.table_counts().items():
            print(f"  {table}: {count} rows")
    return EXIT_OK


def _cmd_dump(config, _args) -> int:
    with Store(config.data_root) as store:
        print(store.dump(), end="")
    return EXIT_OK


def _cmd_pull(config, args) -> int:
    """Fetch a source into the raw cache. Writes nothing to the database.

    The refusal comes first and comes from :func:`require_network_identity`, so a run with the
    placeholder ``user_agent`` stops here rather than at the first request -- there is no state
    to half-write, and the message is the same one ``slkit config check`` gives.

    Every setting is looked up under ``args.source``, never under a hard-coded table name. With
    one source those are the same string; with two, a hard-coded one makes ``slkit pull setlistfm``
    quietly read the archive.org table and pull archive.org.
    """
    require_network_identity(config)
    collection = required_setting(
        config, ("sources", args.source), "collection",
        f"A pull needs to know which {args.source} collection holds this band's tapes.")
    floor = min_year(config, args.source, args.min_year)

    # flush=True on both, because a pull of a whole collection runs for hours and Python buffers
    # stdout whenever it is not a terminal. Redirected to a file or a log, an unflushed progress
    # line arrives after the job it was reporting on has finished, which is not progress.
    def progress(done: int, total: int) -> None:
        if done % _PROGRESS_EVERY == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    def announce(batch_id: str) -> None:
        # Printed so this run is greppable in OUR logs by the same id the source sees in theirs.
        # A batch id only one side of the conversation knows is half a tracking id.
        print(f"batch {batch_id} (sent in the User-Agent of every request this run)", flush=True)

    client = ArchiveOrgClient(config, RawCache(config.data_root))
    result = client.pull(collection, min_year=floor, force_rescan=args.force_rescan,
                         progress=progress, announce=announce)
    print(f"pull {args.source}: {result.listed} listed, {result.fetched} fetched, "
          f"{result.cached} already cached")
    if result.missing:
        print(f"  {len(result.missing)} listed item(s) the metadata API does not have; "
              f"they are retried on the next pull:")
        for identifier in result.missing:
            print(f"    {identifier}")
    if result.unidentified:
        # Counted rather than swallowed. An item in none of the counters is an item nobody
        # misses, and "listed" not adding up is the only signal anything went past unfetched.
        print(f"  {result.unidentified} listed item(s) carried no identifier and could not be "
              "fetched")
    if result.truncated:
        print("  warning: hit the paging backstop, so this listing is a PREFIX of the "
              "collection.\n  Ingest will look complete and will not be. Raise _MAX_PAGES.")
    return EXIT_OK


def _cmd_pack_lint(args) -> int:
    """Lint the resolved pack, reporting findings as human text or JSON with an honest code."""
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    try:
        diagnostics = lint(pack_dir)
    except DiagnosticError as exc:
        diagnostics = [exc.diagnostic]     # a structural failure is itself the one finding

    if getattr(args, "format", "human") == "json":
        print(json.dumps([diag.to_dict() for diag in diagnostics], indent=2))
    else:
        for diag in diagnostics:
            print(render(diag))
            print()
        errors = sum(1 for diag in diagnostics if diag.is_error)
        print(f"{pack_dir}: {errors} error(s), {len(diagnostics) - errors} other finding(s)")

    return EXIT_DIAGNOSTIC if any(diag.is_error for diag in diagnostics) else EXIT_OK


# Command name -> handler. A table rather than an if-chain, so adding a subcommand is one
# entry here and one parser above, and the dispatcher itself stops growing a branch per phase.
_COMMANDS = {
    "config": _cmd_config,
    "store": _cmd_store,
    "pull": _cmd_pull,
    "ingest": ingest,
    "dump": _cmd_dump,
}


def _run(args) -> int:
    """Dispatch a parsed command. Assumes a command argparse already accepted."""
    if args.command == "pack":
        return _cmd_pack_lint(args)          # resolves config lazily, only if --pack is absent
    return _COMMANDS[args.command](load_config(args.config), args)


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises past this boundary."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_OK

    try:
        return _run(args)
    except DiagnosticError as exc:
        print(render(exc.diagnostic), file=sys.stderr)
        return EXIT_DIAGNOSTIC
    except SourceError as exc:
        # An upstream having a bad day is an ordinary Tuesday, not a bug in this program, so it
        # renders like every other failure instead of arriving as a traceback. Nothing is
        # half-written: payloads are cached one at a time, and the next run resumes from there.
        print(render(Diagnostic(
            severity=ERROR,
            summary=f"the source could not be read: {exc}",
            detail="Nothing was lost -- whatever was fetched before this is cached, and the\n"
                   "next pull carries on from there. If it persists, the source is likely down\n"
                   "or serving an error page; wait rather than retrying in a loop.",
        )), file=sys.stderr)
        return EXIT_DIAGNOSTIC


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
