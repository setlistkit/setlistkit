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
from .derive import derive
from .export import export
from .ingest import ingest

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2

# How often a long pull says it is still going. One metadata request per item at roughly a
# request a second means a first pull of a large collection runs for a quarter of an hour, and
# silence for that long is indistinguishable from a hang.
_PROGRESS_EVERY = 25

# Seconds per item, for the estimate a dry run prints. MEASURED against archive.org (94 items in
# 2m25s, then 4614 more), not computed from the configured delay: the delay is only the half we
# control, and the round trip is the other half. An estimate, and labeled as one -- but a real
# one, and the difference between "4222 items" and "about two hours" is the whole decision.
_SECONDS_PER_ITEM = 1.5

# Sources `slkit pull` knows how to fetch. Named here rather than derived from the config's
# [sources.*] tables: a typo'd table name would otherwise become a source that silently does
# nothing, and argparse can reject an unknown name with a usage line instead.
_SOURCES = ("archive_org",)


def _add_dry_run(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the show-me-what-you-would-do flag, under all three names people reach for.

    One flag, three spellings, because which one is "the" name is pure habit and getting it
    wrong on a command whose whole purpose is to be safe to try is a bad first experience. The
    dest stays ``dry_run`` whichever they type.
    """
    parser.add_argument("-n", "--dry-run", "--noop", dest="dry_run", action="store_true",
                        help=help_text)


def _add_config(sub) -> None:
    """The ``slkit config`` parser."""
    config_cmd = sub.add_parser("config", help="inspect the resolved configuration")
    config_sub = config_cmd.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="print the resolved configuration")
    config_sub.add_parser("check", help="validate the config, including network identity")


def _add_store(sub) -> None:
    """The ``slkit store`` parser."""
    store_cmd = sub.add_parser("store", help="create and inspect the state store")
    store_sub = store_cmd.add_subparsers(dest="store_action")
    store_sub.add_parser("init", help="create data_root and apply schema migrations")
    store_sub.add_parser("status", help="show schema version and table counts")


def _add_dump(sub) -> None:
    """The ``slkit dump`` parser."""
    dump_cmd = sub.add_parser("dump", help="print a plain-text view of derived state")
    # Inclusive at both ends. A half-open range is the classic off-by-one, and this is the view
    # someone reaches for when they already suspect something is wrong -- one bug at a time.
    dump_cmd.add_argument("--since", metavar="YYYY-MM-DD",
                          help="only rows on or after this date (tables with no date axis are "
                               "printed in full, and say so)")
    dump_cmd.add_argument("--until", metavar="YYYY-MM-DD",
                          help="only rows on or before this date")


def _add_pull(sub) -> None:
    """The ``slkit pull`` parser."""
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
    _add_dry_run(pull_cmd, "list the collection and report what a real run would fetch, "
                           "without fetching any of it")


def _add_ingest(sub) -> None:
    """The ``slkit ingest`` parser."""
    ingest_cmd = sub.add_parser(
        "ingest", help="parse the cached raw data and publish the merged corpus")
    ingest_cmd.add_argument("source", choices=_SOURCES, nargs="?", default=_SOURCES[0],
                            help="which cached source to parse (default: %(default)s)")
    ingest_cmd.add_argument("--pack", metavar="PATH",
                            help="pack directory to parse with (overrides [catalog] pack)")
    ingest_cmd.add_argument("--min-year", type=int, metavar="YEAR",
                            help="the min_year the cache was pulled under, if not the configured "
                                 "one (a listing is cached per collection and min_year)")
    _add_dry_run(ingest_cmd, "parse, merge and report in full, but write nothing to the database")
    ingest_cmd.add_argument(
        "--force", action="store_true",
        help="publish even when the merge produced far fewer shows than are stored (see the "
             "no-shrink guard)",
    )
    ingest_cmd.add_argument(
        "--profile", metavar="PATH",
        help="write the funnel's per-decision counts to PATH as JSON (see catalog.funnel); the "
             "counts are always taken during a parse, this only decides whether they are "
             "written out",
    )


def _add_derive(sub) -> None:
    """The ``slkit derive`` parser."""
    derive_cmd = sub.add_parser(
        "derive", help="compute derived state from what ingest published")
    derive_sub = derive_cmd.add_subparsers(dest="derive_action")
    durations_cmd = derive_sub.add_parser(
        "durations", help="how long each song runs, reconciled across every tape of a night")
    durations_cmd.add_argument("--pack", metavar="PATH",
                               help="pack directory to derive with (overrides [catalog] pack)")
    _add_dry_run(durations_cmd,
                 "read, reconcile and report in full, but write nothing to the database")


def _add_export_common(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add --since/--until to one export sub-verb's parser, each carrying its own explanation.

    Mirrors ``_add_dry_run``'s idiom: one implementation, called once per sub-verb with the text
    that explains what narrowing means for THAT bundle, because the two do not mean the same
    thing (the Tape Measure recomputes statistics over the window; the Songbook floors and
    dedupes over it). NOT added to the parent ``export`` parser: argparse would then require
    ``slkit export --since ... songbook``, and ``slkit export songbook --help`` would not list the
    flag at all -- the same recursive-help complaint ``_add_dry_run`` already exists to avoid.

    Spelled and bounded exactly as ``slkit dump`` spells them -- inclusive at both ends, same
    YYYY-MM-DD, same two names. A range that means one thing in one command and something a day
    wider in another is a range nobody can check a published number against.
    """
    parser.add_argument("--since", metavar="YYYY-MM-DD",
                        help=f"only {help_text}, on or after this date")
    parser.add_argument("--until", metavar="YYYY-MM-DD",
                        help=f"only {help_text}, on or before this date")


def _add_export(sub) -> None:
    """The ``slkit export`` parser."""
    export_cmd = sub.add_parser(
        "export", help="write what was derived as one file, for something else to draw")
    export_cmd.add_argument(
        "--explain", action="store_true",
        help="print what every configured [reports.*] window resolves to, and why -- builds "
             "nothing",
    )
    export_sub = export_cmd.add_subparsers(dest="export_action")

    tapemeasure_cmd = export_sub.add_parser(
        "tapemeasure", help="song lengths, features, credits and caveats as one JSON bundle")
    tapemeasure_cmd.add_argument("--out", metavar="PATH", default="tapemeasure.json",
                                 help="where to write the bundle (default: %(default)s)")
    _add_export_common(tapemeasure_cmd,
                       "performances (song statistics are recomputed over the window, not "
                       "read whole-corpus)")
    _add_dry_run(tapemeasure_cmd, "build the bundle and report it, but write no file")

    songbook_cmd = export_sub.add_parser(
        "songbook", help="every show's songs, sorted into a vocabulary, as one JSON bundle")
    songbook_cmd.add_argument("--out", metavar="PATH", default="songbook.json",
                              help="where to write the bundle (default: %(default)s)")
    songbook_cmd.add_argument(
        "--pack", metavar="PATH",
        help="pack directory to canonicalize song names with (overrides [catalog] pack)")
    _add_export_common(songbook_cmd,
                       "shows (below-floor shows and out-of-vocabulary songs are still "
                       "counted and reported, never silently dropped)")
    _add_dry_run(songbook_cmd, "build the bundle and report it, but write no file")


def _add_pack(sub) -> None:
    """The ``slkit pack`` parser."""
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
    lint_cmd.add_argument(
        "--no-corpus", action="store_true",
        help="skip the checks that read the cached corpus (dead rules, redundant rules, "
             "unreachable aliases, unknown titles)",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Every command's parser, one call each.

    One builder per command rather than eighty lines in a row, so that adding a command is one
    function here and one entry in ``_COMMANDS`` below -- the same shape the dispatch table
    already has, and the reason it has it. Inline, each command left two or three locals behind
    for the next one to read past.
    """
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

    _add_config(sub)
    _add_store(sub)
    _add_dump(sub)
    _add_pull(sub)
    _add_ingest(sub)
    _add_derive(sub)
    _add_export(sub)
    _add_pack(sub)

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


def _human_bytes(count: int) -> str:
    """A byte count at a glance. Three significant figures is all anyone reads."""
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"                                    # pragma: no cover - loop returns


def _cmd_store_status(config) -> int:
    store = Store(config.data_root)
    print(f"db          : {store.db_path}")
    print(f"raw cache   : {store.raw.root}")
    # What is actually in it, not just where it is. This is the no-network answer to "how far
    # has the pull got", and it is safe to run in a loop -- `slkit pull -n` answers it better
    # but spends search requests to do it, which makes it the wrong thing to poll with.
    for namespace, stats in store.raw.stats().items():
        print(f"  {namespace}: {stats.entries} entries, {_human_bytes(stats.bytes)}")
    # Don't open (and thereby create) the DB just to report it doesn't exist yet.
    if not store.db_path.is_file():
        print("schema      : not initialized — run `slkit store init`")
        return EXIT_OK
    with store:
        print(f"schema      : version {store.schema_version()}")
        for table, count in store.table_counts().items():
            print(f"  {table}: {count} rows")
    return EXIT_OK


def _cmd_dump(config, args) -> int:
    """Print the store as text, optionally narrowed to a date range.

    A malformed date is refused rather than compared. Dates are stored as YYYY-MM-DD strings and
    the range is a string comparison, so ``--until 2023`` sorts below every date IN 2023 and would
    hand back an empty dump of the year that was asked for -- which reads as "there is nothing
    there", the one answer this view exists to make trustworthy.
    """
    try:
        with Store(config.data_root) as store:
            print(store.dump(since=args.since, until=args.until), end="")
    except ValueError as exc:
        raise _malformed_date(exc, "print") from exc
    return EXIT_OK


def _malformed_date(exc: ValueError, verb: str) -> DiagnosticError:
    """The refusal both ranged commands give, so they cannot explain the same trap differently."""
    return DiagnosticError(Diagnostic(
        severity=ERROR,
        summary=str(exc),
        detail="Dates are stored and compared as YYYY-MM-DD text, so a shorter one does not\n"
               "mean what it looks like: `--until 2023` sorts below every date in 2023 and\n"
               f"would {verb} an empty range rather than the year you asked for.",
    ))


def _cmd_export(config, args) -> int:
    """`slkit export`, with the same refusal ``dump`` gives a date it cannot compare.

    Wrapped here rather than inside the exporter because a malformed flag is a CLI fault: the
    exporter's job starts once the window is known to mean what it says.
    """
    try:
        return export(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "write") from exc


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
                         dry_run=args.dry_run, progress=progress, announce=announce)
    if args.dry_run:
        minutes = result.planned * _SECONDS_PER_ITEM / 60
        print(f"pull {args.source} (dry run): {result.listed} listed, "
              f"{result.cached} already cached, {result.planned} would be fetched "
              f"(about {minutes:.0f} min)")
        print("  No item metadata was requested. The listing itself was, because that is how\n"
              "  this learns what is new; it is a few requests against thousands.")
    else:
        print(f"pull {args.source}: {result.listed} listed, {result.fetched} fetched, "
              f"{result.cached} already cached")
    if result.missing:
        print(f"  {len(result.missing)} listed item(s) the metadata API does not have; "
              f"they are retried on the next pull:")
        for identifier in result.missing:
            print(f"    {identifier}")
    if result.failed:
        print(f"  {len(result.failed)} item(s) the source could not serve, even after retries. "
              "Nothing was cached\n  for them, so the next pull tries them again:")
        for identifier in result.failed:
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


def _lint_corpus(args) -> list:
    """The cached source items to lint against, or empty when there is no usable cache.

    Best-effort by design. ``pack lint --pack PATH`` has always worked with no config at all, and
    a pack author checking their own file should not be made to configure a data_root first. When
    there is no corpus the checks that need one report themselves skipped, which is the honest
    outcome and not a failure.
    """
    if getattr(args, "no_corpus", False):
        return []
    # Absence is fine; malformed is not, and only the first is swallowed. No config file at all,
    # or a config with no source configured, are both ordinary states -- `pack lint --pack PATH`
    # has always worked with neither, and a pack author checking their own file should not have
    # to set up a data_root first. A config that EXISTS and is wrong is a different thing: a
    # quoted min_year or unreadable TOML is a fault `slkit ingest` treats as fatal, and
    # swallowing it here told the user to run a pull they had already run while hiding the
    # reason it did not take.
    try:
        config = load_config(args.config)
    except DiagnosticError:
        return []
    if not str(config.section("sources", "archive_org").get("collection") or "").strip():
        return []
    collection = required_setting(config, ("sources", "archive_org"), "collection", "")
    cached = ArchiveOrgClient(config, RawCache(config.data_root)).cached_items(
        collection, min_year=min_year(config, "archive_org", None))
    if cached.truncated or cached.absent:
        # The same facts `slkit ingest` refuses to publish over. A lint run against a fragment of
        # the corpus reports most of a pack dead and ranks its unknown titles against a fraction
        # of the real play counts -- and every one of those findings recommends deletion.
        print(f"warning: the cached corpus is incomplete ({len(cached.absent)} listed item(s) "
              f"unreadable"
              f"{', listing truncated' if cached.truncated else ''}).\n"
              "  Corpus-aware findings below are measured against what is cached, not against "
              "the collection.\n  Finish `slkit pull archive_org` before deleting anything on "
              "their say-so.", file=sys.stderr)
    return cached.items


def _cmd_pack_lint(args) -> int:
    """Lint the resolved pack, reporting findings as human text or JSON with an honest code."""
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    try:
        diagnostics = lint(pack_dir, _lint_corpus(args))
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
    "derive": derive,
    "export": _cmd_export,
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
