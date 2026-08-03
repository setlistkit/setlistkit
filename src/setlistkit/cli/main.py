# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit`` command dispatch.

This module has one job. It is the single place that catches a :class:`DiagnosticError` and
turns it into rendered stderr output and a non-zero exit code. Every other layer raises a
well-formed diagnostic and stays out of the presentation business.

Each command gets one builder function and one entry in ``_COMMANDS``. A command that reads
no config resolves it later, and ``_run`` names those.
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys
from pathlib import Path

from .. import __version__
from ..catalog.suggest import suggest
from ..config import load_config, require_network_identity
from ..diagnostics import ERROR, Diagnostic, DiagnosticError, render
from ..importers import IMPORTERS, configured
from ..picks.ledger import LEDGER_TABLES
from ..sources.client import SourceError
from ..store import Store

from .common import resolve_pack_dir
from .derive import derive
from .export import export
from .ingest import ingest
from ..picks.drops import DEFAULT_DEPTH
from .pack import pack_cmd, pack_init, pack_lint, write_suggestion
from .predict import drops_cmd, predict, replay
from .pull import pull_cmd, source_list
from .sources import curated_names, flatten, mirror_titles, report_inputs

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2

# How often a long pull says it is still going. One metadata request per item at roughly a
# request a second means a first pull of a large collection runs for a quarter of an hour, and
# silence for that long is indistinguishable from a hang.
_PROGRESS_EVERY = 25

# Seconds per REQUEST, for the estimate a dry run prints. MEASURED against archive.org (94 items in
# 2m25s, then 4614 more), not computed from the configured delay: the delay is only the half we
# control, and the round trip is the other half. archive.org makes one request per item, so this
# doubles as seconds-per-item there; setlist.fm pulls ~20 setlists per request, so its dry run
# reports its own request count and this same per-request cost applies. An estimate, and labeled as
# one -- but a real one, and the difference between "4222 items" and "about two hours" is the whole
# decision.
_SECONDS_PER_REQUEST = 1.5

# Sources `slkit pull` knows how to fetch: the importer registry's keys, so argparse and the
# registry can never disagree about what exists. Naming them (rather than deriving choices from the
# config's [sources.*] tables) still lets argparse reject an unknown name with a usage line.
_SOURCES = tuple(IMPORTERS)

# `slkit build` stages, in the order a run applies them.
_STAGES = ("ingest", "durations")

# argparse wraps help to the terminal, which on a wide window is unreadable.
_MAX_HELP_WIDTH = 90


class _WrappedHelp(argparse.ArgumentDefaultsHelpFormatter):
    """Held to a readable width, and printing each default from the parser rather than from prose.

    A default written into a help string is a second copy of the value, and the copy is the one
    that goes stale.
    """

    def __init__(self, prog, **kwargs):
        kwargs.setdefault("width", min(shutil.get_terminal_size().columns - 2, _MAX_HELP_WIDTH))
        super().__init__(prog, **kwargs)

    def _get_help_string(self, action):
        """Print a default only when a reader can act on it. Most of this surface has none."""
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


class _WrappedRawHelp(argparse.RawDescriptionHelpFormatter, _WrappedHelp):
    """The same width, keeping the description as written. The root lists stages; wrapping runs
    them together."""


def _set_formatter(parser: argparse.ArgumentParser) -> None:
    """Give ``parser`` and every command under it the wrapped formatter.

    Walked, because argparse does not hand a formatter down to a sub-parser, and passing it by
    hand is one more thing a new command can silently omit.
    """
    for action in parser._actions:  # pylint: disable=protected-access
        if not isinstance(action, argparse._SubParsersAction):  # pylint: disable=protected-access
            continue
        for child in action.choices.values():
            # Only the ones still on argparse's default. A command that asked for the raw
            # formatter did so because its description is shaped, and wrapping would undo it.
            if child.formatter_class is argparse.HelpFormatter:
                child.formatter_class = _WrappedHelp
            _set_formatter(child)


# The leader paragraph for `slkit --help`, kept out of the builder so it reads as text.
_ROOT_DESCRIPTION = """\
setlistkit predicts what a band plays next, from the tapes and setlists of what it played before.

The work moves in one direction. Each step reads what the step before it published:

  pull      fetch raw data from each configured source into a local cache
  build     parse the cache with a band pack, merge the sources, publish a corpus
  predict   rank every song's chance on one night
  replay    predict past nights blind, and grade the model against them
  export    write a bundle for another program to draw

`slkit bootstrap` runs that chain end to end on an empty deployment.

Two inputs decide the result, and both are yours. The config names the sources that take part.
The band pack names the songs. A mistake in the config makes a source silent; a gap in the pack
makes a real song invisible. Neither raises an error, so `slkit check` and `slkit pack check`
exist to ask.\
"""


# Wrapped by hand, because a shaped list needs the raw formatter and the raw formatter wraps
# nothing at all. Keep these inside the width above.
_BUILD_DESCRIPTION = """\
Turn the cached raw data into a published corpus. The stages run in order:

  ingest      parse the cache, merge every configured source, publish it
  durations   reconcile every tape of a night, so each song gets one length

Run `slkit pull` first; nothing here touches the network. The band pack decides
which titles become songs, so a gap in the pack shows up here as a song that is
quietly absent. The run reports what it dropped and why. A merge that shrinks the
store far below what it already holds is refused, and --force overrides that guard.\
"""

_COMPGEN_DESCRIPTION = """\
Print a completion script to standard output. The script is generated from this
program's own parser, so it names every command and flag that exists now. Nothing
is installed and nothing is written. Load it with:

  eval "$(slkit compgen)"

Put that line in your shell's startup file to keep it. Regenerate after an upgrade;
a script from an older version completes the older tree.\
"""


def _add_dry_run(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the show-me-what-you-would-do flag, under all three names people reach for.

    One flag, three spellings, because which one is "the" name is pure habit and getting it
    wrong on a command whose whole purpose is to be safe to try is a bad first experience. The
    dest stays ``dry_run`` whichever they type.
    """
    parser.add_argument("-n", "--dry-run", "--noop", dest="dry_run", action="store_true",
                        help=help_text)


def _add_check(sub) -> None:
    """The ``slkit check`` parser."""
    sub.add_parser(
        "check", help="validate the config, and ask each source whether it can name its band",
        description="Validate the config file. Then ask each configured source whether it can "
                    "name its band. A source that cannot name its band contributes nothing to a "
                    "merge, and every other command still exits 0, so this is where you find it. "
                    "Run it after any edit to the config.")


def _add_show(sub) -> None:
    """The ``slkit show`` parser."""
    show_cmd = sub.add_parser(
        "show", help="print what is on disk, as plain text",
        description="Print what is on disk, as plain text. Nothing here writes, and nothing here "
                    "reaches the network. Use it to read what a run left behind, without a "
                    "database client.")
    show_sub = show_cmd.add_subparsers(dest="show_action")
    show_sub.add_parser(
        "config", help="the configuration, after every default and path is resolved",
        description="Print the configuration after every default, override and path is resolved. "
                    "Use it to see the values the program uses, not the values the file holds.")
    show_sub.add_parser(
        "corpus", help="every derived table, narrowed by --since and --until",
        description="Print the derived tables. A table with no date axis is printed in full, and "
                    "says so.")
    show_sub.add_parser(
        "picks", help="the prediction ledger",
        description="Print the prediction ledger. These are the only tables here that cannot be "
                    "rebuilt from the cache, which makes this the strongest case for reading them "
                    "before anything else.")


def _add_store(sub) -> None:
    """The ``slkit store`` parser."""
    store_cmd = sub.add_parser(
        "store", help="create and inspect the state store",
        description="Create the state store, and report what it holds. The store is one SQLite "
                    "file under data_root. Every command that writes derived state writes there.")
    store_sub = store_cmd.add_subparsers(dest="store_action")
    store_sub.add_parser(
        "init", help="create data_root and apply schema migrations",
        description="Create data_root, then create or migrate the schema. Run it again on an "
                    "existing store safely: it applies the migrations that are missing, and no "
                    "more. `slkit bootstrap` runs this step for you.")
    store_sub.add_parser(
        "status", help="show schema version and table counts",
        description="Print the schema version, and the row count of each table. Use it to see "
                    "whether a run wrote what you expected.")


def _add_pull(sub) -> None:
    """The ``slkit pull`` parser."""
    parser = sub.add_parser(
        "pull", help="fetch raw source data into the cache",
        description="Fetch raw data from one source into the on-disk cache under data_root. "
                    "Nothing is parsed and nothing reaches the corpus here; run `slkit build` "
                    "next. Each request is rate-limited, and carries the user_agent the config "
                    "names. A cached item is re-asked conditionally, so an unchanged item moves "
                    "no bytes.")
    parser.add_argument(
        "--source", metavar="a,b", type=source_list,
        help="comma-separated sources to fetch (default: every source the config declares)")
    parser.add_argument(
        "--force-rescan", action="store_true",
        help="re-ask for data already cached (still rate-limited, and conditional: an "
             "unchanged item answers 304 and no bytes move)",
    )
    parser.add_argument(
        "--min-year", type=int, metavar="YEAR",
        help="ignore shows played before YEAR (overrides min_year in config)",
    )
    _add_dry_run(parser, "list the collection and report what a real run would fetch, "
                         "without fetching any of it")


def _add_build(sub) -> None:
    """The ``slkit build`` parser."""
    build_cmd = sub.add_parser(
        "build", help="turn the cached raw data into a published corpus",
        description=_BUILD_DESCRIPTION, formatter_class=_WrappedRawHelp)
    build_cmd.add_argument(
        "--stage", choices=_STAGES, default=None,
        help="run one stage only (default: every stage, in order)")
    build_cmd.add_argument("--min-year", type=int, metavar="YEAR",
                           help="the min_year the cache was pulled under, if not the configured "
                                "one (a listing is cached per collection and min_year)")
    _add_dry_run(build_cmd, "parse, merge and report in full, but write nothing to the database")
    build_cmd.add_argument(
        "--force", action="store_true",
        help="publish even when the merge produced far fewer shows than are stored (see the "
             "no-shrink guard)",
    )
    build_cmd.add_argument(
        "--profile", metavar="PATH",
        help="write the funnel's per-decision counts to PATH as JSON (see catalog.funnel); the "
             "counts are always taken during a parse, this only decides whether they are "
             "written out",
    )


def _add_predict(sub) -> None:
    """The ``slkit predict`` parser."""
    predict_cmd = sub.add_parser(
        "predict", help="rank every song's chance of being played on one night",
        description="Rank every song in the pack vocabulary by its chance of appearing on one "
                    "night. The run reads only shows dated before that night, so a night the "
                    "corpus already holds can be predicted blind. Use --asof to move that "
                    "boundary. The command writes nothing.")
    predict_cmd.add_argument(
        "--asof", metavar="YYYY-MM-DD",
        help="the night to predict (default: the day after the last stored show, which is the "
             "earliest date the corpus does not already hold). Nothing dated on or after it is "
             "read, so naming a night inside the corpus gives a blinded re-prediction of it",
    )
    _add_model_flags(predict_cmd)
    predict_cmd.add_argument(
        "--set-len", type=int, metavar="N",
        help="how many songs to assume will be played (overrides set_len in [model]). A live "
             "prediction has to guess this; a replay is told it by the night it grades",
    )


def _add_replay(sub) -> None:
    """The ``slkit replay`` parser."""
    replay_cmd = sub.add_parser(
        "replay",
        help="predict every gradeable night in a window before revealing it, and grade the lot",
        description="Predict each gradeable night in a window, then grade the prediction against "
                    "what the night held. Each night is predicted from the corpus as it stood "
                    "before that night, so no night grades itself. Use it to measure the model "
                    "after a change. The run writes to the picks ledger unless --dry-run is "
                    "given. --since and --until choose which nights are graded; they do not "
                    "change what a prediction may read.")
    # Inclusive at both ends and spelled exactly as `dump` and `export` spell them. A range that
    # means one thing in one command and something a day wider in another is a range nobody can
    # check a published number against.
    _add_model_flags(replay_cmd)
    _add_dry_run(replay_cmd, "predict and grade the whole window and report it, but write "
                             "nothing to the picks ledger")
    replay_cmd.add_argument(
        "--force", action="store_true",
        help="write even when the run leaves a ledger table with less than half the rows it "
             "started with (see the no-shrink guard)",
    )


def _add_drops(sub) -> None:
    """The ``slkit drops`` parser.

    The window flags are spelled and defaulted exactly as ``replay`` spells them, deliberately: a
    churn number and a recall number quoted off the same store have to be about the same nights,
    and two commands that disagree about what "the window" means make that impossible to check.
    """
    drops_parser = sub.add_parser(
        "drops",
        help="predict every gradeable night in a window from several earlier cutoffs, and watch "
             "the shortlist move",
        description="Predict each night in a window several times. Each pass reads the corpus as "
                    "it stood a step further back, so you see the shortlist move as the boundary "
                    "nears the night. A song that holds its place across every pass is a stable "
                    "pick; a song that only arrives at the last pass is not. The walk costs "
                    "nights times depth predictions.")
    drops_parser.add_argument(
        "--depth", type=int, metavar="N",
        help=f"how many earlier cutoffs to predict each night from (default: {DEFAULT_DEPTH}). "
             "The walk costs nights x depth predictions, so this is what decides whether it runs "
             "in seconds or minutes",
    )
    _add_model_flags(drops_parser)
    _add_dry_run(drops_parser, "walk the whole window and report it, but write nothing to the "
                               "picks ledger")
    drops_parser.add_argument(
        "--force", action="store_true",
        help="write even when the run leaves a ledger table with less than half the rows it "
             "started with (see the no-shrink guard)",
    )


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    """``--rollouts``, on both commands that draw.

    An iteration flag rather than a setting: a lower count shows the direction of a model change
    several times faster, and it is the wrong number to leave behind in a file a nightly run
    reads. The cost of a prediction is in the scoring pass over the corpus, not the draws, so
    lowering it buys less than it looks like it should.
    """
    parser.add_argument(
        "--rollouts", type=int, metavar="N",
        help="how many times to draw the night (overrides rollouts in [model])")


def _add_export(sub) -> None:
    """The ``slkit export`` parser."""
    export_cmd = sub.add_parser(
        "export", help="write what was derived as one file, for something else to draw",
        description="Write derived state as one self-contained JSON bundle, for another program "
                    "to draw. Each sub-command writes a different bundle. Nothing in the store "
                    "changes.")
    export_cmd.add_argument(
        "--explain", action="store_true",
        help="print what every configured [reports.*] window resolves to, and why -- builds "
             "nothing",
    )
    export_sub = export_cmd.add_subparsers(dest="export_action")

    tapemeasure_cmd = export_sub.add_parser(
        "tapemeasure", help="song lengths, features, credits and caveats as one JSON bundle",
        description="Write song lengths, features, tape credits and caveats as one JSON bundle. "
                    "Run `slkit build --stage=durations` first, or the lengths are absent. Every "
                    "statistic is recomputed over the window --since and --until name; none is "
                    "read whole-corpus and then given the window as a label.")
    tapemeasure_cmd.add_argument("--out", metavar="PATH", default="tapemeasure.json",
                                 help="where to write the bundle")
    _add_dry_run(tapemeasure_cmd, "build the bundle and report it, but write no file")

    songbook_cmd = export_sub.add_parser(
        "songbook", help="every show's songs, sorted into a vocabulary, as one JSON bundle",
        description="Write every show's songs, sorted into the pack vocabulary, as one JSON "
                    "bundle. --since and --until choose the shows, and the floor and the "
                    "de-duplication run over that window. A show below the floor, and a song the "
                    "pack does not name, are both counted and reported. Neither is dropped "
                    "without a number to show for it.")
    songbook_cmd.add_argument("--out", metavar="PATH", default="songbook.json",
                              help="where to write the bundle")
    _add_dry_run(songbook_cmd, "build the bundle and report it, but write no file")


def _add_pack(sub) -> None:
    """The ``slkit pack`` parser."""
    parser = sub.add_parser(
        "pack", help="work with band packs",
        description="Work with a band pack. A pack holds one band's vocabulary, aliases and "
                    "classifier rules. It is data you own and edit by hand, and the parser trusts "
                    "it, so a gap in a pack does not raise an error: it makes a real song "
                    "invisible. These sub-commands exist to find that gap before the corpus does.")
    pack_sub = parser.add_subparsers(dest="pack_action")
    init_cmd = pack_sub.add_parser(
        "init", help="scaffold the starter files a valid pack needs, without clobbering any",
        description="Create the starter files a valid pack needs. A file that already exists is "
                    "left alone, unless you give --force.")
    init_cmd.add_argument(
        "--force", action="store_true",
        help="overwrite scaffold files that already exist (default: leave them untouched)")
    check_cmd = pack_sub.add_parser(
        "check", help="validate a pack and run conformance checks",
        description="Check that a pack is well formed, then check its rules against the cached "
                    "corpus. Reports a rule that never fires, a rule another rule already covers, "
                    "an alias nothing can reach, and a title no rule names. It reads titles the "
                    "parser KEPT, so a song on no tape at all cannot reach it. `slkit pack "
                    "missing` asks that other question.")
    check_cmd.add_argument(
        "--format", choices=("human", "json"), default="human",
        help="output format",
    )
    check_cmd.add_argument(
        "--no-corpus", action="store_true",
        help="skip the checks that read the cached corpus (dead rules, redundant rules, "
             "unreachable aliases, unknown titles)",
    )
    draft_cmd = pack_sub.add_parser(
        "draft",
        help="draft a pack's vocabulary and aliases from raw tape titles",
        description="Cluster raw tape titles into a draft vocabulary and alias map, and WRITE "
                    "them to --out. A source that publishes song names wins the canonical "
                    "spelling, so a title welded to a track number does not become a song. The "
                    "draft is a starting point: read it before you trust it. To ask what the "
                    "pack is missing without writing anything, run `slkit pack missing`.",
    )
    draft_cmd.add_argument(
        "--from", dest="source", metavar="PATH",
        help="a JSON array of raw title strings (e.g. raw/archive_org/titles.json). Omit it to "
             "read the mirror source's cache, which is where `slkit bootstrap` reads",
    )
    draft_cmd.add_argument(
        "--curated", metavar="PATH",
        help="a JSON array of song names from a source that PUBLISHES titles (setlist.fm), "
             "which are clustered with --from but win the canonical spelling. Read from the "
             "configured setlist sources when --from is omitted",
    )
    draft_cmd.add_argument(
        "--out", metavar="DIR", required=True,
        help="directory to write vocabulary.json, aliases.json and SUGGESTIONS.md into",
    )
    draft_cmd.add_argument(
        "--cutoff", type=float, default=0.88, metavar="RATIO",
        help="difflib similarity at which two titles merge; higher is more conservative",
    )
    draft_cmd.add_argument(
        "--force", action="store_true",
        help="overwrite existing non-empty vocabulary.json / aliases.json in --out",
    )
    missing_cmd = pack_sub.add_parser(
        "missing",
        help="name the drafted candidates the pack does not already know",
        description="Draft candidates the same way `slkit pack draft` does, then name the ones "
                    "the pack does not already know. Writes nothing. This is the question to ask "
                    "after a later pull, and `slkit pack check` cannot ask it: check reads the "
                    "corpus, so a published song on no tape at all never reaches it.",
    )
    missing_cmd.add_argument(
        "--from", dest="source", metavar="PATH",
        help="a JSON array of raw title strings. Omit it to read the mirror source's cache",
    )
    missing_cmd.add_argument(
        "--curated", metavar="PATH",
        help="a JSON array of song names from a source that PUBLISHES titles. Read from the "
             "configured setlist sources when --from is omitted",
    )
    missing_cmd.add_argument(
        "--cutoff", type=float, default=0.88, metavar="RATIO",
        help="difflib similarity at which two titles merge; higher is more conservative",
    )


def _default_shell() -> str:
    """The login shell from ``$SHELL``, which is not always the running one."""
    return Path(os.environ.get("SHELL", "")).name or "bash"


def _add_compgen(sub) -> None:
    """The ``slkit compgen`` parser."""
    compgen_cmd = sub.add_parser(
        "compgen", help="print a shell completion script for this command tree",
        description=_COMPGEN_DESCRIPTION, formatter_class=_WrappedRawHelp)
    compgen_cmd.add_argument(
        "shell", nargs="?", default=_default_shell(),
        help="which shell to write for (default: read from $SHELL). shtab decides which shells "
             "it can write, and names them if it cannot write yours")


def _add_bootstrap(sub) -> None:
    """The ``slkit bootstrap`` parser: the whole pipeline behind one verb.

    The command takes no source argument. The config decides which sources take part.
    """
    boot = sub.add_parser(
        "bootstrap",
        help="carry an empty deployment to a reviewable pack and a published corpus in one run",
        description="Carry an empty deployment to a published corpus in one run. The run creates "
                    "the store, pulls each configured source, drafts a pack from what it pulled, "
                    "then ingests. The config decides which sources take part: a [sources.*] "
                    "table is the opt-in, and no source name is built into this command. A source "
                    "that returns nothing is named in the report, not treated as an error. Review "
                    "the drafted pack afterwards, and run `slkit pack check`.",
    )
    boot.add_argument(
        "--cutoff", type=float, default=0.88, metavar="RATIO",
        help="draft merge similarity; higher is more conservative")
    boot.add_argument(
        "--skip-pull", action="store_true",
        help="do not touch the network; draft and ingest from whatever is already cached")
    boot.add_argument(
        "--force-rescan", action="store_true",
        help="passed to the pull step: re-ask for data already cached")
    boot.add_argument(
        "--force", action="store_true",
        help="passed to the suggest step: overwrite an existing non-empty vocabulary/aliases")
    _add_dry_run(boot, "run store init, then report what a real pull would fetch, and stop "
                       "before drafting, scaffolding or ingesting anything")


def _build_parser() -> argparse.ArgumentParser:
    """Every command's parser, one call each.

    One builder per command rather than eighty lines in a row, so that adding a command is one
    function here and one entry in ``_COMMANDS`` below -- the same shape the dispatch table
    already has, and the reason it has it. Inline, each command left two or three locals behind
    for the next one to read past.
    """
    parser = argparse.ArgumentParser(
        prog="slkit",
        description=_ROOT_DESCRIPTION,
        epilog="Run `slkit <command> --help` to read what one command does.",
        formatter_class=_WrappedRawHelp,
    )
    parser.add_argument("--version", action="version", version=f"slkit {__version__}")
    parser.add_argument(
        "--config",
        metavar="PATH",
        help="path to a config file (overrides SLKIT_CONFIG and the default search order)",
    )
    # Global, because each one meant the same thing on every command that had it, and a window
    # that is a day wider in one command is a number nobody can check against another.
    parser.add_argument(
        "--pack", metavar="DIR",
        help="pack directory to work with (overrides [catalog] pack in config)")
    parser.add_argument(
        "--since", metavar="YYYY-MM-DD",
        help="ignore anything before this date. Inclusive, and read only by the commands that "
             "carry a date axis")
    parser.add_argument(
        "--until", metavar="YYYY-MM-DD",
        help="ignore anything after this date. Inclusive")

    sub = parser.add_subparsers(dest="command")

    _add_check(sub)
    _add_show(sub)
    _add_store(sub)
    _add_pull(sub)
    _add_build(sub)
    _add_predict(sub)
    _add_replay(sub)
    _add_drops(sub)
    _add_export(sub)
    _add_pack(sub)
    _add_bootstrap(sub)
    _add_compgen(sub)

    _set_formatter(parser)
    return parser


def _cmd_show(config, args) -> int:
    """`slkit show [config|corpus|picks]`, defaulting to the configuration."""
    action = getattr(args, "show_action", None)
    if action in ("corpus", "picks"):
        return _cmd_dump(config, argparse.Namespace(
            what="all" if action == "corpus" else "picks",
            since=args.since, until=args.until))
    return _cmd_config_show(config)


def _cmd_store(config, args) -> int:
    """`slkit store [init|status]`, defaulting to init."""
    if args.store_action == "status":
        return _cmd_store_status(config)
    return _cmd_store_init(config)


def _cmd_config_show(config) -> int:
    print(f"config file : {config.source_path}")
    print(f"data_root   : {config.data_root}")
    # Resolved the same way the working commands resolve it (relative to this config's own
    # directory, not the process cwd), so what prints here is the pack those commands will act on.
    pack = config.section("catalog").get("pack")
    if pack:
        pack_path = Path(str(pack)).expanduser()
        if not pack_path.is_absolute():
            pack_path = config.source_path.parent / pack_path
        print(f"pack        : {pack_path.resolve()}")
    else:
        print("pack        : (not set — pass --pack or set [catalog] pack)")
    print(f"user_agent  : {config.user_agent}")
    if config.user_agent_is_sentinel:
        print("            : (placeholder — network operations will refuse to run)")
    return EXIT_OK


def _unknown_sources(config) -> list[str]:
    """Report every ``[sources.*]`` table that matches no importer, and return their names.

    A table name IS the opt-in, so a name nothing matches opts nothing in. Nothing else can see
    this: the table is not a source that failed, it is a source that never existed, so no
    importer is ever asked about it and every command exits 0.

    Measured on one real config: ``[sources.setlistfm]`` against an importer id of
    ``setlist_fm``. That deployment ran for months on one source and read as if it had two.
    """
    # Read live rather than from _SOURCES, which argparse froze at import. A source registered
    # after that is still a source, and naming it unknown here would be this bug's own shape.
    known = tuple(IMPORTERS)
    unknown = [name for name in config.section("sources") if name not in IMPORTERS]
    for name in unknown:
        near = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
        suggestion = f" Did you mean [sources.{near[0]}]?" if near else ""
        print(f"\n[sources.{name}] matches no source this build knows, so nothing reads it."
              f"{suggestion}", file=sys.stderr)
        print(f"  Known sources: {', '.join(known)}", file=sys.stderr)
    return unknown


def _cmd_config_check(config) -> int:
    """Check the config, then ask each configured source whether it can identify its band.

    Two silences end here, and they look identical afterwards.

    A table nothing matches opts nothing in, so the source it names never runs. A table that DOES
    match can still fail to name its band, because a source needs one setting to say what it
    fetches: a collection, a MusicBrainz id. Each importer resolves that itself and raises its own
    diagnostic when the setting is absent.

    Both failures leave a full run, a clean exit and a smaller corpus. Measured on two real
    configs: a placeholder MusicBrainz id cost 265 songs and 346 shows, and a misspelled table
    name cost a whole source for months. Ask here, where the answer is cheap and the message
    names the exact line to change.

    Every source is asked, and one failure does not hide the next.
    """
    require_network_identity(config)
    unknown = _unknown_sources(config)
    broken = []
    for importer in configured(config):
        try:
            importer.identity(config)
        except DiagnosticError as err:
            broken.append((importer, err))
    for importer, err in broken:
        print(f"\n{importer.display_name} is configured but cannot name its band:", file=sys.stderr)
        print(render(err.diagnostic), file=sys.stderr)
    if broken or unknown:
        if broken:
            names = ", ".join(importer.id for importer, _ in broken)
            print(f"\n{config.source_path}: {names} cannot run. Each one would contribute nothing\n"
                  "to a merge, and every command would still exit 0.", file=sys.stderr)
        return EXIT_DIAGNOSTIC
    checked = ", ".join(importer.id for importer in configured(config)) or "(no sources)"
    print(f"ok: {config.source_path} is valid and identifies itself to the network")
    print(f"    sources that can name their band: {checked}")
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
    only = None if args.what == "all" else LEDGER_TABLES
    try:
        with Store(config.data_root) as store:
            print(store.dump(since=args.since, until=args.until, only=only), end="")
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


def _cmd_predict(config, args) -> int:
    """`slkit predict`, with the same refusal every other dated command gives."""
    try:
        return predict(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "predict from") from exc


def _cmd_replay(config, args) -> int:
    """`slkit replay`, likewise.

    Wrapped here rather than inside the walk for the reason ``_cmd_export`` gives: a malformed
    flag is a CLI fault, and the replay's job starts once the window is known to mean what it
    says. Unwrapped, ``--since "the fall tour"`` arrives as a traceback -- which is the one shape
    of output this program does not use to say no.
    """
    try:
        return replay(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "grade") from exc


def _cmd_drops(config, args) -> int:
    """`slkit drops`, with the same refusal ``replay`` gives a window it cannot read."""
    try:
        return drops_cmd(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "walk") from exc


def _cmd_export(config, args) -> int:
    """`slkit export`, with the same refusal ``dump`` gives a date it cannot compare.

    Wrapped here rather than inside the exporter because a malformed flag is a CLI fault: the
    exporter's job starts once the window is known to mean what it says.
    """
    try:
        return export(config, args)
    except ValueError as exc:
        raise _malformed_date(exc, "write") from exc


_BOOTSTRAP_STEPS = 7


def _step(number: int, title: str) -> None:
    """Print one banner for one bootstrap step.

    The banners divide a run into marked phases. Without them the output is one wall of text.
    """
    print(f"\n=== step {number}/{_BOOTSTRAP_STEPS}: {title} ===", flush=True)


def _cmd_bootstrap(config, args) -> int:
    """Run the whole pipeline behind one verb: store, pull, draft, scaffold, lint, ingest, derive.

    The pipeline initializes itself. Every parameter comes from the resolved config. A new
    deployment thus goes from an empty directory to a draft pack and a published corpus in one
    run.

    The config decides which sources take part. A ``[sources.<id>]`` table is the opt-in, and the
    operator writes one table for each source they have. This command names no source itself. A
    source added to the registry later works here when the operator configures it.

    Each step is cheap to repeat. Store init and pack init make the same result every time. A
    pull reads its own cache for free. The draft step refuses to overwrite a vocabulary that a
    person started to edit.
    """
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    sources = configured(config)

    _step(1, "store init — create data_root and apply the schema")
    _cmd_store_init(config)

    if args.skip_pull:
        _step(2, "pull — SKIPPED (--skip-pull); drafting from the existing cache")
    else:
        _step(2, "pull — fetch raw data into the cache, one configured source at a time")
        pull_cmd(config, argparse.Namespace(
            source=[importer.id for importer in sources], min_year=None,
            force_rescan=args.force_rescan, dry_run=args.dry_run))

    if args.dry_run:
        print("\ndry run: stopping before draft, scaffold and ingest. "
              "Re-run without --dry-run to build.")
        return EXIT_OK

    _step(3, f"pack draft — draft a vocabulary into {pack_dir}")
    # Two inputs, and a deployment can have either one alone. Tape titles come from the mirror
    # source. Published song names come from the setlist sources. A band that nobody taped has
    # only the names, and those are the better input: a person typed them on purpose.
    titles = mirror_titles(config)
    by_source = curated_names(config)
    curated = flatten(by_source)
    report_inputs(titles, by_source, config)
    if not titles and not curated:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="the cache holds nothing to draft a vocabulary from",
            detail="Each configured source cached no usable data under this config. Check that\n"
                   "your config declares a [sources.<id>] table for each source you have. Then\n"
                   "run `slkit pull` (or re-run bootstrap without --skip-pull) and retry.\n\n"
                   f"Sources configured now: {', '.join(imp.id for imp in sources) or '(none)'}",
        ))
    write_suggestion(suggest(titles, cutoff=args.cutoff, curated=curated),
                     pack_dir, force=args.force)

    _step(4, "pack init — scaffold the remaining pack files, clobbering none")
    pack_init(argparse.Namespace(pack=str(pack_dir), force=False, config=args.config))

    _step(5, "pack check — validate the drafted pack")
    if pack_lint(argparse.Namespace(
            pack=str(pack_dir), format="human", no_corpus=False, config=args.config)) != EXIT_OK:
        print("  (lint reported findings — expected for a fresh draft; they are a worklist, "
              "not a failure)")

    _step(6, "build --stage=ingest — parse the cache and publish the merged corpus")
    # No source is named here. Ingest reads the mirror the config declares, and it folds each
    # configured setlist source on top of it.
    ingest(config, argparse.Namespace(
        source=None, pack=str(pack_dir), min_year=None,
        dry_run=False, force=False, profile=None, config=args.config))

    _step(7, "build --stage=durations — reconcile each song's length across every tape of a night")
    derive(config, argparse.Namespace(
        derive_action="durations", pack=str(pack_dir), dry_run=False, config=args.config))

    print(f"\nbootstrap complete. The corpus is ingested and durations are derived — "
          f"`slkit export`\nwill now write a bundle to report from. Review "
          f"{pack_dir / 'SUGGESTIONS.md'}, trim vocabulary.json\nby hand, then re-run "
          f"`slkit build` and `slkit build --stage=durations` to apply your edits.")
    return EXIT_OK


# Command name -> handler. A table rather than an if-chain, so adding a subcommand is one
# entry here and one parser above, and the dispatcher itself stops growing a branch per phase.
def _cmd_build(config, args) -> int:
    """Run the build stages in order, or the one ``--stage`` names.

    A later stage reads what the one before it published, so a run that stops on a failure stops
    there rather than carrying on with a corpus that was never written.
    """
    stages = {"ingest": ingest, "durations": derive}
    for name in ([args.stage] if args.stage else _STAGES):
        if len(_STAGES) > 1 and not args.stage:
            print(f"\n== {name} ==", flush=True)
        code = stages[name](config, args)
        if code != EXIT_OK:
            return code
    return EXIT_OK


_COMMANDS = {
    "check": lambda config, args: _cmd_config_check(config),
    "show": _cmd_show,
    "store": _cmd_store,
    "pull": pull_cmd,
    "build": _cmd_build,
    "predict": _cmd_predict,
    "replay": _cmd_replay,
    "drops": _cmd_drops,
    "export": _cmd_export,
    "bootstrap": _cmd_bootstrap,
}


def _for_completion(parser: argparse.ArgumentParser, shtab) -> None:
    """Prepare a throwaway parser for shtab, so ``--help`` keeps what it prints.

    Two edits. Each command takes the one line its parent lists it under, because a completion
    menu gives one row per command, and shtab reads ``description`` and never ``help``: a
    paragraph arrives there unwrapped and cut mid-word.

    An argument that takes a path also says so. Without that, the shell offers a command name
    where a filename belongs. The metavar carries the answer this project already writes.
    """
    for action in parser._actions:  # pylint: disable=protected-access
        if isinstance(action, argparse._SubParsersAction):  # pylint: disable=protected-access
            # pylint: disable-next=protected-access
            summaries = {pseudo.dest: pseudo.help for pseudo in action._choices_actions}
            for name, child in action.choices.items():
                child.description = summaries.get(name) or child.description
                _for_completion(child, shtab)
        elif action.metavar == "PATH":
            action.complete = shtab.FILE
        elif action.metavar == "DIR":
            action.complete = shtab.DIRECTORY


def _cmd_compgen(args) -> int:
    """Write a completion script for the parser this program builds."""
    # Imported here so the other commands do not pay for it.
    import shtab  # pylint: disable=import-outside-toplevel

    parser = _build_parser()
    _for_completion(parser, shtab)
    try:
        script = shtab.complete(parser, shell=args.shell)
    except NotImplementedError as err:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"no completion script for this shell: {err}",
            detail="Name a shell the library writes, or ask upstream for yours. Completions are\n"
                   "the shell's own format, so each one is separate work.",
        )) from err
    print(script)
    return EXIT_OK


def _run(args) -> int:
    """Dispatch a parsed command. Assumes a command argparse already accepted."""
    if args.command == "pack":
        return pack_cmd(args)               # resolves config lazily, only if --pack is absent
    if args.command == "compgen":
        return _cmd_compgen(args)            # reads the parser, never the config
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
