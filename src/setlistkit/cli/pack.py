# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""``slkit pack``: scaffold a pack, lint it, draft a vocabulary, and diff a draft against it.

A pack is a different kind of input from a source, and these commands are a different kind of
command. A source produces claims, and a pull of it ends. A pack states what anything is called,
a person edits it by hand, and the work ends when that person is satisfied.

The failure modes differ too, and that decides how much this module reports. A source that is
absent leaves gaps somebody can count. A pack that is thin leaves a full run, a clean exit, and
wrong numbers downstream. So every command here says what it measured against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..catalog.lint import lint
from ..catalog.pack import load_pack
from ..catalog.scaffold import starter_pack
from ..catalog.suggest import render_markdown, suggest
from ..config import load_config
from ..diagnostics import ERROR, Diagnostic, DiagnosticError, render
from ..importers import mirror
from ..store.raw_cache import RawCache
from .common import min_year, resolve_pack_dir
from .sources import curated_names, flatten, mirror_titles, report_inputs

EXIT_OK = 0
EXIT_DIAGNOSTIC = 2


def _lint_corpus(args) -> list:
    """The cached mirror items to lint against, or empty when there is no usable cache.

    Best-effort by design. ``pack lint --pack PATH`` works with no config at all, and a pack
    author who checks their own file must not have to configure a data_root first. The checks
    that need a corpus report themselves skipped when there is none.

    Absence is fine here; a malformed config is not. No config file, or a config that declares no
    mirror, are both ordinary states. A config that EXISTS and is wrong is a different thing. A
    quoted min_year or unreadable TOML is a fault that ``slkit ingest`` treats as fatal. This once
    swallowed it, and then told the operator to run a pull they had already run.

    The mirror comes from the config, never from a name written here. A deployment with no mirror
    lints its pack against no corpus, which is correct: it has no tapes to measure against.
    """
    if getattr(args, "no_corpus", False):
        return []
    try:
        config = load_config(args.config)
    except DiagnosticError:
        return []
    importer = mirror(config)
    if importer is None:
        return []
    try:
        identity = importer.identity(config)
    except DiagnosticError:
        # The mirror is declared but cannot name what it fetches. `slkit config check` reports
        # that fault with the exact setting to add. A lint run states what it lost and continues.
        print(f"warning: {importer.display_name} is configured but cannot name its band, so "
              "corpus-aware\n  checks are skipped. Run `slkit config check`.", file=sys.stderr)
        return []
    client = importer.make_client(config, RawCache(config.data_root))
    cached = client.cached_items(identity, min_year=min_year(config, importer.id, None))
    if cached.truncated or cached.absent:
        # The same facts `slkit ingest` refuses to publish over. A lint run against a fragment of
        # the corpus reports most of a pack dead, and it ranks unknown titles against a fraction
        # of the real play counts. Every one of those findings recommends a deletion.
        print(f"warning: the cached corpus is incomplete ({len(cached.absent)} listed item(s) "
              f"unreadable"
              f"{', listing truncated' if cached.truncated else ''}).\n"
              "  Corpus-aware findings below are measured against what is cached, not against "
              "the collection.\n  Finish the pull before you delete anything on their say-so.",
              file=sys.stderr)
    return cached.items


def pack_lint(args) -> int:
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


def pack_cmd(args) -> int:
    """`slkit pack [init|lint|suggest-vocab]`, defaulting to lint (the original behavior)."""
    action = getattr(args, "pack_action", None)
    if action == "init":
        return pack_init(args)
    if action == "suggest-vocab":
        return pack_suggest(args)
    return pack_lint(args)


def pack_init(args) -> int:
    """Drop the starter files a valid pack needs into --pack, never clobbering existing ones.

    Writes only files that are absent, so it composes with suggest-vocab in either order: run it
    first for an empty skeleton suggest-vocab then fills, or run it after to add the pack.json and
    policy files a suggest-vocab run leaves out -- and in that second case it leaves the
    already-written vocabulary.json and aliases.json exactly as they are. ``--force`` overwrites.
    """
    pack_dir = Path(args.pack).expanduser()
    files = starter_pack(pack_dir.resolve().name or "pack")
    pack_dir.mkdir(parents=True, exist_ok=True)

    created, skipped = [], []
    for name, content in files.items():
        path = pack_dir / name
        if path.exists() and not args.force:
            skipped.append(name)
            continue
        path.write_text(content, encoding="utf-8")
        created.append(name)

    if created:
        print(f"scaffolded {pack_dir}:")
        for name in created:
            print(f"  wrote {name}")
    if skipped:
        print(f"left {len(skipped)} existing file(s) untouched: {', '.join(skipped)}")
        print("(pass --force to overwrite them)")
    if not created:
        print(f"{pack_dir} already has every starter file; nothing to do")
    else:
        print("\nEdit pack.json (add a band_name), fill vocabulary.json, then `slkit pack lint`.")
    return EXIT_OK


def _nonempty_pack_file(path: Path) -> bool:
    """True when ``path`` holds a pack file with real content already -- the thing --force guards.

    An empty ``[]`` / ``{}`` is treated as absent: a fresh pack directory that was `git init`ed with
    placeholder files should not need --force to fill in. Anything unparseable is treated as real,
    because refusing to clobber something we cannot read is the safe direction."""
    if not path.is_file():
        return False
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return True


def _read_titles(source: Path) -> list[str]:
    """A JSON array of title strings from ``source``, or a diagnostic naming what was wrong.

    Shared by ``--from`` and ``--curated`` so the two read the same shape and fail the same way.
    """
    try:
        titles = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"could not read titles from {source}",
            detail=f"{exc}\n\nExpected a JSON array of title strings, such as the "
                   f"titles.json produced by the dirty-stripper jq filter.",
        )) from exc
    if not isinstance(titles, list) or not all(isinstance(t, str) for t in titles):
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary=f"{source} is not a JSON array of strings",
            detail="This command reads a flat list of raw title strings and clusters them.",
        ))
    return titles


def _suggest_inputs(args) -> tuple[list[str], list[str]]:
    """Resolve the two draft inputs: tape titles, and published song names.

    A path given on the command line wins. For an input with no path, read the same place
    ``bootstrap`` reads: the mirror source's cache for titles, and each configured setlist source
    for names.

    This command once demanded both paths. ``bootstrap`` could find these inputs and this could
    not, which put the hand work on the repeat case and the automation on the case that runs one
    time. A band with a pack under revision is the one that redrafts, so it had the worst of it.
    """
    if args.source:
        titles = _read_titles(Path(args.source).expanduser())
        curated = _read_titles(Path(args.curated).expanduser()) if args.curated else []
        return titles, curated
    config = load_config(args.config)
    titles = mirror_titles(config)
    by_source = curated_names(config)
    report_inputs(titles, by_source, config)
    return titles, flatten(by_source)


def _pack_diff(args, suggestion) -> int:
    """Report the drafted candidates that the pack does not already name. Write nothing.

    This answers a question no other command answers. ``pack lint`` reads the corpus and reports
    titles the parser KEPT but the pack cannot name. This reads the sources and reports songs
    they PUBLISH that the pack lacks. A song on no tape at all still appears.

    Compare by normalized key, not by spelling. A pack that spells a song its own way already
    knows it.
    """
    pack_dir = resolve_pack_dir(getattr(args, "pack", None), args.config)
    pack = load_pack(pack_dir)
    known = {pack.normalizer.normalize(name) for name in pack.vocabulary}
    known |= {pack.normalizer.normalize(name) for name in pack.normalizer.aliases()}
    fresh = [cluster for cluster in suggestion.songs
             if pack.normalizer.normalize(cluster.canonical) not in known]
    print(f"\n{pack_dir}: {len(pack.vocabulary)} song(s) in the pack, "
          f"{len(suggestion.songs)} candidate(s) drafted, {len(fresh)} not in the pack")
    if not fresh:
        print("The pack already names every candidate. Nothing to add.")
        return EXIT_OK
    named = [cluster for cluster in fresh if cluster.curated]
    print(f"{len(named)} of them carry a name a setlist source published.\n")
    for cluster in sorted(fresh, key=lambda c: (not c.curated, c.canonical.casefold())):
        print(f" {'*' if cluster.curated else ' '} {cluster.canonical}")
    print("\n* = a setlist source published this name. Add what belongs, by hand.")
    return EXIT_OK


def pack_suggest(args) -> int:
    """Draft vocabulary.json, aliases.json and SUGGESTIONS.md from a bag of raw titles."""
    titles, curated = _suggest_inputs(args)
    if not titles and not curated:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="no titles to draft from",
            detail="Pass --from PATH, or configure the sources and pull them. Then run\n"
                   "`slkit config check` to confirm each one can name its band.",
        ))
    suggestion = suggest(titles, cutoff=args.cutoff, curated=curated)
    if args.diff:
        return _pack_diff(args, suggestion)
    if not args.out:
        raise DiagnosticError(Diagnostic(
            severity=ERROR,
            summary="--out is required when a draft is written",
            detail="Pass --out DIR to write the draft, or pass --diff to compare against the\n"
                   "pack and write nothing.",
        ))
    out_dir = Path(args.out).expanduser()
    write_suggestion(suggestion, out_dir, force=args.force)
    if (out_dir / "pack.json").is_file():
        print("\nReview SUGGESTIONS.md, trim by hand, then `slkit pack lint`.")
    else:
        print(f"\nReview SUGGESTIONS.md and trim by hand. This is not a complete pack yet: run\n"
              f"  slkit pack init --pack {out_dir}\nto add pack.json and the policy files, "
              f"then `slkit pack lint`.")
    return EXIT_OK


def write_suggestion(suggestion, out_dir: Path, *, force: bool) -> None:
    """Write vocabulary.json, aliases.json and SUGGESTIONS.md into ``out_dir``.

    Shared by ``pack suggest-vocab`` and ``bootstrap`` so the refuse-to-clobber rule and the
    summary line are defined once. A non-empty vocabulary/aliases already present is refused
    unless ``force``; SUGGESTIONS.md is always safe to rewrite (it is only ever read by a human).
    """
    vocab_path = out_dir / "vocabulary.json"
    aliases_path = out_dir / "aliases.json"
    sugg_path = out_dir / "SUGGESTIONS.md"

    if not force:
        clashing = [p for p in (vocab_path, aliases_path) if _nonempty_pack_file(p)]
        if clashing:
            names = ", ".join(str(p) for p in clashing)
            raise DiagnosticError(Diagnostic(
                severity=ERROR,
                summary="refusing to overwrite existing pack files",
                detail=f"Already present and non-empty:\n    {names}\n\n"
                       f"Pass --force to overwrite, or point the output at an empty directory. "
                       f"(SUGGESTIONS.md is always rewritten.)",
            ))

    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps(suggestion.vocabulary(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    aliases_path.write_text(
        json.dumps(suggestion.aliases(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sugg_path.write_text(render_markdown(suggestion), encoding="utf-8")

    folded = sum(c.size for c in suggestion.songs) - len(suggestion.songs)
    print(f"{suggestion.raw_count} raw titles -> {len(suggestion.songs)} song candidates "
          f"({folded} variants folded in), {len(suggestion.non_songs)} probable non-songs")
    # Reported separately because the two numbers answer different questions: how many names a
    # setlist source got to SPELL, and how many songs would not be in the draft at all without it.
    if suggestion.curated:
        print(f"  {len(suggestion.curated)} named by a setlist source "
              f"({len(suggestion.backfilled)} of them on no tape at all)")
    for path in (vocab_path, aliases_path, sugg_path):
        print(f"  wrote {path}")
