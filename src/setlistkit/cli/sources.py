# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""What each configured source holds in the cache, read the same way by every command.

Two commands draft a pack vocabulary: ``bootstrap`` on a new deployment, and ``pack
pack draft`` on one under revision. Both need the same two inputs. Both must find them the
same way, or a redraft answers a different question than the draft it replaces.

The functions here are that one way. Each reads the cache alone and makes no network call.
"""

from __future__ import annotations

from ..catalog.normalizer import Normalizer
from ..catalog.suggest import titles_from_cache
from ..importers import configured, mirror, secondary
from ..store.raw_cache import RawCache
from .common import min_year


def mirror_titles(config) -> list[str]:
    """Return each distinct track title the mirror source holds in the cache.

    Return an empty list when the config declares no mirror source. Tape titles are one input to
    a draft, and a deployment with no tapes has none of them. The caller reports the absence.
    """
    importer = mirror(config)
    if importer is None:
        return []
    client = importer.make_client(config, RawCache(config.data_root))
    cached = client.cached_items(importer.identity(config),
                                 min_year=min_year(config, importer.id, None))
    return titles_from_cache(cached.items)


def curated_names(config) -> dict[str, list[str]]:
    """Return the song names each configured setlist source reports as played, keyed by source id.

    A setlist source publishes the names the band uses. Tape titles are what a taper typed into a
    filename. Both feed a draft, and only the published names may name a candidate.

    Read each source through its own ``candidates`` function, not off the raw blobs. The parsers
    already flag ``non_song`` entries. setlist.fm's ``tape`` marker names a song played over the
    PA, and that flag keeps it out. No rule here must know what a tape marker is.

    A configured source that gave nothing keeps its key with an empty list, so the caller can
    name it. A source can be configured, hold a full cache, and still give nothing, because one
    required setting is absent. That happened on a real config and cost 265 songs. A flat list
    cannot show it; this shape can.

    Sources this deployment did not configure are absent from the result. They contribute
    nothing, and they are not a fault to report.

    The normalizer is empty on purpose. No pack exists yet at draft time, because the pack is
    what a draft produces. Names thus arrive as each source spells them.
    """
    cache = RawCache(config.data_root)
    empty = Normalizer(vocabulary=[])
    have = {importer.id for importer in configured(config)}
    found: dict[str, list[str]] = {}
    for importer in (imp for imp in secondary() if imp.id in have):
        names: set[str] = set()
        for record in importer.candidates(config, cache, normalizer=empty,
                                          min_year=min_year(config, importer.id, None)) or ():
            for entries in list(record.get("sets") or ()) + [record.get("encore") or ()]:
                for entry in entries:
                    if not entry.get("non_song") and entry.get("song"):
                        names.add(str(entry["song"]).strip())
        found[importer.id] = sorted(names, key=str.casefold)
    return found


def report_inputs(titles: list[str], by_source: dict[str, list[str]], config) -> None:
    """Print what each draft input gave, and name any configured source that gave nothing.

    The report exists because the failure it exposes is silent. A source that gives nothing still
    lets the run finish and exit 0, on a corpus missing most of its shows.
    """
    print(f"  {len(titles):5} cached tape title(s) from the mirror")
    if not mirror(config):
        print("  no mirror source is configured, so this draft reads no tape titles")
    for source_id, names in by_source.items():
        print(f"  {len(names):5} published song name(s) from {source_id}"
              + ("   <- configured, gave nothing" if not names else ""))
    if any(not names for names in by_source.values()):
        print("  A configured source that gives nothing has a setting missing or a cache that\n"
              "  was never pulled. Run `slkit check` to see which.")


def flatten(by_source: dict[str, list[str]]) -> list[str]:
    """Every published name from every source, deduped and sorted for a stable draft."""
    return sorted({name for names in by_source.values() for name in names}, key=str.casefold)
