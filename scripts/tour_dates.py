#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Fetch announced tour dates into a reviewable JSON file.

    pyenv/bin/python scripts/tour_dates.py --config moe-slkit.toml

Reads ``[tour]`` from the config, asks that provider for the artist's announced shows, and
appends the ones it has not offered before to the file's ``dates`` list. Nothing is filtered
and nothing is judged: a tour listing carries side projects, benefit shows and multi-day
festival runs, and no rule this script could apply tells them from a full-band night. Read the
file and delete what does not belong.

A deleted entry stays deleted. Every id this script has ever offered is recorded in ``seen``,
which is what it checks before appending -- so a rerun after a review does not resurrect the
entry that review removed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# pylint: disable=wrong-import-position
from setlistkit.config import load_config, require_network_identity   # noqa: E402
from setlistkit.diagnostics import DiagnosticError                    # noqa: E402
from setlistkit.sources.client import TransportError, urllib_transport  # noqa: E402

SEATED_API = "https://cdn.seated.com/api/tour"


def _seated(artist_id: str, user_agent: str) -> tuple[str, list[dict]]:
    """The public widget endpoint: the artist's name, and one record per announced event."""
    url = f"{SEATED_API}/{artist_id}?include=tour-events"
    # JSON:API, and strict about it: an Accept of application/json comes back 406.
    headers = {"User-Agent": user_agent, "Accept": "application/vnd.api+json"}
    response = urllib_transport(url, headers)
    if response.status != 200:
        raise SystemExit(f"seated: HTTP {response.status} for artist {artist_id}")
    payload = json.loads(response.body)
    artist = str(payload.get("data", {}).get("attributes", {}).get("name") or "").strip()
    if not artist:
        raise SystemExit(f"seated: no artist name in the response for {artist_id}")

    dates = []
    for item in payload.get("included", []):
        if item.get("type") != "tour-events":
            continue
        attrs = item["attributes"]
        # starts-at is UTC and disagrees with the local date on any late show: one moe. date
        # carries 2026-08-16T00:00:00Z for a night played on the 15th.
        start = attrs.get("starts-at-date-local")
        if not start:
            continue
        entry = {
            "artist": artist,
            "date": start,
            "venue": str(attrs.get("venue-name") or "").strip(),
            "city": str(attrs.get("formatted-address") or "").strip(),
            "id": item["id"],
        }
        review = _review_note(attrs, start)
        if review:
            entry["review"] = review
        dates.append(entry)
    return artist, dates


def _review_note(attrs: dict, start: str) -> str:
    """What the provider said that a reviewer needs to see. Reported, never acted on."""
    notes = []
    end = attrs.get("ends-at-date-local")
    if end and end != start:
        notes.append(f"listing runs through {end}")
    details = str(attrs.get("details") or "").strip()
    if details:
        notes.append(details)
    return " -- ".join(notes)


PROVIDERS = {"seated": _seated}


def _load_file(path: Path) -> dict:
    """The existing file, or an empty one. A file that is not an object is an error, not a start."""
    if not path.exists():
        return {"dates": [], "seen": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("dates"), list):
        raise SystemExit(f"{path}: expected an object with a 'dates' list")
    data.setdefault("seen", [])
    return data


def _resolve_target(config, section: dict) -> Path:
    """Where the file lives. A relative path is anchored at the config, never the cwd."""
    name = str(section.get("file") or "tour.json")
    path = Path(name).expanduser()
    return path if path.is_absolute() else (config.source_path.parent / path).resolve()


def _settings(config) -> tuple[str, str]:
    """The provider and artist id from ``[tour]``, or an error naming which one is missing."""
    section = config.section("tour")
    if not section:
        raise SystemExit("no [tour] table in this config; see slkit.example.toml")
    artist_id = str(section.get("artist_id") or "").strip()
    if not artist_id:
        raise SystemExit("[tour] has no artist_id, so there is nothing to ask for")
    provider = str(section.get("provider") or "").strip()
    if provider not in PROVIDERS:
        known = ", ".join(sorted(PROVIDERS))
        raise SystemExit(f"[tour] provider {provider!r} is not one this build knows: {known}")
    return provider, artist_id


def main(argv=None) -> int:
    """Fetch, append what is new, and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", metavar="PATH", help="path to a config file holding [tour]")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be added and write nothing")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        require_network_identity(config)
    except DiagnosticError as err:
        print(err, file=sys.stderr)
        return 1

    provider, artist_id = _settings(config)
    try:
        artist, offered = PROVIDERS[provider](artist_id, config.user_agent)
    except TransportError as err:
        print(f"{provider}: {err}", file=sys.stderr)
        return 1

    target = _resolve_target(config, config.section("tour"))
    data = _load_file(target)
    seen = set(data["seen"])
    fresh = [row for row in offered if row["id"] not in seen]
    fresh.sort(key=lambda row: row["date"])

    print(f"{provider}: {artist} -- {len(offered)} announced, {len(fresh)} not offered before")
    for row in fresh:
        note = f"  [{row['review']}]" if row.get("review") else ""
        print(f"  + {row['date']}  {row['venue']} · {row['city']}{note}")
    if args.dry_run:
        print(f"dry run: {target} not written")
        return 0

    data["dates"].extend(fresh)
    data["seen"] = sorted(seen | {row["id"] for row in offered})
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {target}: {len(data['dates'])} date(s) kept, {len(data['seen'])} ever offered")
    if fresh:
        print("review the new entries and delete any that are not full-band shows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
