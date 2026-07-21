# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""The raw source cache: fetch results kept as plain files under ``<data_root>/raw/``.

Raw snapshots from archive.org, setlist.fm, and Instagram stay as files, not database
rows, for one reason worth keeping: when a parser misbehaves you want to open the exact
bytes it choked on in a text editor. Each payload gets a sidecar ``.meta.json`` recording
where it came from and when, so the cache is self-describing without the database.

The cache stores and reports; it does not decide freshness. A source client asks for the
:meth:`age` of an entry and applies its own staleness policy, and ``--force-rescan`` is just
the client choosing to ignore a hit. Nothing here ever reaches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Beyond this, a url-quoted key stops being a readable filename and starts being a liability
# (path-length limits), so we hash instead. The original key always lives in the sidecar.
_MAX_STEM = 120

# Namespaces are our own source ids ("archive_org", "setlistfm"). Fence them so a stray "/"
# or ".." can never walk the payload out of the cache root.
_NS = re.compile(r"^[a-z0-9_]+$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(when: datetime) -> datetime:
    """Treat a naive datetime as UTC. HTTP Date headers hand us both kinds, and mixing an
    aware `now` with a naive `fetched_at` is a TypeError waiting to happen in age()."""
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _ns(namespace: str) -> str:
    if not _NS.match(namespace):
        raise ValueError(f"bad cache namespace {namespace!r}; expected [a-z0-9_]+")
    return namespace


def _safe_stem(key: str) -> str:
    quoted = urllib.parse.quote(key, safe="")
    if len(quoted) <= _MAX_STEM:
        return quoted
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp file and rename, so a reader never sees a half-written payload."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


@dataclass(frozen=True)
class NamespaceStats:
    """How much of one source is cached. Counted off the filesystem, never the network."""

    entries: int = 0
    bytes: int = 0


class RawCache:
    """A namespaced, on-disk cache of raw fetch results.

    ``namespace`` keeps sources from colliding (an archive.org identifier and a setlist.fm
    id can be the same string); ``key`` identifies the item within a source.
    """

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root) / "raw"

    def _payload_path(self, namespace: str, key: str) -> Path:
        # Payloads and sidecars live in separate dirs so no key's payload name can ever land
        # on another key's metadata path (a key of "foo.meta.json" used to do exactly that).
        return self.root / _ns(namespace) / "blob" / _safe_stem(key)

    def _meta_path(self, namespace: str, key: str) -> Path:
        return self.root / _ns(namespace) / "meta" / (_safe_stem(key) + ".json")

    def has(self, namespace: str, key: str) -> bool:
        """Whether this entry's payload is cached."""
        return self._payload_path(namespace, key).is_file()

    def put(self, namespace: str, key: str, data: bytes, *,
            url: str | None = None, content_type: str | None = None,
            etag: str | None = None, last_modified: str | None = None,
            fetched_at: datetime | None = None) -> Path:
        """Write the payload and its sidecar. Returns the payload path.

        ``etag``/``last_modified`` are stored for a later conditional re-fetch; ``put`` does
        no HTTP itself. ``fetched_at`` is injectable so tests can pin an age.
        """
        payload = self._payload_path(namespace, key)
        payload.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(payload, data)

        meta = {
            "key": key,
            "url": url,
            "content_type": content_type,
            "etag": etag,
            "last_modified": last_modified,
            "fetched_at": _as_utc(fetched_at or _now()).isoformat(timespec="seconds"),
            "bytes": len(data),
        }
        meta_path = self._meta_path(namespace, key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(meta_path, (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return payload

    def get(self, namespace: str, key: str) -> bytes | None:
        """The cached payload bytes, or ``None`` if not cached."""
        payload = self._payload_path(namespace, key)
        return payload.read_bytes() if payload.is_file() else None

    def get_text(self, namespace: str, key: str, encoding: str = "utf-8") -> str | None:
        """The cached payload decoded as text, or ``None`` if not cached."""
        data = self.get(namespace, key)
        return None if data is None else data.decode(encoding)

    def meta(self, namespace: str, key: str) -> dict | None:
        """The sidecar metadata for an entry, or ``None`` if not cached."""
        meta_path = self._meta_path(namespace, key)
        if not meta_path.is_file():
            return None
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def age(self, namespace: str, key: str, *, now: datetime | None = None) -> timedelta | None:
        """How long since this entry was fetched, or ``None`` if it isn't cached.

        The freshness question a source client asks before deciding to re-hit an API.
        """
        meta = self.meta(namespace, key)
        if not meta or not meta.get("fetched_at"):
            return None
        fetched = _as_utc(datetime.fromisoformat(meta["fetched_at"]))
        return _as_utc(now or _now()) - fetched

    def stats(self) -> dict[str, NamespaceStats]:
        """Entries and payload bytes per namespace, alphabetized. Touches no network.

        Here so "how far has the pull got" has an answer that costs nothing. ``slkit pull -n``
        answers it too and answers it better -- it knows how many items remain, not just how many
        are held -- but it spends a handful of search requests to do it, which makes it the wrong
        thing to poll with. This is a directory listing.

        ``.tmp`` files are skipped. An atomic write lands as ``<name>.tmp`` and is renamed, so a
        payload being written right now would otherwise be counted as an entry that does not
        exist yet, and the count would flicker upward and back during a pull.
        """
        out: dict[str, NamespaceStats] = {}
        if not self.root.is_dir():
            return out
        for namespace in sorted(path.name for path in self.root.iterdir() if path.is_dir()):
            blobs = self.root / namespace / "blob"
            if not blobs.is_dir():
                continue
            entries, total = 0, 0
            for payload in blobs.iterdir():
                if payload.is_file() and not payload.name.endswith(".tmp"):
                    entries += 1
                    total += payload.stat().st_size
            out[namespace] = NamespaceStats(entries=entries, bytes=total)
        return out

    def delete(self, namespace: str, key: str) -> bool:
        """Drop an entry and its sidecar. Returns whether anything was there."""
        existed = False
        for path in (self._payload_path(namespace, key), self._meta_path(namespace, key)):
            if path.is_file():
                path.unlink()
                existed = True
        return existed
