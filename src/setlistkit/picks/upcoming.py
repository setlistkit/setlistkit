# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
"""Nights that have not happened, predicted with nothing to grade them against.

WHY THERE IS A SCHEDULE HERE WHEN :mod:`setlistkit.picks.drops` REFUSED ONE. That refusal stands
and this does not reverse it. The argument there was that GRADING needs no schedule: the corpus
already holds every night that happened and what was played on it, so a walk can blind itself at a
cutoff, predict the next night, and check its own answer. Acquiring a schedule to do that would
have bought a source, a table and four parser changes to learn dates the corpus already knew.

A night that has not happened is the one case the corpus cannot supply, because the fact being
asked for -- that the band will be somewhere on 2026-09-04 -- is not in it and never will be until
afterwards. So the schedule here answers exactly one question, WHICH DATES TO PREDICT, and it is
not training data: it names no songs, reaches no scorer, and cannot move a ranking. The leak
boundary is unchanged and is still :func:`~setlistkit.picks.drops.predict_drop`.

WHAT AN UPCOMING NIGHT CANNOT HAVE, and why the page has to say so rather than quietly rendering
it the same as the rest:

* **No outcome.** Nothing has been played, so there is nothing to grade. ``outcome`` is ``None``
  and ``played`` is empty -- the shapes :class:`~setlistkit.picks.drops.NightDrops` already allows.
* **No known set length.** A scored night tells the model how many songs to draw, taken from what
  was actually played. Nobody knows that in advance, so this passes the configured default. That
  makes a forecast card a HARDER prediction than any scored card beside it, and the two are not
  comparable on recall for that reason alone.

EVERY UPCOMING NIGHT SHARES ITS CUTOFFS, which surprises people and is correct. Cutoffs are played
shows, no show happens between two future dates, so the last twelve gradeable nights are the last
twelve for all of them. The forecasts still differ, because ``asof`` moves: a night in January 2027
is scored with five more months of decay against the same evidence than a night next week. What
the per-night spark shows, then, is the model's picture of THAT date firming up as the past filled
in -- not new information arriving between the dates.

Nothing here is written to the ledger. The ledger's job is the record-then-grade round trip, and a
night with no outcome has nothing to complete it; a forecast is recomputed from the corpus every
build instead. Persisting "what we said beforehand" is worth doing and is a different feature with
a different shape.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..model.backtest import gradeable
from .drops import DEFAULT_DEPTH, NightDrops, predict_drop
from .replay import ReplaySpec

# A record in a phish.net /upcoming copy-paste: artist, weekday and date, venue, then "City, ST".
# Four lines, no separator, which is the whole reason this is validated rather than chunked --
# see :func:`parse_tour`.
_RECORD_LINES = 4

# "Saturday, 07/25/2026". The weekday is carried in the source and deliberately not trusted: it is
# redundant with the date, and a copy-paste that disagrees with itself should not get a vote.
_DATED = re.compile(r"^[A-Za-z]+,\s*(\d{2})/(\d{2})/(\d{4})$")

# The JSON form: what a TourDate needs, and the reserved name the file carries inside a pack.
_JSON_FIELDS = ("artist", "date", "venue", "city")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SCHEDULE_NAME = "tour.json"


@dataclass(frozen=True)
class TourDate:
    """One announced show. ``date`` is ISO so it sorts and compares against corpus dates."""

    date: str
    artist: str
    venue: str
    city: str


@dataclass(frozen=True)
class TourSchedule:
    """Everything the file said, and everything it could not be read as.

    ``problems`` exists because the format has no separator between records. Four lines is not a
    parse, it is an assumption, and a file that gains a line somewhere in the middle would silently
    reinterpret every record after it -- venues becoming artists, cities becoming venues, dates
    landing on the wrong show. Reported rather than dropped, so a schedule that half-parsed says so
    instead of rendering a confidently wrong board.
    """

    dates: tuple[TourDate, ...]
    problems: tuple[str, ...]

    def by_artist(self, artist: str) -> tuple[TourDate, ...]:
        """Only this act's shows. Matched case-insensitively on the whole name.

        Not optional. A tour listing carries side projects -- the file this was built against holds
        a Bob Wagner date and a Mike Gordon date among the Phish ones -- and predicting a Phish
        setlist for somebody else's gig would be a confident answer to a question nobody asked.
        """
        wanted = artist.strip().casefold()
        return tuple(row for row in self.dates if row.artist.casefold() == wanted)


def parse_tour(text: str) -> TourSchedule:
    """Read a phish.net /upcoming listing pasted to a file.

    Blank lines and ``#`` comments are skipped, then what remains is taken four lines at a time.
    The second line of each record MUST look like a date, and that check is what makes the chunking
    safe: if the file drifts out of step the dates stop landing in slot two, every subsequent record
    is refused by name, and the caller sees a list of complaints rather than a plausible schedule
    assembled out of venue names.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not line.startswith("#")]
    dates: list[TourDate] = []
    problems: list[str] = []
    for start in range(0, len(lines), _RECORD_LINES):
        record = lines[start:start + _RECORD_LINES]
        if len(record) < _RECORD_LINES:
            problems.append(f"trailing {len(record)} line(s) are not a whole record: "
                            f"{' / '.join(record)}")
            continue
        artist, when, venue, city = record
        matched = _DATED.match(when)
        if not matched:
            problems.append(f"line {start + 2} of the records is not a date: {when!r} "
                            f"(record began {artist!r})")
            continue
        month, day, year = matched.groups()
        dates.append(TourDate(date=f"{year}-{month}-{day}", artist=artist, venue=venue, city=city))
    return TourSchedule(dates=tuple(dates), problems=tuple(problems))


def parse_json_tour(text: str) -> TourSchedule:
    """Read the file ``scripts/tour_dates.py`` writes: ``{"dates": [{artist, date, venue, city}]}``.

    An entry missing any of the four is reported rather than filled in. The extra keys that file
    carries -- the provider's id, and the note a reviewer reads -- are ignored here.
    """
    payload = json.loads(text)
    rows = payload.get("dates") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return TourSchedule(dates=(), problems=("no 'dates' list in this file",))
    dates: list[TourDate] = []
    problems: list[str] = []
    for index, row in enumerate(rows):
        fields = {key: str(row.get(key) or "").strip() for key in _JSON_FIELDS} \
            if isinstance(row, Mapping) else {}
        missing = [key for key in _JSON_FIELDS if not fields.get(key)]
        if missing:
            problems.append(f"entry {index} is missing {', '.join(missing)}")
            continue
        if not _ISO_DATE.match(fields["date"]):
            problems.append(f"entry {index} has no ISO date: {fields['date']!r}")
            continue
        dates.append(TourDate(**fields))
    return TourSchedule(dates=tuple(dates), problems=tuple(problems))


def schedule_path(pack_dir) -> Path:
    """Where the announced-shows file lives. A reserved name in the pack, not a setting.

    The file is reviewed band data and belongs beside the vocabulary, so the writer and every
    reader derive one path from the pack rather than agreeing on a configured one.
    """
    return Path(pack_dir) / SCHEDULE_NAME


def after(dates: Sequence[TourDate], last_show: str) -> tuple[TourDate, ...]:
    """Only the shows still ahead of the corpus, oldest first.

    Cut against the corpus's own last show rather than against today's date, so a build is a pure
    function of the store. Using ``date.today()`` would make the same store render a different
    board depending on when the command ran, which is the failure `[reports.*.window]` exists to
    end -- and it would put a night the corpus already holds into the forecast strip on any day the
    ingest had not caught up.
    """
    return tuple(sorted((row for row in dates if row.date > last_show), key=lambda r: r.date))


def forecast(corpus: Sequence[Mapping], spec: ReplaySpec, dates: Sequence[TourDate], *,
             depth: int = DEFAULT_DEPTH) -> tuple[NightDrops, ...]:
    """Predict each announced night from the last ``depth`` played shows behind it.

    Returns the same :class:`~setlistkit.picks.drops.NightDrops` a scored night produces, with
    ``outcome`` ``None`` and ``played`` empty, so a page can render both from one code path and has
    to opt IN to showing an answer rather than remembering to suppress one.
    """
    cutoffs = [night for night, _songs in gradeable(corpus, spec.min_songs)][-depth:]
    if not cutoffs:
        return ()
    return tuple(
        NightDrops(
            night=row.date,
            played=(),
            # The set length nobody can know yet. See the module docstring: this is the one input a
            # scored night gets for free and a forecast has to assume.
            drops=tuple(predict_drop(corpus, spec, row.date, cutoff, spec.predict.set_len)
                        for cutoff in cutoffs),
            outcome=None)
        for row in dates)


def read_schedule(path: str, artist: str, last_show: str) -> tuple[tuple[TourDate, ...],
                                                                   tuple[str, ...]]:
    """The announced shows for one act that the corpus has not reached, plus any parse complaints.

    The three filters in the order that makes each one cheap to explain: read the file, keep the
    act, drop what already happened.

    Two formats, told apart by suffix. ``.json`` is what ``scripts/tour_dates.py`` writes; anything
    else is a pasted phish.net listing, which stays readable because that is how one band's
    schedule is still gathered.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    read = parse_json_tour if str(path).endswith(".json") else parse_tour
    schedule = read(text)
    return after(schedule.by_artist(artist), last_show), schedule.problems


def as_of(dates: Sequence[TourDate]) -> dt.date | None:
    """The first announced date, or ``None``. Used only to caption the strip."""
    return dt.date.fromisoformat(dates[0].date) if dates else None
