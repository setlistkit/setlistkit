#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Tim Case <tim@lnx.cx>
#
# Every test function in the suite has a name nothing else uses. Lifted from bitmath, where the
# same check guards TestCase class names; the idiom here is a plain pytest function, so that is
# what this counts.
#
# TWO DIFFERENT FAILURES SHARE THIS ONE RULE, and the second is why the check exists at all:
#
#   SHADOWED   two `def test_x` in ONE file. Python keeps the later definition, pytest collects
#              one of the two, and the other STOPS RUNNING. Nothing is reported -- the count in
#              the green bar goes down by one and no test fails. This happened here: a
#              `[model.score]` test shadowed the `[model]` test it was named after, and it took
#              running pylint over tests/ for the first time to notice.
#   AMBIGUOUS  the same name in two files. Both still run, so nothing is lost, but a failure
#              report names a test that grep finds in two places, and the person reading it has
#              to work out which layer broke. Cheap to prevent, tedious to live with.
#
# Shadowing is also caught by pylint's function-redefined, which `make lint` now runs. Ambiguity
# is NOT -- pylint sees one module at a time and cannot know. That gap is this script's job.
#
# Deliberately grep and awk rather than Python: it runs before the venv exists and cannot itself
# depend on the thing it checks.

set -uo pipefail

root="${1:-tests}"

# file:line:def test_name for every test in the tree. -o keeps the match, so a decorated or
# oddly-wrapped signature cannot smuggle extra text into the name.
sites="$(grep -rnoE '^def test_[A-Za-z0-9_]+' "$root" --include='*.py' | sed 's/(.*$//')"

total="$(printf '%s\n' "$sites" | grep -c . || true)"
if [ "$total" -eq 0 ]; then
    echo "Error: no test functions found under ${root}/ -- the check found nothing to check."
    exit 1
fi

# Count what came out, not just whether anything errored: a check that silently examines zero
# names passes just as quietly as one that examines all of them.
dupes="$(printf '%s\n' "$sites" | sed 's/^.*:def /def /' | sort | uniq -d)"

if [ -z "$dupes" ]; then
    echo "OK: ${total} test names, all unique"
    exit 0
fi

status=0
while IFS= read -r name; do
    [ -n "$name" ] || continue
    # -E, not -F: a fixed-string grep treats the trailing $ as a literal dollar sign rather than
    # as an end anchor, so it matches nothing and every collision reports zero sites. The names
    # are [A-Za-z0-9_ ] by construction, so there is nothing here for a regex to misread.
    where="$(printf '%s\n' "$sites" | grep -E ":${name}$")"
    files="$(printf '%s\n' "$where" | cut -d: -f1 | sort -u | wc -l | tr -d ' ')"
    seen="$(printf '%s\n' "$where" | grep -c . || true)"
    if [ "$files" -lt "$seen" ]; then
        echo "SHADOWED: ${name#def } is defined more than once in one file -- one of these never runs"
    else
        echo "AMBIGUOUS: ${name#def } is used in ${files} files"
    fi
    printf '%s\n' "$where" | sed 's/^/    /'
    status=1
done <<EOF
$dupes
EOF

if [ "$status" -ne 0 ]; then
    echo
    echo "Error: duplicate test names found (${total} tests scanned)"
fi
exit "$status"
