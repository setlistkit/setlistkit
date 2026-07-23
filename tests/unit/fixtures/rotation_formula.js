// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Tim Case <tim@lnx.cx>
//
// Maintained COPY of the rotation arithmetic inside famoe.ly/bin/songbook/logic.js's
// aggregateWindow() -- the gap / meanGap / due lines, reduced to just that arithmetic so it
// can run headless under node with no DOM. setlistkit's test suite does not check out
// famoe.ly, so this file -- not the tracked source -- is what
// tests/unit/test_rotation_parity.py actually executes.
//
// IF dueRatio() CHANGES, THIS FILE MUST CHANGE WITH IT. Nothing here will notice on its own
// if they drift apart; that is the whole limit this file exists to be honest about, recorded
// in docs/plans/2026-07-22-songbook-slice-4-dry-plan.md (Task 4) and in the formula published
// in src/setlistkit/catalog/songbook.py's module docstring, which is what BOTH this file and
// the real logic.js are meant to match.
//
// Protocol: reads a JSON array from stdin, each item {n, lastIdx, gaps, fallback}. Writes a
// JSON array of {gap, meanGap, ratio} to stdout, one per input item, in order.

// Named aggregateWindow() because it reduces a whole window's worth of gaps for one song
// (n, lastIdx, gaps) to a result, the way the real aggregateWindow() does -- but the drift
// this file exists to catch is in the ratio step, dueRatio(), which is the one formula all
// three implementations (this copy, logic.js, and setlistkit.model.scores) actually share.
function aggregateWindow(n, lastIdx, gaps, fallback) {
  const gap = n - 1 - lastIdx;
  const meanGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : n;
  const ratio = meanGap > 0 ? gap / meanGap : fallback;  // dueRatio(gap, meanGap)
  return { gap, meanGap, ratio };
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => { input += chunk; });
process.stdin.on('end', () => {
  const cases = JSON.parse(input);
  const results = cases.map(c => aggregateWindow(c.n, c.lastIdx, c.gaps, c.fallback));
  process.stdout.write(JSON.stringify(results));
});
