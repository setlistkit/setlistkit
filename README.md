# setlistkit

A generic, open-source setlist prediction toolkit. It ingests setlists from multiple
sources, builds a catalog of songs, segues, families, and durations, and predicts what a
band will play next. moe. is the first dataset it targets, not the limit of what it does.

> **Status: early development.** The architecture is settled and the scaffold is in place;
> the ingest, catalog, model, and reporting layers are being built out phase by phase.

## Design principles

- **State lives outside the repository.** All mutable state sits under a `data_root` you
  point at from config. Nothing derived is committed.
- **Band knowledge lives in data, not code.** A *band pack* — a directory of JSON governed
  by a schema — carries the vocabulary, aliases, and classifier rules. Code is the last
  resort.
- **The catalog stands alone.** Someone who wants a jam-band song graph and no prediction
  can install setlistkit and use the `catalog` layer by itself. The layering is enforced by
  a test, not by convention.
- **Be a good network citizen.** Every source client identifies itself with a mandatory
  User-Agent, is cached and rate-limited, and backs off on error. setlistkit ships no
  scraper that violates a source's terms.

## Install

setlistkit develops against Python 3.14 and supports 3.11+ (it uses the standard-library
`tomllib`). The reporting extra pulls in the presentational dependencies:

```sh
pip install setlistkit           # core: catalog, model, picks
pip install 'setlistkit[report]' # adds the themeable dashboards and feeds
```

## Configure

Copy `slkit.example.toml` to `slkit.toml` and edit it. At minimum you must set `data_root`
and change `user_agent` from its placeholder — until you do, any command that would touch
the network refuses to run.

```sh
slkit config show    # print the resolved configuration
slkit config check   # validate it, including network identity
```

## License

**AGPL-3.0-or-later** (the full `LICENSE` text lands with the first release). This project
is made possible by the people who transcribe setlists and tape shows, and a network
-reaching copyleft keeps it available to them: nobody gets to take a hosted fork private.

A setlist — the songs played, their order, segues, and encore — is a human-entered fact
about a public performance, and setlistkit stores those. It deliberately does **not** depend
on the derived aggregates (play-frequency averages, historical base rates) that a tracker
computes on top of them; those are the tracker's IP.
