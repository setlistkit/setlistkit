# example pack

A pack is how you teach setlistkit about one band without touching its code. This directory
is the smallest one that still shows every moving part. Copy it somewhere of your own, point
`[catalog] pack` at the copy in your config, and start editing.

setlistkit ships no band data itself. The catalog is generic; the knowledge lives here, in
data you own.

## The files

- **`pack.json`** — identity. Name, version, and where the song list came from.
- **`vocabulary.json`** — the canonical song list, your dictionary. Canonicalization maps
  messy taper spellings onto these names. Ship names only, never play-counts (those are a
  tracker's derived data, not yours to redistribute).
- **`aliases.json`** — a flat map from a normalized spelling to the canonical name, for the
  cases fuzzy matching can't reach ("long one" is nowhere near "The Long One" as a string).
  Keys are written normalized; the loader normalizes them again anyway, so a readable key is
  fine.
- **`classifiers.json`** — the rules that say an entry is NOT a song (a setbreak, a
  soundcheck). An **anchored** pattern (`^setbreak$`, `foo$`) ships as a bare string. A
  **free-floating** substring can reach into a real title, so it MUST justify itself: ship it
  as an object with a `why`, and a `must_not_match` list of real songs it must leave alone.
- **`protected.json`** — titles that are real songs even though a shape rule might mistake
  them for junk. A protected title is always a song. This list is what stops an over-eager
  rule from silently deleting one.

## Check it

`slkit pack lint` validates the shape, proves every free-floating rule earns its keep, and
runs each rule's `must_not_match` and every protected title against the whole classifier set.
Run it before you trust a pack you just edited.
