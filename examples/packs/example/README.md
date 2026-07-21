# example pack

A pack is how you teach setlistkit about one band without touching its code. This directory
is the smallest one that still shows every moving part. Copy it somewhere of your own, point
`[catalog] pack` at the copy in your config, and start editing.

setlistkit ships no band data itself. The catalog is generic; the knowledge lives here, in
data you own.

## The files

- **`pack.json`** — identity. Name, version, and where the song list came from. `band_name` is
  what the band calls *itself*, which is not always the pack's name: archive.org puts it in the
  title of every item, so it is how a side project's tape gets told apart from yours. Leave it
  out and no band filter runs, because guessing at it is worse than not filtering.
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
- **`corpus.json`** — what your band's tapes get wrong, and the residue only your tapers write.
  Four keys, all optional, and **every entry has to say why**:
  - `drop_dates` — nights that are not evidence about this band: a tribute set, an all-star
    jam, a costume show whose "songs" were bits. You lose the real songs buried in that night
    too, so say what makes the trade worth it.
  - `date_overrides` — items filed under a date the show did not happen on. An uploader can
    type anything, and a well-formed lie is indistinguishable from the truth to every parser
    downstream, so an entry here carries the evidence rather than the hunch.
  - `junk_patterns` — cover artists and member surnames a taper writes where the song title
    belongs. These **drop** the entry outright rather than tagging it, because nothing was
    performed under that name.
  - `gear_patterns` — the tape-lineage shorthand your scene uses and the next one has never
    heard of. setlistkit ships the gear words every taper writes; the dialect is yours.

  A fragment here is never anchored — it is folded into an alternation and bounded by non-word
  edges — so unlike a classifier there is no bare-string form. Write the object, say the why.

## Check it

`slkit pack lint` validates the shape, proves every free-floating rule earns its keep, and
runs each rule's `must_not_match` and every protected title against the whole classifier set.
It also warns when a `corpus.json` fragment reaches a title you actually play — checked against
`vocabulary.json`, `aliases.json` and `protected.json` together, since all three are you saying
"this is a song". That one is a warning and not an error, because it cannot cost you the song:
no rule in setlistkit that drops an entry is allowed to delete a title the pack claims, and the
parser checks that before it applies any of them. It is still worth fixing, because a fragment
wide enough to reach a real title is wider than you meant, and the guard only knows about the
titles that are in your pack today.

Run it before you trust a pack you just edited.
