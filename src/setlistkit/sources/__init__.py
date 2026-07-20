"""Sources: pluggable ingest from archive.org, setlist.fm, Instagram, and others.

Every source client is polite by construction — cached, rate-limited, backing off on
error, and sending a mandatory identifying User-Agent taken from config. A source is one
input among several and a sanity check, never a hard dependency on a tracker's derived
aggregates. (Populated in a later phase.)
"""
