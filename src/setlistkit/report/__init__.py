"""Report: optional themeable dashboards and feeds, installed via ``setlistkit[report]``.

May import ``catalog``, ``model``, and ``picks``; nothing imports ``report``. The subpackage
always ships in the wheel, but its presentational dependencies (jinja2 and friends) are
declared under the ``report`` extra, so a headless install stays headless. Importing this
package without the extra installed will raise a diagnostic naming the missing extra rather
than a bare ImportError. (Populated in a later phase.)
"""
