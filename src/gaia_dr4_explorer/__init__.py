"""Interactive explorer for Gaia DR4 epoch astrometry.

Importing this package must never start the application or perform network I/O.
"""

__version__ = "0.1.0.dev0"

#: Release string carried by the June-2026 prerelease VOTable.  It is a release
#: *candidate*, so its ``source_id`` values are not guaranteed to survive to the
#: public Gaia DR4.
PRERELEASE_RELEASE = "Gaia DR4_RC3"

__all__ = ["__version__", "PRERELEASE_RELEASE"]
