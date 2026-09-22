"""Exports for epoch astrometry.

CSV is offered because people ask for it, but it does not preserve Astropy
units, masks or column descriptions, so it is always labelled as lossy.
"""

from __future__ import annotations

import io
import json

from astropy.table import Table

#: format -> (file extension, MIME type, whether metadata survives)
FORMATS: dict[str, tuple[str, str, bool]] = {
    "votable": ("xml", "application/x-votable+xml", True),
    "parquet": ("parquet", "application/vnd.apache.parquet", True),
    "ecsv": ("ecsv", "text/plain", True),
    "csv": ("csv", "text/csv", False),
    "json": ("json", "application/json", True),
}

LOSSY_CSV_NOTE = (
    "CSV does not preserve Astropy units, masks or column descriptions. "
    "Use Parquet or ECSV to keep them."
)


def export_table(table: Table, fmt: str) -> bytes:
    """Serialise *table* to *fmt* and return the bytes."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; known: {sorted(FORMATS)}")
    buffer = io.BytesIO()
    if fmt == "parquet":
        table.to_pandas().to_parquet(buffer, index=False)
    elif fmt == "ecsv":
        text = io.StringIO()
        table.write(text, format="ascii.ecsv")
        buffer.write(text.getvalue().encode("utf-8"))
    elif fmt == "csv":
        text = io.StringIO()
        text.write(f"# {LOSSY_CSV_NOTE}\n")
        table.write(text, format="ascii.csv")
        buffer.write(text.getvalue().encode("utf-8"))
    elif fmt == "votable":
        table.write(buffer, format="votable")
    elif fmt == "json":
        buffer.write(json.dumps(table.to_pandas().to_dict(orient="records"),
                                indent=2, default=str).encode("utf-8"))
    buffer.seek(0)
    return buffer.getvalue()


def export_provenance(record) -> bytes:
    """Serialise a provenance record to JSON bytes."""
    return record.to_json().encode("utf-8")


def filename(source_id: int, release: str, what: str, fmt: str) -> str:
    ext = FORMATS[fmt][0]
    safe_release = release.replace(" ", "-").replace("/", "-")
    return f"{safe_release}_{source_id}_{what}.{ext}"
