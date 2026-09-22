"""Shared fixtures.

The real prerelease archive is committed under ``tests/fixtures`` so the
integration tests never need the network.  Unit tests build tiny synthetic
VOTables instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Column, MaskedColumn, Table

from gaia_dr4_explorer.config import AppConfig

FIXTURES = Path(__file__).parent / "fixtures"
PRERELEASE_ZIP = FIXTURES / "prerelease.zip"
PRERELEASE_SHA256 = "07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b"


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """Config pointing at a temporary cache and the committed fixture archive."""
    return AppConfig(
        cache_dir=tmp_path / "cache",
        local_prerelease_zip=PRERELEASE_ZIP,
        allow_network=False,
    )


@pytest.fixture
def offline_config(tmp_path) -> AppConfig:
    """Config with no local archive and no network."""
    return AppConfig(cache_dir=tmp_path / "cache", allow_network=False)


def synthetic_table(
    *,
    n_transits: int = 3,
    n_ccd: int = 10,
    source_ids: list[int] | None = None,
    mask_parallax_factor: bool = False,
    empty_ac: bool = True,
    ragged: bool = False,
) -> Table:
    """Build a small epoch-astrometry table with the published column names."""
    ids = source_ids or [4318465066420528000] * n_transits
    n = len(ids)
    rng = np.random.default_rng(7)

    def _cells(values):
        """1-D object array of arrays, as astropy produces for arraysize='*'."""
        out = np.empty(len(values), dtype=object)
        for i, v in enumerate(values):
            out[i] = v
        return out

    def arr(dtype, scale=1.0, unit=None):
        cells = []
        for i in range(n):
            size = n_ccd - 3 if (ragged and i == 0) else n_ccd
            if np.issubdtype(np.dtype(dtype), np.integer):
                cells.append(np.arange(size, dtype=dtype))
            elif np.dtype(dtype) == bool:
                cells.append(np.ones(size, dtype=bool))
            else:
                cells.append((rng.normal(size=size) * scale).astype(dtype))
        return Column(_cells(cells), dtype=object, unit=unit)

    t = Table()
    t["solution_id"] = Column(np.full(n, 2888461026632663040, dtype="int64"))
    t["source_id"] = Column(np.asarray(ids, dtype="int64"))
    t["transit_id"] = Column(np.arange(n, dtype="int64") * 100003 + 248777679747269187)
    t["ra0"] = Column(np.full(n, 294.8278, dtype="float64"), unit="deg")
    t["dec0"] = Column(np.full(n, 14.9309, dtype="float64"), unit="deg")
    t["agis_source_excess_noise"] = Column(np.full(n, 6.5829, dtype="float32"), unit="mas")
    base = 248777679747269187
    t["obs_time_tcb"] = Column(
        _cells([np.arange(n_ccd, dtype="int64") * 4_900_000_000 + base + i * 10**13
                for i in range(n)]), dtype=object, unit="ns",
    )
    t["obs_time_bary_corr"] = Column(np.full(n, 1.0e6, dtype="float32"), unit="ns")
    t["scan_pos_angle"] = arr("float64", 90.0, unit="deg")
    t["parallax_factor_al"] = (
        MaskedColumn(np.full(n, 0.5, dtype="float32"),
                     mask=[True] + [False] * (n - 1))
        if mask_parallax_factor
        else Column(np.full(n, 0.5, dtype="float32"))
    )
    t["centroid_pos_al"] = arr("float64", 0.2, unit="mas")
    t["centroid_pos_error_al"] = arr("float32", 0.05, unit="mas")
    t["used_by_agis_al"] = arr(bool)
    t["ccd_proc_flags"] = arr("int16")
    t["gates"] = arr("int16")
    t["transit_proc_flags"] = Column(np.full(n, -32208, dtype="int16"))
    t["transit_acq_flags"] = Column(np.full(n, 8240, dtype="int16"))
    t["multipeak"] = Column(np.zeros(n, dtype=bool))
    t["blended"] = Column(np.zeros(n, dtype=bool))
    t["g_mag"] = Column(np.full(n, 11.231, dtype="float32"), unit="mag")
    if empty_ac:
        t["centroid_pos_ac"] = Column(
            _cells([np.array([], dtype="float64") for _ in range(n)]),
            dtype=object, unit="mas",
        )
    return t
