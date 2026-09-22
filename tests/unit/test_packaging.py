"""Package data must actually ship.

The browser build installs a wheel and has no repository checkout, so anything
missing from package-data is silently absent there while working perfectly in a
development tree.
"""

from importlib.resources import files

import pytest

RESOURCES = files("gaia_dr4_explorer.resources")

REQUIRED = ("prerelease_reference.csv", "dr3_products.csv", "dr3_products_bundle.zip")


@pytest.mark.parametrize("name", REQUIRED)
def test_resource_is_present(name):
    assert (RESOURCES / name).is_file(), f"{name} is missing from package resources"


def test_resource_globs_cover_every_resource():
    """A new resource extension must be added to package-data, not just dropped in."""
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    globs = cfg["tool"]["setuptools"]["package-data"]["gaia_dr4_explorer.resources"]
    covered = {g.lstrip("*") for g in globs}
    on_disk = {
        p.suffix for p in (root / "src" / "gaia_dr4_explorer" / "resources").iterdir()
        if p.is_file() and p.suffix != ".py"
    }
    missing = on_disk - covered
    assert not missing, f"resource types not covered by package-data: {sorted(missing)}"


def test_reference_table_carries_everything_the_sky_model_needs():
    """A browser build cannot fit, so the model must be fully described here."""
    from gaia_dr4_explorer.data.catalog import reference_fits

    required = {
        "fit_delta_alpha_star_mas", "fit_delta_delta_mas", "fit_parallax_mas",
        "fit_pmra_mas_yr", "fit_pmdec_mas_yr",
        "ra0_deg", "dec0_deg", "t_rel_min_yr", "t_rel_max_yr",
    }
    fits = reference_fits()
    assert len(fits) == 12
    for sid, values in fits.items():
        missing = required - set(values)
        assert not missing, f"{sid} is missing {sorted(missing)} from the reference table"


def test_reference_geometry_is_physically_plausible():
    from gaia_dr4_explorer.data.catalog import reference_fits

    for sid, v in reference_fits().items():
        assert 0.0 <= v["ra0_deg"] <= 360.0, sid
        assert -90.0 <= v["dec0_deg"] <= 90.0, sid
        # The DR4 baseline is 2014-07 to 2020-01 about J2017.5.
        assert -3.0 < v["t_rel_min_yr"] < 0.0, sid
        assert 0.0 < v["t_rel_max_yr"] < 3.0, sid
