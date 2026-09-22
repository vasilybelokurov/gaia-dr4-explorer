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
