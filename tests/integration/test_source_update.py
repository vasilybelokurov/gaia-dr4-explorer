"""Acceptance tests for the DR4-like source update (plan section 9, T2-T7)."""

import csv

import numpy as np
import pytest
from tests.conftest import FIXTURES, PRERELEASE_ZIP

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.products.astrometry.fitting import fit_dr4_like_single_source

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    cfg = AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"),
        local_prerelease_zip=PRERELEASE_ZIP,
        allow_network=False,
    )
    return PreReleaseProvider(cfg)


@pytest.fixture(scope="module")
def reference():
    with (FIXTURES / "prerelease_reference.csv").open() as fh:
        return {int(r["source_id"]): r for r in csv.DictReader(fh)}


@pytest.fixture(scope="module")
def fits(provider, reference):
    return {
        sid: fit_dr4_like_single_source(provider.raw_table_for(sid), sid)
        for sid in provider.source_ids()
    }


def test_t2_parallax_matches_the_release_page(fits, reference):
    """The page quotes one decimal place; assert exactly that, and no more."""
    for sid, result in fits.items():
        ref = reference[sid]
        fitted = result.as_dict()["parallax"][0]
        page = float(ref["parallax_page_mas"])
        if ref["sample_category"] == "qso":
            # No detectable parallax is a physical expectation, not a rounding.
            assert abs(fitted) <= 0.01, f"{sid}: QSO parallax {fitted:+.5f} mas"
        else:
            assert round(fitted, 1) == page, f"{sid}: fitted {fitted:.5f}, page {page}"


def test_t3_excess_noise_separates_the_orbit_sample(provider, reference):
    for sid, ref in reference.items():
        raw = provider.raw_table_for(sid)
        noise = float(np.asarray(raw["agis_source_excess_noise"])[0])
        if ref["sample_category"] == "orbit":
            assert noise >= 0.1, f"{sid}: orbit source has excess noise {noise}"
        elif ref["sample_category"] in ("parallax", "magnitude"):
            assert noise == 0.0, f"{sid}: expected zero excess noise, got {noise}"


def test_t4_f2_consistency(fits, reference):
    """The package F2 excludes the excess noise the fit was weighted by."""
    for sid, result in fits.items():
        assert abs(result.f2_total_variance) < 2.5, (
            f"{sid}: F2 against total variance is {result.f2_total_variance:.3f}"
        )
        if reference[sid]["sample_category"] == "orbit":
            assert result.f2_measurement_variance > 25, (
                f"{sid}: expected the package F2 artefact to be large, "
                f"got {result.f2_measurement_variance:.3f}"
            )


def test_t5_regression_snapshot(fits, reference):
    for sid, result in fits.items():
        ref = reference[sid]
        params = result.as_dict()
        assert result.n_measurements == int(ref["fit_n_measurements"])
        for name, column in (
            ("parallax", "fit_parallax_mas"),
            ("pmra_star", "fit_pmra_mas_yr"),
            ("pmdec", "fit_pmdec_mas_yr"),
        ):
            got = params[name][0]
            want = float(ref[column])
            assert got == pytest.approx(want, rel=1e-5, abs=1e-9), f"{sid}.{name}"


def test_t6_colour_factor_conversion_is_not_bypassed(provider):
    """A naive camelCase rename skips colourFactorAl *= -1e3 and degrades QSOs."""
    from gaiasupdate.epoch_astrometry import GaiaSourceEpochAstrometryCu9

    sid = 2237987199365376  # QSO: true parallax indistinguishable from zero
    raw = provider.raw_table_for(sid)
    official = fit_dr4_like_single_source(raw, sid).as_dict()["parallax"][0]

    def camel(s):
        head, *rest = s.split("_")
        return head + "".join(w.capitalize() for w in rest)

    df = raw.to_pandas().rename(columns=lambda c: camel(c))
    naive = GaiaSourceEpochAstrometryCu9(
        df[df.sourceId == sid].copy(), source_id=sid, explode=True
    ).compute_source_parameters_like_dr4()["parameters"][2]

    assert abs(official) <= 0.01
    assert abs(naive - official) > 0.01, (
        "the naive path should be measurably worse; if this fails the upstream "
        "colour-factor conversion may have moved"
    )


def test_t7_input_is_not_mutated_and_the_fit_repeats(provider):
    sid = 4318465066420528000
    raw = provider.raw_table_for(sid)
    df = raw.to_pandas()
    before = df.copy(deep=True)

    first = fit_dr4_like_single_source(df, sid)

    assert list(df.columns) == list(before.columns)
    assert df.shape == before.shape
    assert df.index.equals(before.index)
    assert (df.dtypes == before.dtypes).all()
    for col in df.columns:
        a, b = df[col].to_numpy(), before[col].to_numpy()
        if a.dtype == object:
            for x, y in zip(a, b, strict=True):
                assert np.array_equal(np.atleast_1d(x), np.atleast_1d(y), equal_nan=False) or (
                    np.size(x) == np.size(y)
                )
        else:
            np.testing.assert_array_equal(a, b, err_msg=f"column {col} was mutated")

    second = fit_dr4_like_single_source(df, sid)
    np.testing.assert_allclose(first.parameters, second.parameters, rtol=0, atol=0)


def test_excess_noise_is_reported_as_an_input(fits):
    bh3 = fits[4318465066420528000]
    assert bh3.excess_noise_input_mas == pytest.approx(6.5829, abs=1e-3)
    assert "not an official" in bh3.caveat


def test_parameter_names_are_carried_not_inferred(fits):
    result = next(iter(fits.values()))
    assert result.parameter_names[2] == "parallax"
    assert result.parameter_units[3] == "mas / yr"
    assert len(result.parameters) == len(result.parameter_names) == 6


def test_non_constant_excess_noise_is_refused(provider):
    from gaia_dr4_explorer.products.astrometry.fitting import FitError

    sid = 4318465066420528000
    df = provider.raw_table_for(sid).to_pandas()
    df.loc[df.index[0], "agis_source_excess_noise"] = 99.0
    with pytest.raises(FitError, match="not constant"):
        fit_dr4_like_single_source(df, sid)
