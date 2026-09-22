import numpy as np

from gaia_dr4_explorer.products.astrometry.times import (
    TIME_ORIGIN_JD,
    tcb_ns_to_jd,
    tcb_ns_to_jyear,
    tcb_ns_to_time,
)

_NS_PER_DAY = 86_400_000_000_000


def test_zero_is_the_time_origin():
    assert tcb_ns_to_jd(np.array([0], dtype="int64"))[0] == TIME_ORIGIN_JD


def test_one_day():
    jd = tcb_ns_to_jd(np.array([_NS_PER_DAY], dtype="int64"))
    assert jd[0] == TIME_ORIGIN_JD + 1.0


def test_real_prerelease_span():
    # First and last obs_time_tcb in Gaia DR4_RC3.
    ns = np.array([144415731432594100, 316750871920371197], dtype="int64")
    t = tcb_ns_to_time(ns)
    assert t.scale == "tcb"
    assert t[0].isot.startswith("2014-07-30")
    assert t[1].isot.startswith("2020-01-15")
    span = (t[1] - t[0]).to("yr").value
    assert 5.45 < span < 5.47


def test_sub_microsecond_precision_is_kept():
    # A float64 JD near the Gaia epoch resolves ~40 ns; the two-part split must
    # do better than a naive origin + ns/86400e9.
    base = 248777679747269187
    ns = np.array([base, base + 1000], dtype="int64")  # 1 microsecond apart
    t = tcb_ns_to_time(ns)
    delta_s = (t[1] - t[0]).to("s").value
    assert abs(delta_s - 1e-6) < 1e-9


def test_jyear_is_monotonic():
    ns = np.array([0, _NS_PER_DAY * 365, _NS_PER_DAY * 730], dtype="int64")
    y = tcb_ns_to_jyear(ns)
    assert np.all(np.diff(y) > 0)
