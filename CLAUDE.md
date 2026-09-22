# CLAUDE.md — gaia-dr4-explorer

Invariants for anyone (human or agent) working in this repository.

## Data integrity

1. **A `source_id` is release-specific.** A Gaia DR3 identifier is never assumed to denote the
   same object as the numerically equal DR4 identifier. Cache keys always include the release.
   (All 12 prerelease IDs happen to exist in DR3 with the same number — that is a property of
   ESA's curated sample, not a general rule.)
2. **The raw downloaded product is immutable.** Everything else is derived and regenerable.
3. **Missing stays missing.** Never substitute zero for a masked or NaN value.
4. **Units and masks are preserved** as far as practical. Plotting code reads units from
   `products/astrometry/schema.py`; it never converts angles implicitly.
5. **`used_by_agis_al = False` is a rejected observation, not corrupt data.** It stays
   inspectable. The default view may highlight used observations; it must not delete the rest.
6. Every derived dataset carries a `ProvenanceRecord`.
7. This release declares itself `Gaia DR4_RC3` — a release *candidate*. Say so in the UI.

## Architecture

8. **UI modules never perform archive queries.** All I/O sits behind the providers in `data/`.
9. **No network I/O at import time**, anywhere.
10. Plot functions take domain objects and return plot objects. They do no I/O.
11. No global mutable state. Archive access must be mockable.

## gaiasupdate

12. **Do not reimplement its algorithms.** Use `GaiaEpochAstrometryArchive.supdate`, which
    applies an official column mapping *and* a `colourFactorAl *= -1e3` conversion. A
    hand-rolled snake→camel rename silently skips that conversion and degrades faint and red
    sources (the QSO parallaxes move from ~0.003 mas to 0.02–0.09 mas).
13. **Its objects are single-use and its helpers mutate their inputs.** Build a fresh object
    from a deep copy for every fit. Never pass a UI-owned DataFrame to a `gaiasupdate` helper.
14. **Its `chi2`/`F2` are NOT the catalogue `astrometric_gof_al`.** They are computed against
    the measurement variance *excluding* the AGIS source excess noise, while the fit weights by
    the total variance including it. Always display both, each labelled by its variance
    definition. See `PLAN.md` §4.1.
15. `gaiasupdate.utils.fov_from_transit_id` raises `OverflowError` on every real transit ID
    under NumPy 2.x. Use `products.astrometry.normalize.fov_from_transit_id`.
16. Pin `gaiasupdate==0.1.2` and `astropy<8.0` (its own pin).

## Science

17. **Never present recomputed values as official catalogue values.**
18. The binarity signal in this sample is `agis_source_excess_noise`, an AGIS *input*. A large
    package-reported F2 is an artefact of invariant 14, not a detection.
19. CCD samples within one transit are not statistically independent.
20. Do not build a sky-track or tangent-plane view by guessing the scan-angle sign convention.
    Validate against an official example first.

## Process

21. **No conclusions without tests.** Numerical transforms require tests before claims.
22. Network tests carry the `network` marker and are excluded from the default run.
23. When a sign, coordinate or time convention is uncertain, stop and add a test or a reference
    before implementing it. Items marked **UNVERIFIED** in `PLAN.md` are not yet settled.
