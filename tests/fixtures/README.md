# Test fixtures

## `prerelease.zip`

The official ESA Gaia DR4 epoch-astrometry prerelease archive, redistributed
**unmodified** so the test suite never needs the network.

```
url     https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip
bytes   625651
sha256  07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b
```

Gaia data are © ESA/Gaia/DPAC, under the
[Gaia data licence](https://www.cosmos.esa.int/web/gaia-users/license).
See the repository `LICENSE` for the acknowledgement requirements.

## `prerelease_reference.csv`

Per-source reference values, regenerated from `prerelease.zip` by this package's
own code. Release-page metadata, VOTable-derived measurements and gaiasupdate
fit results are kept in separate columns. Pinned to `gaiasupdate==0.1.2`; the
`fit_*` columns are a regression snapshot and will move if that pin changes.

## `dr3_epoch_photometry_gaia4.vot`

Gaia DR3 `EPOCH_PHOTOMETRY` DataLink response for Gaia-4
(`source_id = 1457486023639239296`), retrieved unmodified on 2026-09-22. Used to
pin the band-column mapping to the columns the archive actually serves, without
a network call.
