# gaia-dr4-explorer

Interactive explorer for **Gaia DR4 epoch astrometry**, one source at a time.

Built against the official ESA prerelease of 12 sources published on 2026-06-26, and designed
to extend to arbitrary DR4 sources when the public archive opens on 2026-12-02.

This is an explorer and diagnostic environment, not a bulk catalogue-analysis system.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Project bootstrap, CLI | done |
| 1 | Prerelease acquisition, checksum, provenance | done |
| 2 | Domain model and cache layout | done |
| 3 | CCD-level normalization, schema registry, time handling | done |
| 7 | DR4-like source update via `gaiasupdate` | done (library + CLI) |
| 4, 5 | Source selector and the astrometry views | done |
| 6 | Plugin registry | done |
| 8 | Raw inspector and export | done |
| 9 | Photometry plugin (Gaia DR3 light curves) | done |
| 10–11 | Spectra, SIMBAD/ADS context | not started |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

## Use

```bash
# Download and verify the prerelease archive (~626 kB)
gaia-dr4-explorer fetch-prerelease

# Summarise one source, and recompute its astrometry from the epoch data
gaia-dr4-explorer inspect 4318465066420528000 --fit

gaia-dr4-explorer cache-info
gaia-dr4-explorer clear-cache

# The interactive application
gaia-dr4-explorer serve --port 5006 --show
```

The application has four tabs: **Overview** (identity and statistics, with each block
labelled by where its numbers came from), **Astrometry** (coverage, along-scan centroid,
scan geometry, uncertainties, the transit x focal-plane matrix, and a filterable flag
table), **Source fit** (the DR4-like source update and its residual diagnostics), and
**Raw / metadata** (published schema, VOTable parameters, provenance, exports).

A `source_id` in the URL makes a view shareable:
`http://localhost:5006/?source_id=4318465066420528000`.

To work offline from the committed fixture instead of downloading:

```bash
export GAIA_DR4_EXPLORER_PRERELEASE_ZIP=$PWD/tests/fixtures/prerelease.zip
```

## The data

`gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip`, sha256 `07f0e8d9…d94d35fa0b`, from the
[Gaia DR4 prerelease page](https://www.cosmos.esa.int/web/gaia/dr4-prerelease). It declares
itself `Gaia DR4_RC3` and contains 1008 FoV transits for 12 sources: 10080 CCD observations,
8941 with a finite along-scan centroid, 7467 used by AGIS, spanning 2014-07-30 to 2020-01-15
(5.461 yr).

`docs/prerelease_reference.csv` holds the per-source reference values, with the release-page
metadata, the quantities we measure from the VOTable, and the fit results kept in separate
columns so the interface can always say which is which.

## Photometry, and which release it comes from

Gaia DR4 epoch photometry is **not** in the June-2026 prerelease and is not public
until 2026-12-02. The Photometry tab therefore draws on **Gaia DR3**, and says so on
every panel. Of the 12 prerelease sources, DR3 published epoch photometry for three:
Gaia-4 and the two variable QSOs. `docs/dr3_products.csv` records the availability
flags for all twelve, queried on 2026-09-22, so the application can report
availability without a network call.

Loading a light curve is the only action that contacts the Gaia archive. It is never
automatic, and the retrieved product is cached.

## Scientific caveats

Read `CLAUDE.md` before trusting any number this produces. The two that bite hardest:

- The fit results are **recomputed from epoch data**, not official catalogue values.
- `gaiasupdate`'s reported χ² and F2 exclude the AGIS source excess noise that the fit was
  weighted by, so they are **not** comparable to catalogue `astrometric_gof_al`. Gaia BH3
  reports F2 = 894 on that convention and F2 = 0.13 against the total variance. Both are
  displayed, each labelled.

## Tests

```bash
pytest                 # unit + integration, no network
pytest -m network      # the downloader against the live ESA server
ruff check .
```

## Credit

Gaia data are © ESA/Gaia/DPAC. Follow the
[citation instructions](https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/)
and the [licence](https://www.cosmos.esa.int/web/gaia-users/license). The source update is
computed with [`gaiasupdate`](https://github.com/esa/gaia-supdate).
