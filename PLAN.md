# Gaia DR4 Object Explorer — implementation plan (rev. 2)

Revised 2026-09-22. Supersedes rev. 1. Every quantitative statement below was measured
from the actual prerelease file or read from the `gaiasupdate` 0.1.2 source; nothing here
is assumed. Claims that remain unverified are marked **UNVERIFIED**.

---

## 1. Objective

A research-grade Python application for inspecting **one Gaia DR4 source at a time**.

First implemented product: Gaia DR4 `epoch_astrometry`, using the official June-2026
prerelease of 12 sources. The codebase must be designed to admit arbitrary DR4 sources and
further TAP/DataLink products once the public archive opens (2026-12-02), but **no
speculative code is written against the unreleased schema**.

This is an explorer and diagnostic environment, not a bulk catalogue-analysis system.

---

## 2. Verified facts about the data

Source: <https://www.cosmos.esa.int/web/gaia/dr4-prerelease>

```
URL     https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip
bytes   625651
mtime   2026-06-26T06:57:22Z
sha256  07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b
members GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml (1183282 B), readme.txt (387 B)
```

VOTable 1.4. `TIMESYS`: `timescale=TCB`, `refposition=BARYCENTER`, `timeorigin=2455197.5` (JD).
`PARAM release = "Gaia DR4_RC3"`. Single `solution_id = 2888461026632663040`.

| Quantity | Measured value |
|---|---|
| Rows (FoV transits) | 1008 |
| Columns | 37 |
| Unique `source_id` | 12 (62–115 transits each) |
| CCD array length | **10, in every row** (SM + AF1…AF9) |
| CCD slots | 10080 |
| `centroid_pos_al` finite | 8941 (1139 NaN) |
| `used_by_agis_al = True` | 7467 (74.1% of slots) |
| `used_by_agis_al` where centroid is NaN | **0** — NaN ⇒ not used, always |
| Time span | 2014-07-30T11:28:51 → 2020-01-15T02:21:11 TCB (5.461 yr) |
| Median `centroid_pos_error_al` (used) | 0.147 mas |

### 2.1 Columns that are degenerate in the prerelease

These are **not** bugs; they are properties of this sample. The UI must handle them without
appearing broken, and the code must not assume they will stay this way in DR4 proper.

| Column | Measured state | Consequence |
|---|---|---|
| `centroid_pos_ac`, `centroid_pos_error_ac` | **zero-length array in all 1008 rows** | any naive zip/explode over array columns must tolerate length 0 ≠ 10 |
| `used_by_agis_ac` | False in all 10080 slots | no AC-based view is meaningful here |
| `multipeak`, `blended` | False in all 1008 rows | filters are no-ops; auto-disable and say why |
| `g_mag` | zero spread within every source | "σ vs G across transits" is a no-op here |
| `parallax_factor_al`, `parallax_factor_ac`, `obs_time_bary_corr`, `zeta` | masked in 97 of 1008 rows | genuine missing data; never fill with zero |

The 97 transits with masked `parallax_factor_al` are **exactly** the 97 with masked
`obs_time_bary_corr`, and all 970 of their CCD slots have `used_by_agis_al = False`.
`gaiasupdate` therefore removes them via two unrelated filters. **This is a coincidence the
code relies on, not a guard.** Assert it in the normalizer.

### 2.2 Full column schema (37)

Per-transit scalars (19):
`solution_id`(i8) `source_id`(i8) `transit_id`(i8) `ra0`(f8,deg) `dec0`(f8,deg)
`agis_source_excess_noise`(f4,mas) `obs_time_bary_corr`(f4,ns,masked) `zeta`(f4,deg,masked)
`parallax_factor_al`(f4,masked) `parallax_factor_ac`(f4,masked)
`nu_eff_used_in_astrometry`(f4,1/µm) `nu_eff_error`(f4,1/µm)
`transit_acq_flags`(i2) `transit_proc_flags`(i2) `multipeak`(bool) `blended`(bool)
`g_mag`(f4,mag) `g_class`(i2) `ac_rate`(f4,pix/s)

Per-CCD arrays, `arraysize="*"`, all length 10 (16):
`obs_time_tcb`(i8,ns) `scan_pos_angle`(f8,deg) `colour_factor_al`(f4) `colour_factor_ac`(f4)
`centroid_pos_al`(f8,mas) `calculated_pos_ac`(f8,mas) `centroid_pos_error_al`(f4,mas)
`used_by_agis_al`(bool) `used_by_agis_ac`(bool) `ccd_proc_flags`(i2) `ipd_error_al`(f4,mas)
`ipd_error_ac`(f4,mas) `gates`(i2) `source_dist_to_last_ci`(f4,pix) `sub_pixel_coord`(f4,pix)
`mu`(f4,pix)

Per-CCD arrays that are empty (2): `centroid_pos_ac`, `centroid_pos_error_ac`

Rev. 1 of this plan listed 13 fields. All 37 must be carried through normalization. In
particular `agis_source_excess_noise` and `obs_time_bary_corr` are **required by the fit**.

### 2.3 Flag encoding

`transit_acq_flags`, `transit_proc_flags`, `ccd_proc_flags`, `gates` are VOTable `short`
(int16) carrying **unsigned** bitmasks.

- `transit_proc_flags` observed values: `{-32208, -24016, 560, 564, 8756}` → as uint16
  `{33328, 41520, 560, 564, 8756}`. Bit 15 is set.
- `ccd_proc_flags` observed range 0–9040, 28 distinct values, none negative **in this file**.
- `gates` observed values: `{0, 4, 7, 8, 9, 10, 11, 12}`. `g_class`: `{0, 1, 2, 3}`.

**Verified behaviour under numpy 2.5 / pandas 2.3:** masking bits 0–14 on an int16 array is
correct; `int16_series & 0x8000` raises `OverflowError` (loud, not silent). Cast to `uint16`
before any bit test. `gaiasupdate`'s decoders are safe — their key arrays are int64, so int16
inputs promote and sign-extend correctly.

### 2.4 Time

`obs_time_tcb` is **int64 nanoseconds** since `timeorigin` JD 2455197.5, per CCD.
`obs_time_bary_corr` is a separate **per-transit** scalar in ns.

- Canonical store stays int64 ns. float64 JD loses ~µs.
- **RESOLVED 2026-09-22** from the draft DR4 data model
  (`gaia-dr4-prerelease-draft-data-model_2026-06-26.zip`, sha256 `d807eae9…f42c66`):

  > `obs_time_bary_corr` — "Barycentric correction to the observation time, **in the sense of
  > TCB(barycentric) − TCB(at Gaia)**, calculated for the obsTime of the AF4 CCD, i.e. at the
  > middle of the FoV transit."

  So `obs_time_tcb` is **TCB at Gaia**, and `t_bary = obs_time_tcb + obs_time_bary_corr`.
  The VOTable's `TIMESYS/@refposition="BARYCENTER"` contradicts the data model and must not
  be believed. `gaiasupdate.set_relative_time()` agrees with the data model: it adds the
  correction. Note the correction is computed at AF4, so it is exact at mid-transit and
  approximate for the other CCDs of the same transit — say so wherever it is used.
  `obs_time_bary_corr` appears in `epoch_astrometry` and in no other DR4 table, consistent
  with DPAC's convention that epoch photometry is already barycentric while epoch astrometry
  is at Gaia.
- **RESOLVED 2026-09-22.** The DR4 reference epoch is **J2017.5** (DR3 used J2016.0);
  verified in `gaiasupdate.constants`:
  `DR4_REFERENCE_EPOCH = Time('2017.5', format='jyear', scale='tcb')`.
  `relative_time_year` is barycentric TCB years relative to J2017.5, so zero is J2017.5.

---

## 3. `gaiasupdate` integration — corrected

Version **0.1.2** (pin exactly; the public API is undocumented and moving).

### 3.1 The supported entry point

```python
from gaiasupdate.epoch_astrometry import GaiaEpochAstrometryArchive

result = GaiaEpochAstrometryArchive.supdate(df_snake_case, source_id)   # dict
```

The prerelease VOTable uses **snake_case**, which is the "Archive" naming. `supdate` is a
classmethod on `GaiaEpochAstrometryArchive` (`epoch_astrometry.py:667`); it calls
`archive_to_cu9` (`:646`), which selects 13 columns and renames them to camelCase, then
delegates to `GaiaEpochAstrometryCu9.supdate`.

**Do not hand-roll the rename.** `GaiaEpochAstrometryCu9.supdate` additionally applies

```python
ea_df.loc[:, 'colourFactorAl'] *= -1e3
```

a sign-and-unit conversion on the colour factor. Omitting it leaves the 6th basis column off
by −1000 and measurably degrades faint/red sources: the three QSO parallaxes move from
≈0.003 mas (correct) to 0.02–0.09 mas, and the G=19 parallax source from 0.9993 to 0.9619 mas.
Bright sources are unaffected, so the error hides in exactly the cases you would test first.

`GaiaSourceEpochAstrometryArchive` exposes only `get_design_parameters` and
`from_gacs_datalink`; neither reproduces the DR4-like fit on its own.

### 3.2 Model and parameter vector

`compute_source_parameters_like_dr4` fixes `model='6p_constrained_colour'`, `solver='agis'`,
`compute_excess_noise=False`, with a Gaussian prior of `0.085e-3` (1/nm) on the last parameter.

Design columns, in order (`epoch_astrometry.py:345,366,369`):

| i | design column | astrometric meaning |
|---|---|---|
| 0 | `sin_theta` | Δα* |
| 1 | `cos_theta` | Δδ |
| 2 | `parallax_factor_al` | ϖ |
| 3 | `sin_theta_time` | μα* |
| 4 | `cos_theta_time` | μδ |
| 5 | `colour_factor_al` | pseudocolour term |

`get_design_equation_parameters` returns these names as `normal_matrix_column_names`, but
**`solve()` does not propagate that key into the result dict**. Capture it yourself from
`get_design_parameters()` and carry it into the domain result object. Hard-coded indices are
valid only for this model and this package version; assert `n_parameters == 6` and the name
list before indexing.

### 3.3 Mutation hazards (all verified in source)

`gaiasupdate` objects are **single-use** and several helpers mutate their input:

- `compute_source_parameters` replaces `self.epoch_data` with the filtered subset (`:921-928`).
  Calling it twice does not re-run on the original data.
- `get_design_parameters` likewise replaces `self.epoch_data` (`:887-900`).
- `explode_ccdlevel_columns` does `df.dropna(axis='columns', how='all', inplace=True)` on its
  **argument** (`:101`) and adds `component_index`. Verified: passing a frame directly costs it
  two columns and gains one. Always pass `.copy()`.
- `set_relative_time` (`:275`), `set_scan_angle_derived_columns` (`:291`), `sort_by_column`
  (`:307`) all add or reorder columns in place.
- `fix_blended_column` (`:70`) mutates then rebinds the accessor without returning.
- The constructor *does* `.copy()` before exploding (`:461`) — but only on the explode branch;
  with `is_exploded=True` it binds the caller's frame directly.

Rule for the app: construct a fresh object from a deep copy for every fit; never hold one as
UI state; never pass a UI-owned DataFrame to a `gaiasupdate` helper.

### 3.4 Known upstream bug

`gaiasupdate.utils.fov_from_transit_id` (`utils.py:34`) computes
`np.byte(transit_id >> 15) & 0x03` and raises
`OverflowError: Python integer 7592092277443 out of bounds for int8` on **every** real
transit ID under numpy 2.x. Reimplement locally as `(int(tid) >> 15) & 0x03`, cite
GAIA-C3-TN-UB-JP-011 p.9 as the upstream does, and report it to ESA.

The two flag decoders `decode_ccd_proc_flag_to_description` and
`decode_transit_acquisition_flag_to_description` work correctly and should be used rather than
reimplemented. Note neither decodes `transit_proc_flags`, and `gates` has no decoder.

---

## 4. Scientific guardrails

These replace the rev. 1 guardrails and add the one that matters most.

### 4.1 `chi2` / `F2` from `gaiasupdate` are NOT the catalogue `astrometric_gof_al`

`solver.py:672` sets `results['measurement_variance'] = self.observation_variances[index_keep]`
— **excluding** the AGIS source excess noise — and `chi2` is then computed against that. But
the fit itself weights by `total_variance = observation_variances + excess_noise²`
(`epoch_astrometry.py:948-957`). The reported χ² and F2 therefore use a different variance
from the one that produced the parameters.

Measured consequence:

| source | excess noise (mas) | median σ_AL (mas) | F2 as reported | F2 vs total variance |
|---|---|---|---|---|
| HD 183633 | 0.000 | 0.105 | −0.89 | −0.89 |
| HD 114762 | 1.227 | 0.125 | 186.50 | **0.95** |
| Gaia-4 | 0.119 | 0.082 | 31.53 | **2.01** |
| Gaia BH3 | 6.583 | 0.085 | 893.97 | **0.13** |

Gaia BH3 goes from F2 = 894 to F2 = 0.13. **The apparent "single-star model fails on binaries"
signal is an artefact of the variance inconsistency, not a detection.** Rev. 1 of this plan got
this wrong and proposed it as an acceptance test.

Requirements:
- Display both statistics side by side, each labelled with its variance definition.
- Never label either as `astrometric_gof_al`. The `metrics.py:279-285` comment asserts the
  identification; the variance conventions do not support it.
- The binarity signal in this dataset is `agis_source_excess_noise`, an **AGIS input**
  (6.583 / 1.227 / 0.119 mas for the three `orbit` sources; 0.000 for eight of nine others).
  It requires no fit at all. Present it as an input, not as a result of our computation.

### 4.2 Other guardrails (retained from rev. 1)

- Do not implement a tangent-plane or reconstructed sky-track view by guessing the scan-angle
  sign convention. Validate against an official example first.
- CCD samples within one transit are not statistically independent. The explorer may display
  them individually; aggregation and model choice belong in an explicit analysis layer.
- `used_by_agis_al = False` is a rejected observation, not corrupt data. Preserve and display it.
- Never present recomputed parameters as official catalogue values. Label them "recomputed from
  epoch data with gaiasupdate 0.1.2".
- `release = "Gaia DR4_RC3"`. These are release-*candidate* identifiers. Record the release
  string and `solution_id` in provenance and warn in the UI header. Even these DR4 IDs may not
  survive to the public release.
- A DR3 `source_id` is never assumed equal to a DR4 `source_id`.
- `compute_excess_noise=True` is a different estimator from the DR4-like path. Offer it as an
  explicit alternative, never as the default.

---

## 5. Core design principles

1. Science/data code is independent of the GUI. UI modules never query an archive.
2. Preserve Gaia metadata, masks and units. Missing stays missing; never fill with zero.
3. The raw downloaded product is immutable; normalized data are derived and regenerable.
4. Every derived dataset carries provenance (URL, sha256, retrieval time, release string,
   `solution_id`, package versions).
5. `source_id` is release-specific. Cache keys always include release **and** product type.
6. DataLink products are lazy-loaded; never fetch all products for a source, never
   `retrieval_type="ALL"` from interactive navigation.
7. Use official Gaia/ESA routines where they exist and are correct. Where they are wrong
   (§3.4) reimplement minimally and document why.
8. Transformations are small, typed, and individually tested.
9. No global mutable state. Archive access is mockable. Plot functions take domain objects and
   return plot objects; they never perform I/O.
10. Every phase leaves the application runnable.

---

## 6. Technology stack

Python ≥ 3.11. `astropy`, `numpy`, `scipy`, `pandas`, `pyarrow`, `panel`, `param`,
`holoviews`, `hvplot`, `bokeh`, `platformdirs`, `pytest`, `ruff`.
`gaiasupdate==0.1.2` **pinned exactly**.

**`astroquery` is a base dependency, not a Phase-9 one.** Verified: `gaiasupdate` does
`from astroquery.gaia import GaiaClass, Gaia` at module scope
(`epoch_astrometry.py:17`), and its wheel declares `astroquery>=0.4.7`. Importing
`gaiasupdate` at all therefore imports `astroquery`.

**`astropy` must be pinned `>=6.0,<8.0`.** `gaiasupdate` 0.1.2 declares `astropy<8.0`.
Note the measurements in §2 were taken with astropy 8.0.1 for parsing and 7.2.2 for fitting;
the pinned environment uses 7.2.2 throughout.

**Deployment — a Phase 0 decision, not a Phase 8 check.**

Two build profiles, fixed before any UI code is written:

- **`server`** — `gaia-dr4-explorer serve`, a normal Panel app with a filesystem cache, the
  CLI, and live archive access. This is the development and research target.
- **`static`** — `panel convert --to pyodide-worker`, for GitHub Pages. No Python server, no
  host filesystem, no deploy-time secret, no raw sockets, no blocking synchronous HTTP.

Rules that follow, enforced from Phase 0:
- The static entry point must not import any server-only module. Keep a
  `gaia_dr4_explorer.ui.entry_static` that imports only from `domain`, `products` and a
  bundled-resource provider.
- No module performs network I/O at import time.
- The prerelease VOTable is **bundled** into the static build as a resource; `platformdirs`
  and the filesystem cache are server-profile only. The static profile uses an in-memory
  store with optional browser storage.
- External services (Gaia TAP, SIMBAD, ADS) are subject to browser CORS and are
  **UNVERIFIED** from a Pyodide origin. Until tested, the static profile ships Phases 0–8
  only, with Phases 9–11 marked server-only.
- **An ADS token is never embedded in a static artefact.** In the static profile ADS is
  either omitted or accepts a user-supplied token entered at runtime. A test must assert no
  token string appears in any built artefact.
- **The fit cannot run in the static profile.** Measured 2026-09-22:
  `import gaiasupdate.epoch_astrometry` pulls in ~110 top-level modules, including
  `astroquery`, `requests`, `urllib3`, `pyarrow`, `multiprocessing`, `socket`, `ssl`,
  `subprocess`, `fcntl`, `termios`, `grp` and `pwd`. Two hard blockers:
  1. **`astroquery` is not in the Pyodide distribution.** `astropy`, `pandas`, `scipy`,
     `pyarrow` and `requests` are; `astroquery` is not. Since `gaiasupdate` imports it at
     module scope, `import gaiasupdate` fails under Pyodide unless `astroquery` and its own
     dependency tree (including `keyring`) install cleanly via `micropip`.
  2. **Pyodide ships pandas 3.0.2, and `gaiasupdate` 0.1.2 declares `pandas<3.0`.**

  Consequence, decided now rather than at Phase 8: the **static profile is an
  inspection-only viewer** — bundled prerelease data, normalization, and every Phase 4–5
  astrometry view, with precomputed fit results loaded from
  `docs/prerelease_reference.csv` instead of a live fit. The **server profile** keeps the
  live `Run DR4-like source update` button. The UI must therefore be able to render the fit
  panel from a stored result object, not only from a freshly computed one — a Phase 7
  requirement, not a later retrofit.

Normal `pyproject.toml`. No database server. Local filesystem for product storage.
**No SQLite manifest in the MVP** — the whole prerelease is 626 KB; a JSON sidecar per product
covers provenance. Add SQLite only when something needs it.

Package `gaia_dr4_explorer`, console script `gaia-dr4-explorer` with subcommands
`serve`, `fetch-prerelease`, `inspect SOURCE_ID`, `cache-info`, `clear-cache`.

---

## 7. Scope

Rev. 1 had 19 phases, most targeting a schema that does not exist until 2026-12-02. Building
them now against mocks guarantees rework.

**In scope now — Milestones A, B and C:** Phases 0–11 below.

Phases 9–11 add photometry, spectra and external context. Verified 2026-09-22: **none of these
products exist in the DR4 prerelease**, but DR3 counterparts are retrievable today (§12), so
these phases are built against DR3 and re-pointed at DR4 in December.

**Deferred to a design note (`docs/future-products.md`), not implemented:** DR3→DR4 resolver,
DR4 source overview, NSS, variability, astrophysical parameters, and image products. One page
each stating the intended interface. Revisit after the archive opens.

The plugin registry and provider contract are **Phase 2**, before the source selector and the
astrometry views, so those are built as plugins from the start rather than refactored into the
registry afterwards. (Rev. 2 initially placed the registry at Phase 6, which guaranteed a
rework of Phases 4–5; corrected 2026-09-22.)

---

## 8. Phases

### Phase 0 — Bootstrap
`pyproject.toml`, package skeleton, pytest + ruff config, CLI, minimal Panel app showing title
and version, `CLAUDE.md`, `README.md`.

`CLAUDE.md` must state prominently: DR3 ≠ DR4 source IDs; raw data untouched; units and masks
preserved; DataLink lazy; UI never queries; numerical code requires tests; `gaiasupdate` is an
external dependency whose known defects are listed in §3.

**Accept:** `pytest` passes; `ruff check .` passes; app starts from CLI; package imports
without starting the app. Commit.

### Phase 1 — Prerelease acquisition
`PreReleaseProvider`: download the ZIP, store in cache, record sha256 + retrieval timestamp +
URL, extract safely, locate `GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml`, read with
`Table.read(..., format="votable")`, expose unique source IDs, return the raw table per source.

No redownload of an unchanged local copy (compare sha256 against §2). Configurable local
override so tests use a fixture instead of the network.

Runtime validation: 12 unique source IDs; all array columns length 10 **or 0**; unknown extra
columns must not break parsing. Do not overfit to row counts.

**Tests:** missing ZIP; corrupt ZIP; XML absent; malformed VOTable; unknown source ID; masked
columns; zero-length array columns; extra unexpected columns; provenance record content.
Network tests carry an explicit pytest marker and are excluded from the default run. Commit.

### Phase 2 — Domain model and cache
`SourceKey(release, source_id)`, `SourceContext`, `ProductDescriptor`, `ProductPayload`,
`ProvenanceRecord`. No GUI objects in any of them.

```
cache/prerelease/
cache/raw/RELEASE/SOURCE_ID/
cache/normalized/RELEASE/SOURCE_ID/
cache/metadata/
```

Raw is immutable; normalized is regenerable. Never key a cache entry on `source_id` alone.
Commit.

### Phase 3 — Epoch-astrometry normalization
Produces three representations: (1) the untouched raw table; (2) a transit-level table;
(3) a canonical CCD-level table.

Requirements:
- Carry all 37 columns (§2.2). Validate each array column's length explicitly before
  broadcasting — do not blanket-`explode()`.
- Handle length-0 arrays. `gaiasupdate.explode_ccdlevel_columns` drops them via
  `dropna(axis='columns', how='all')`; our normalizer should instead **keep the column and
  mark it empty**, so the UI can say "no AC data in this release" rather than silently omitting it.
- Emit `ccd_index` (0–9) and `ccd_name` (`SM`, `AF1`…`AF9`) preserving original array position.
- Emit `fov` from `transit_id` using the local reimplementation (§3.4).
- Cast flag columns to `uint16` at normalization time (§2.3).
- Add an assertion that every row with a masked `parallax_factor_al` has no AGIS-used slot
  (§2.1); fail loudly if DR4 breaks this.
- Schema registry: `field, original unit, canonical unit, description, level ∈ {source,
  transit, ccd}`. No implicit degree/radian conversion in plotting code.
- Time: keep int64 ns canonical; derived Astropy `Time` / JD columns produced by one isolated,
  tested function. Never overwrite the original timestamp.

**Oracle test:** assert the CCD-level table agrees row-for-row with
`GaiaCentroidDataFrame.explode_ccdlevel_columns(df.copy())` on shared columns. Pass a copy
(§3.3). Commit.

### Phase 4 — Source selector and prerelease overview
Sidebar: data source ("DR4 prerelease"), source dropdown, source-ID text input, reload, cache
status, and an "RC3" release badge.

Ship `docs/prerelease_reference.csv` (already generated, 12 rows × 26 columns) as both the
overview metadata and a regression fixture. It separates:
- **from the release page:** `common_name`, `sample_category`, `g_mag_page`, `parallax_page_mas`
- **computed from the VOTable:** `n_transits`, `n_ccd_slots`, `n_ccd_finite`,
  `n_used_by_agis_al`, `frac_used`, `t_start_tcb`, `t_end_tcb`, `span_yr`,
  `median_sigma_al_used_mas`, `agis_source_excess_noise_mas`, `n_transits_masked_plxfac`
- **computed by us with gaiasupdate:** `fit_*`

The UI must label which of the three each displayed number is. Commit.

### Phase 5 — Astrometry exploration views
A. Coverage timeline (time by transit; all / AGIS-used / rejected; group by CCD or FoV).
B. AL centroid vs time with error bars; filters on used/rejected, SM/AF, transit, gate,
   σ threshold. Controls whose column is constant in the loaded source are **disabled with an
   explanatory tooltip** (`multipeak`, `blended` in this release).
C. Scan angle vs time; exact value on hover; angular convention unchanged.
D. AL parallax factor vs time (transit-level). Masked transits rendered as a distinct
   "missing" band, not dropped silently.
E. Uncertainty diagnostics: σ_AL distribution, IPD σ distribution, σ by CCD. The "σ vs G"
   panel is built but auto-hidden when G has zero spread (§2.1).
F. **Transit × focal-plane matrix.** Rows `transit_id`, columns SM/AF1…AF9. Cell quantity
   selectable: present/missing, used by AGIS, σ, `ccd_proc_flags`, `gates`. Largest source is
   115 × 10 = 1150 cells; `holoviews.HeatMap` is adequate.
G. Filterable flag table exposing raw flags and booleans, decoded via the `gaiasupdate`
   decoders where they exist. No single "quality score".

Hover everywhere shows: transit ID, CCD name, timestamp, AL centroid, σ, scan angle, AGIS-use
flag, decoded flags. Selection is shared between plots and the table where feasible. Commit.

### Phase 6 — Product-plugin architecture
```python
class ProductPlugin(ABC):
    key: str
    title: str
    def discover(self, context) -> ProductDescriptor: ...
    def load(self, context) -> ProductPayload: ...
    def summarize(self, payload): ...
    def build_view(self, context, payload): ...
    def export(self, payload, format): ...
```
Loading happens only on selection. Navigation is built from the registry. A missing product is
normal, not an exception: states are available / unavailable / not checked / loading / error.
Epoch astrometry becomes the first plugin with no behaviour change. Commit.

### Phase 7 — `gaiasupdate` fit, as the second plugin
`products/astrometry/fitting.py` wraps `GaiaEpochAstrometryArchive.supdate` per §3. Returns a
domain result object: model name, parameter names (from `normal_matrix_column_names`), fitted
parameters, formal uncertainties, covariance, residuals, `index_keep`, both χ²/F2 variants,
excess noise **as input**, n measurements, n outliers, provenance including `gaiasupdate`
version.

UI "Source fit" section, button `Run DR4-like source update`. Display Δα*, Δδ, ϖ, μα*, μδ,
pseudocolour term, each with its uncertainty and unit; both fit statistics labelled by variance
definition (§4.1); `agis_source_excess_noise` shown as an AGIS input.

Plots: residual vs time; normalized residual vs time; residual vs scan angle; residual vs
parallax factor; residual histogram; residual by CCD; residual by transit. Toggle on the
transit × CCD matrix: raw measurement / fit residual / normalized residual.

Note in the UI that the DR4-like path uses fixed external variance and therefore performs **no
robust downweighting or outlier rejection** (`solver.py:614-620`); `n_outliers = 0` and unit
weights are expected, not a bug. Commit.

### Phase 8 — Raw inspector and export
Every product page exposes a Raw/Metadata view: original field names, dtype, units, masks,
descriptions, VOTable metadata, provenance.

Exports: original raw product; normalized Parquet; normalized ECSV; filtered CSV; provenance
JSON. CSV export states that CSV does not preserve Astropy metadata. Commit.

---

## 8b. Phases 9–11 — photometry, spectra, context

### §12 availability matrix (measured 2026-09-22)

All 12 RC3 `source_id` values exist in `gaiadr3.gaia_source` with the **same number** and
matching coordinates and G. This is a property of ESA's curated sample, **not** a licence to
relax §4.2's DR3 ≠ DR4 rule; the resolver stays on the roadmap for arbitrary sources.

| Product | In DR4 prerelease | In DR3 today | Retrieval verified |
|---|---|---|---|
| Epoch astrometry | yes, all 12 | — | file |
| G/BP/RP epoch photometry | no | **3 of 12** | `EPOCH_PHOTOMETRY`, 23 and 65 transits |
| XP mean spectrum | no | **8 of 12** | `XP_SAMPLED` 343 pts 336–1020 nm; `XP_CONTINUOUS` coeffs |
| XP epoch spectra | no | no — DR4-only | — |
| RVS mean spectrum | no | **0 of 12** (`has_rvs=false`) | HTTP 200, zero-byte body |
| RVS epoch spectra | no | no — DR4-only | — |
| SIMBAD identity | — | **6 of 12** by `Gaia DR3 <id>` | TAP `ident`⋈`basic` |
| ADS bibliography | — | yes | `ADS_API_TOKEN` env var |

The six sources SIMBAD does not know are absent entirely, including at 5″ positional search.
"Not in SIMBAD" is a normal state, never an error.

DR3 and DR4 astrometry differ materially for this sample (Gaia BH3 ϖ = 1.644 mas in DR3 vs
1.961 from our DR4 fit; the three QSOs move from −0.286/−1.008/+0.352 to ≈0.003). **Every
panel must state which release it draws from.** A DR3 number and a DR4 number must never share
an unlabelled axis.

### Phase 9 — Photometry plugin (DR3 source)
`EPOCH_PHOTOMETRY` DataLink, wide format: `transit_id`, `g_transit_time`, `g_transit_mag`,
`bp_obs_time`, `bp_mag`, `rp_obs_time`, `rp_mag` plus fluxes, errors and rejection flags.

Views: magnitude vs time per band; flux-over-error vs time; rejected epochs drawn hollow, never
hidden. No period search in v1; phase-fold only when a period is explicitly supplied.

For the nine sources without epoch photometry, render "no epoch photometry published in DR3"
— not an empty axis. Commit.

### Phase 10 — Spectra plugin
Mean BP/RP: flux vs wavelength with error band, BP and RP visually distinguished.
Toggle between the `XP_SAMPLED` representation and one reconstructed locally from
`XP_CONTINUOUS` coefficients. **Basis-function reconstruction lives in a science module and is
unit-tested against `XP_SAMPLED` on the 8 sources that have both.** It never appears in a
plotting function.

RVS panel present, showing "unavailable for all prerelease sources" until DR4.

**Multi-epoch BP/RP** (no data until DR4; build the view, drive it from a synthetic fixture):
1. waterfall heatmap — epoch × wavelength, flux as colour;
2. overplot of selected epochs over the bold mean spectrum;
3. **residual-to-mean** — each epoch ratioed to the mean. This is the view that shows
   variability; a raw overplot is dominated by the common continuum and hides it.

**RESOLVED 2026-09-22, with a caveat.** `transit_id` is defined identically in **25** DR4
tables, including `epoch_astrometry`, `epoch_photometry`, `epoch_photometry_ccd` and
`xp_epoch_spectrum`: "a unique identifier assigned to each detected (and confirmed) source as
it transits the Gaia focal plane… the along-scan time and the across-scan position along with
the telescope in which the source was detected are used to [construct it]". It is a physical
focal-plane transit identifier, not a per-product row key, so `(source_id, transit_id)` is the
join key for cross-product linked selection.

The caveat: no sentence in the public documentation states verbatim that the values are
guaranteed equal across products, and the data model is still marked draft. Build the linked
selection against `(source_id, transit_id)`, but treat a non-matching transit as a normal
absence — not every transit yields a usable row in every product. Re-check on 2026-12-02.
Commit.

### Phase 11 — Context plugin (SIMBAD + ADS)
SIMBAD via TAP, `ident` ⋈ `basic`, keyed on `Gaia DR3 <source_id>` with a 5″ positional
fallback: main identifier, object type, spectral type, `nbref`, redshift, full cross-ID list.
ADS via `ADS_API_TOKEN`, reference list sorted by citation count, each row linking to its
bibcode.

Both are third-party services: cache responses, degrade to "unavailable" on failure, never
block the astrometry tabs on a network call, and never send anything but coordinates and public
identifiers. Commit.

---

## 9. Acceptance tests

Integration fixture: `docs/prerelease_reference.csv`. All values below were produced with
astropy 8.0.1, numpy 2.5.3, pandas 2.3.3, gaiasupdate 0.1.2 on 2026-09-22.

**T1 — structure.** 1008 rows, 12 sources, 37 columns, all CCD arrays length 10 or 0,
10080 slots, 8941 finite AL centroids, 7467 AGIS-used, `solution_id = 2888461026632663040`,
`release = "Gaia DR4_RC3"`. Exact equality.

**T2 — parallax recovery.** For the nine non-QSO sources, assert
`round(fit_parallax_mas, 1) == parallax_page_mas`. The release page quotes one decimal place,
so this states exactly what the page asserts and nothing more — no invented tolerance. All
nine pass (largest deviation 0.042 mas, for the ϖ = 4.9 source).

For the three QSOs the page value 0.0 is a physical expectation rather than a rounded
measurement, so test `abs(fit_parallax_mas) <= 0.01`. Justification: 0.01 mas is ≈ 1/30 of
their formal σ (0.32–0.48 mas); measured values are −0.0023, +0.0032, +0.0027. This is the
test that fails if the colour-factor conversion of §3.1 is lost (those values degrade to
−0.089, +0.015, −0.016).

**T3 — excess noise separates the orbit sample.** `agis_source_excess_noise` ≥ 0.1 mas for all
three `sample_category == "orbit"` sources and = 0.000 for all `parallax`- and
`magnitude`-category sources. This is an AGIS input, not our result (§4.1).

**T4 — F2 consistency.** F2 recomputed against total variance satisfies |F2| < 2.5 for all 12
sources (measured range −0.886 to +2.013), while the package-reported F2 exceeds 25 for all
three orbit sources (measured 31.5, 186.5, 894.0). The gap between the two is the artefact of
§4.1. This test fails loudly if `gaiasupdate` changes its variance convention upstream, which
would otherwise silently alter every statistic the app displays.

**T5 — regression snapshot.** Fitted ϖ, μα*, μδ, σ_ϖ and `fit_n_measurements` match
`prerelease_reference.csv` to 5 significant figures, pinned to `gaiasupdate==0.1.2`.

**T6 — colour-factor conversion.** Fitting via a naive camelCase rename (without `*= -1e3`)
must produce a QSO parallax differing from the official path by > 0.01 mas. Guards against
accidentally reintroducing the bug of §3.1.

**T7 — non-mutation and repeatability.** After a fit, the caller's DataFrame is deeply equal
to a pre-fit deep copy: same shape, columns, dtypes, index order, values, masks and nested
array contents (§3.3). Separately, fitting the *same source* twice through the adapter must
return identical parameters — proving the adapter builds a fresh `gaiasupdate` object rather
than reusing a filtered one.

**T8 — archive safety.** Extraction rejects members with absolute paths, `..` components or
symlinks; a truncated download leaves no file in place; a checksum mismatch raises rather than
returning the file; raw cache entries are never rewritten once created.

**T9 — array-length validation.** A table whose CCD array columns have *mixed* lengths within
one source is rejected by our normalizer with a clear message. Verified upstream behaviour:
`explode_ccdlevel_columns` raises `ValueError: columns must have matching element counts`
rather than misaligning silently, and zero-length columns are dropped by its
`dropna(axis='columns', how='all')` — so the T-oracle of Phase 3 is safe on both counts.

Unit tests must additionally cover: mask survival; unit survival; TCB conversion round-trip;
transit/CCD explosion; scalar broadcasting; source isolation (no mixing between the 12);
flag preservation and uint16 casting incl. bit 15; zero-length array columns; empty products;
malformed data; cache identity including release; DR3/DR4 identifier separation.

Use small synthetic VOTables as unit fixtures. GUI tests target state transitions, not pixels.

---

## 10. Definition of done (prerelease MVP)

A researcher can: start the app locally; select any of the 12 sources; inspect transit-level
and every CCD-level sample; see which CCD observations AGIS used and which it rejected;
filter interactively; inspect timing and scan geometry; see focal-plane coverage transit by
transit; run the official-style source update; inspect residuals and both fit statistics with
their variance definitions stated; export raw or normalized data; and trace every displayed
quantity back to a named Gaia field or a documented transformation in `docs/data-model.md`.

---

## 11. Documentation

`README.md` — purpose, supported products, install, running, prerelease acquisition, caching,
architecture, scientific caveats (§4), provenance and citation, development, tests.

`docs/data-model.md` — raw → normalization → domain → visualization, with the §2.2 schema.

`docs/product-plugin.md` — how to add a product without touching the core.

`docs/astrometry.md` — how each visualization maps onto Gaia fields, including the §4.1
statistics caveat.

`docs/future-products.md` — the deferred archive/TAP/DataLink/plugin designs (§7).

`docs/gaiasupdate-notes.md` — §3 in full: supported entry point, colour-factor conversion,
parameter ordering, mutation hazards, the `fov_from_transit_id` bug, version pin.

Citation: <https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/>
License: <https://www.cosmos.esa.int/web/gaia-users/license>
