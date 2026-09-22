Research mode used: ordinary Codex web research fallback
Native Deep Research: unavailable from this invocation path
Scientific source rules: yes
Model effort: high   Date: 2026-09-22

---

## Executive summary

- `obs_time_tcb` in epoch astrometry is the effective observation time expressed as TCB at Gaia, not a pre-applied barycentric-arrival time. `obs_time_bary_corr` is an additive Roemer/light-travel correction.
- A barycentric astrometry time is therefore `obs_time_tcb + obs_time_bary_corr`. Label it “BJD(TCB) − 2455197.5” or “barycentric TCB time”; label raw `obs_time_tcb` “TCB at Gaia.”
- Gaia DR4 uses reference epoch J2017.5. `gaiasupdate.set_relative_time()` returns barycentric-corrected TCB time relative to J2017.5, not relative to J2016.0 or J2010.0.
- `transit_id` denotes the physical Gaia focal-plane transit, not a product-specific row identifier. A cross-product join on `(source_id, transit_id)` across epoch astrometry, epoch photometry, and XP epoch spectra is supported by the data model, subject to product-specific missing/rejected observations.
- The prerelease VOTable’s `refposition="BARYCENTER"` is not reconciled explicitly in the public documentation with the Gaia-centric astrometry convention. The operational documentation and official software both indicate that the barycentric correction must be added.

## Detailed findings

### 1. `obs_time_tcb` and `obs_time_bary_corr`

The DPAC time-conventions document explicitly distinguishes the two cases:

- Epoch photometry for sources outside the Solar System is time-tagged as TCB at the Solar-System barycentre.
- Epoch astrometry for those sources is time-tagged as TCB at Gaia.  
  [DPAC, *Reference systems, Conventions and Notations for Gaia*, GAIA-CA-SP-ARI-BAS-003-07, §3.8](https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=358698)

The same document states that Gaia scientific times are represented as nanoseconds since J2010.0/TCB, where J2010.0 is JD 2455197.5. It also emphasizes that TCB is a time coordinate, not the proper time of the barycentre or any other body.

For epoch astrometry, DPAC’s local-plane-coordinate technical note defines the barycentric time as

\[
t_B = t_{\rm Gaia} + \Delta t_{\rm bary},
\]

where the correction is the Roemer/light-travel correction. It states that the correction is positive when the corresponding event at the barycentre occurs later than the event at Gaia. [DPAC, *Local plane coordinates for the detailed analysis of complex Gaia sources*, GAIA-C3-TN-LU-LL-061-08, Eq. 5 and footnote 8](https://dms.cosmos.esa.int/COSMOS/doc_fetch.php?id=504573)

The official `gaiasupdate` implementation makes the convention unambiguous. Its `set_relative_time()` method computes:

```text
relative_time_year =
    2010.0
  + obs_time_tcb * ns_to_year
  + obs_time_bary_corr * ns_to_year
  - DR4_REFERENCE_EPOCH
```

The source code explicitly describes this as “including the barycentric correction.” [Official `gaiasupdate` source](https://esa.github.io/gaia-supdate/_modules/gaiasupdate/epoch_astrometry.html)

Therefore:

```text
t_barycentric = obs_time_tcb + obs_time_bary_corr
```

in nanoseconds relative to J2010.0.

Recommended labels:

- Raw field: `TCB at Gaia − 2455197.5 [day]`
- Corrected field: `BJD(TCB) − 2455197.5 [day]`
- Or, in years: `barycentric TCB years from J2017.5`

Do not add the correction again if using `relative_time_year` produced by `gaiasupdate`.

The prerelease VOTable’s `refposition="BARYCENTER"` declaration is therefore ambiguous or inconsistent with the documented epoch-astrometry semantics. I found no official note stating that this declaration means `obs_time_bary_corr` has already been applied. The stronger operational evidence is the DPAC formula and the official `gaiasupdate` implementation: the correction is additive and not already included.

### 2. DR4 reference epoch and `relative_time_year`

The official Gaia DR4 content page states that:

- DR4’s reference epoch is J2017.5.
- DR4’s time coordinate is TCB.
- DR4 covers observations from 25 July 2014 to 20 January 2020.  
  [Gaia DR4 content page](https://www.cosmos.esa.int/web/gaia/dr4)

The official `gaiasupdate` constants document independently defines:

- `TCB_REFERENCE_EPOCH`: 2010-01-01T00:00:00 TCB
- `DR4_REFERENCE_EPOCH`: J2017.5 TCB  
  [Official `gaiasupdate` constants](https://esa.github.io/gaia-supdate/gaiasupdate.html)

Thus the exact meaning is:

\[
\texttt{relative\_time\_year}
=
\left[
J2010.0
+
\frac{\texttt{obs\_time\_tcb}
+\texttt{obs\_time\_bary\_corr}}
{365.25\,{\rm d}}
\right]
-
J2017.5.
\]

So `relative_time_year = 0` corresponds to the barycentric-corrected epoch J2017.5 TCB.

The observed range in a particular prerelease source does not determine the reference epoch; it reflects that source’s actual observation times and scan coverage. It is fully compatible with a J2017.5 reference epoch.

### 3. Is `transit_id` shared across products?

The Gaia DR3 data model defines `transit_id` as a unique identifier assigned when a detected source transits the Gaia focal plane. The identifier encodes the along-scan time, field of view, CCD row, and across-scan position. This is a physical-transit identifier, not an identifier allocated independently by the photometry or astrometry products. [Gaia DR3 epoch-photometry data model, §20.7.1](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_photometry/ssec_dm_epoch_photometry.html)

The DR3 processing documentation explicitly says that G, BP, and RP measurements from a transit “share the same transit_id.” [Gaia DR3 global variability processing](https://gea.esac.esa.int/archive/documentation/GDR3/Data_analysis/chap_cu7var/sec_cu7var_global/ssec_cu7var_global_processing.html)

The DR4 product overview defines:

- `epoch_astrometry` as individual astrometric measurements;
- `epoch_photometry` as data aggregated per FoV transit;
- `xp_epoch_spectrum` as BP/RP spectra collected at transit level.  
  [Gaia DR4 product overview](https://www.cosmos.esa.int/web/gaia/dr4)

The draft DR4 data model is the relevant table-level specification for the prerelease products. [Draft Gaia DR4 data model, June 2026](https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/gaia-dr4-prerelease-draft-data-model_2026-06-26.zip)

Conclusion: a cross-product selection feature is buildable using:

```text
(source_id, transit_id)
```

as the join/highlight key across epoch astrometry, epoch photometry, and XP epoch spectra.

Caveats:

- Not every transit will have a usable row in every product.
- Photometry may be band- or transit-aggregated, while astrometry contains CCD-level arrays.
- Rejection, unavailable windows, or product-specific processing can produce missing counterparts.
- Joining on the floating-point time is unnecessary and less robust than joining on the identifier.

## Disagreements and unresolved points

1. The public documentation clearly supports the additive barycentric correction, but I found no ESA/DPAC clarification of why the prerelease VOTable declares `refposition="BARYCENTER"` for the astrometry table.
2. The public DR4 page still describes the data model as draft and subject to change. Recheck the final DR4 documentation before freezing labels or assuming every product’s row multiplicity.
3. The cross-product meaning of `transit_id` is strongly supported by its physical definition and the DR3 documentation’s explicit shared-use statement. However, I did not find a single public sentence saying verbatim “the DR4 epoch-astrometry, epoch-photometry, and XP-spectrum `transit_id` values are guaranteed equal.” The final DR4 data model should be treated as the final authority.

No journal papers or preprints were needed; consequently, no DOI or arXiv identifiers apply.