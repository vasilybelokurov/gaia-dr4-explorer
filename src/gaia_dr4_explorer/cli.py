"""Command-line interface.

Importing this module must not start the application or touch the network.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.config import AppConfig, CacheLayout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaia-dr4-explorer",
        description="Explore Gaia DR4 epoch astrometry one source at a time.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override the cache root")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the interactive application")
    serve.add_argument("--port", type=int, default=5006)
    serve.add_argument("--show", action="store_true", help="open a browser window")

    sub.add_parser("fetch-prerelease", help="download and verify the prerelease archive")

    inspect = sub.add_parser("inspect", help="print a summary for one source")
    inspect.add_argument("source_id", type=int)
    inspect.add_argument("--fit", action="store_true", help="also run the DR4-like source update")
    inspect.add_argument("--json", action="store_true", help="emit JSON instead of text")

    sub.add_parser("cache-info", help="show what is in the cache")

    snap = sub.add_parser(
        "snapshot-spectra",
        help="search external archives for every prerelease source and save the result",
    )
    snap.add_argument("--out", type=Path, default=None,
                      help="output JSON (default: the package resource the app ships)")
    snap.add_argument("--timeout", type=float, default=180.0,
                      help="wall-clock budget per source, seconds")

    clear = sub.add_parser("clear-cache", help="delete cached products")
    clear.add_argument("--yes", action="store_true", help="do not prompt")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {}
    if args.cache_dir is not None:
        overrides["cache_dir"] = args.cache_dir
    config = AppConfig.from_env(**overrides)

    if args.command == "serve":
        return _serve(config, port=args.port, show=args.show)
    if args.command == "fetch-prerelease":
        return _fetch(config)
    if args.command == "inspect":
        return _inspect(config, args.source_id, fit=args.fit, as_json=args.json)
    if args.command == "cache-info":
        return _cache_info(config)
    if args.command == "snapshot-spectra":
        return _snapshot_spectra(out=args.out, timeout_s=args.timeout)
    if args.command == "clear-cache":
        return _clear_cache(config, assume_yes=args.yes)
    return 2


def _fetch(config: AppConfig) -> int:
    from gaia_dr4_explorer.data import PreReleaseError, PreReleaseProvider

    provider = PreReleaseProvider(config)
    try:
        archive = provider.ensure_archive()
        provider.ensure_extracted()
        ids = provider.source_ids()
    except PreReleaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"archive: {archive}")
    print(f"release: {provider.release()}")
    print(f"sources: {len(ids)}")
    for sid in ids:
        print(f"  {sid}")
    return 0


def _inspect(config: AppConfig, source_id: int, *, fit: bool, as_json: bool) -> int:
    from gaia_dr4_explorer.data import PreReleaseError, PreReleaseProvider
    from gaia_dr4_explorer.products.astrometry import normalize_epoch_astrometry

    provider = PreReleaseProvider(config)
    try:
        raw = provider.raw_table_for(source_id)
    except (PreReleaseError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    normalized = normalize_epoch_astrometry(raw, provenance=provider.provenance)
    payload = {"release": provider.release(), **normalized.summary()}

    if fit:
        from gaia_dr4_explorer.products.astrometry.fitting import (
            FitError,
            fit_dr4_like_single_source,
        )

        try:
            result = fit_dr4_like_single_source(raw, source_id, provenance=provider.provenance)
        except FitError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        payload["fit"] = {
            "model": result.model,
            "parameters": {k: {"value": v, "error": e, "unit": u}
                           for k, (v, e, u) in result.as_dict().items()},
            "n_measurements": result.n_measurements,
            "chi2_measurement_variance": result.chi2_measurement_variance,
            "f2_measurement_variance": result.f2_measurement_variance,
            "chi2_total_variance": result.chi2_total_variance,
            "f2_total_variance": result.f2_total_variance,
            "excess_noise_input_mas": result.excess_noise_input_mas,
            "caveat": result.caveat,
        }

    if as_json:
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(f"source_id {source_id}   release {payload['release']}")
    for key in ("n_transits", "n_ccd_slots", "n_ccd_finite", "n_used_by_agis_al", "frac_used",
                "span_yr", "median_sigma_al_used_mas", "agis_source_excess_noise_mas"):
        print(f"  {key:32s} {payload[key]}")
    for warning in normalized.warnings:
        print(f"  note: {warning}")
    if fit:
        f = payload["fit"]
        print(f"\n  DR4-like source update ({f['model']}) -- {f['caveat']}")
        for name, entry in f["parameters"].items():
            print(f"    {name:20s} {entry['value']:+14.6f} +/- {entry['error']:.6f} {entry['unit']}")
        print(f"    {'n_measurements':20s} {f['n_measurements']}")
        print(f"    {'F2 (measurement var)':20s} {f['f2_measurement_variance']:+.4f}  "
              "<- as gaiasupdate reports it, excess noise EXCLUDED")
        print(f"    {'F2 (total var)':20s} {f['f2_total_variance']:+.4f}  "
              "<- includes the excess noise the fit was weighted by")
        print(f"    {'excess noise (input)':20s} {f['excess_noise_input_mas']:.4f} mas")
    return 0


def _cache_info(config: AppConfig) -> int:
    layout = CacheLayout(config.cache_dir)
    root = layout.root
    print(f"cache root: {root}")
    if not root.exists():
        print("  (empty)")
        return 0
    total = 0
    for path in sorted(root.rglob("*")):
        if path.is_file():
            size = path.stat().st_size
            total += size
            print(f"  {size:>12,}  {path.relative_to(root)}")
    print(f"  {total:>12,}  TOTAL")
    return 0


def _clear_cache(config: AppConfig, *, assume_yes: bool) -> int:
    root = CacheLayout(config.cache_dir).root
    if not root.exists():
        print(f"nothing to remove at {root}")
        return 0
    if not assume_yes:
        reply = input(f"remove everything under {root}? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("cancelled")
            return 1
    shutil.rmtree(root)
    print(f"removed {root}")
    return 0


def _serve(config: AppConfig, *, port: int, show: bool) -> int:
    try:
        from gaia_dr4_explorer.app import serve
    except ImportError as exc:
        print(
            f"error: the interactive application needs the 'ui' extra: {exc}\n"
            "  pip install 'gaia-dr4-explorer[ui]'",
            file=sys.stderr,
        )
        return 1
    serve(config, port=port, show=show)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def _snapshot_spectra(*, out: Path | None, timeout_s: float) -> int:
    """Search every archive for the prerelease sources; write the snapshot."""
    from importlib.resources import files

    from gaia_dr4_explorer.data import external_spectra as xs
    from gaia_dr4_explorer.data.bundled import BundledProductProvider
    from gaia_dr4_explorer.data.catalog import reference_fits

    names = {sid: v.get("main_id", "") for sid, v in BundledProductProvider().simbad().items()}
    positions = {
        sid: xs.position_from_reference(values, name=names.get(sid, ""))
        for sid, values in reference_fits().items()
    }

    def search(pos: xs.SkyPosition) -> xs.SearchReport:
        report = xs.search_external_spectra(pos, timeout_s=timeout_s)
        summary = ", ".join(f"{r.archive} {r.status.value}"
                            + (f" {r.n_spectra}" if r.n_spectra else "") for r in report.results)
        print(f"({pos.ra_deg:9.4f}, {pos.dec_deg:+8.4f}) {pos.name or '-':28} {summary}",
              flush=True)
        return report

    snapshot = xs.build_snapshot(positions, search)
    target = out or Path(str(files("gaia_dr4_explorer.resources") / xs.SNAPSHOT_RESOURCE))
    target.write_text(json.dumps(snapshot, indent=1, allow_nan=False) + "\n")
    incomplete = [sid for sid, rep in snapshot["sources"].items()
                  if not xs.SearchReport.from_dict(rep).complete()]
    print(f"wrote {target} ({len(snapshot['sources'])} sources"
          + (f"; incomplete searches for {len(incomplete)}: {incomplete}" if incomplete else "")
          + ")")
    return 0

