from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import warnings

from map import (
    RENDER_MODES,
    get_plot_metadata,
    normalize_mode_and_month,
    select_field,
)
from name_map import (
    convert_kestrel_name_to_published_name,
    convert_published_name_to_kestrel_name,
)


LIVE_DATA_COPY = Path("/projects/inspire/PySAM-MAPS/v1.2/")
DATA_DIR = LIVE_DATA_COPY / "final-backup"
DEFAULT_CMAP = "inferno"
DEFAULT_LABEL = "Mean Edge-to-Edge Value"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate standalone colorbar images for existing map plots."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing per-configuration zarr stores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to write standalone colorbar images into.",
    )
    parser.add_argument(
        "--mode",
        choices=RENDER_MODES,
        default="mean-edgetoedge",
        help=(
            "Field selection to summarize for the colorbar. "
            "'mean-edgetoedge' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        default=None,
        metavar="N",
        help=(
            "Optional month (1-12) for mean-daily-insolation and shading-factor. "
            "Omit for annual / full-year aggregation."
        ),
    )
    parser.add_argument(
        "--months",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Periods to include when aggregating ranges. Use 'annual' (or '0') and/or "
            "month numbers 1-12. Implies scanning each period across --configs."
        ),
    )
    parser.add_argument(
        "--all-periods",
        action="store_true",
        help="Aggregate over annual plus months 1-12 (for time-varying modes).",
    )
    parser.add_argument(
        "--aggregate-range",
        action="store_true",
        help=(
            "Compute global min/max across --configs and selected periods without "
            "writing per-config colorbar PNGs."
        ),
    )
    parser.add_argument(
        "--nice-bounds",
        action="store_true",
        help="Round aggregated (or explicit) cmin down and cmax up to nice values.",
    )
    parser.add_argument(
        "--bounds-file",
        type=Path,
        default=None,
        help="Optional path to write CMIN/CMAX as a shell-sourceable bounds.env file.",
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Colorbar label text. Defaults to the map metadata label for --mode.",
    )
    parser.add_argument(
        "--orientation",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="Standalone colorbar orientation.",
    )
    parser.add_argument(
        "--cmin",
        type=float,
        default=None,
        help="Optional lower bound for the colorbar range.",
    )
    parser.add_argument(
        "--cmax",
        type=float,
        default=None,
        help="Optional upper bound for the colorbar range.",
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        required=True,
        help="Configuration numbers to render.",
    )
    parser.add_argument(
        "--config-names",
        choices=("kestrel-names", "publication-names"),
        required=True,
        help=(
            "Whether --configs are the original Kestrel/on-disk names or the "
            "publication names."
        ),
    )
    parser.add_argument(
        "--shared-colorbar-name",
        type=str,
        default=None,
        help=(
            "If set with --cmin/--cmax (or after aggregation), write a single shared "
            "colorbar PNG with this filename into --output-dir."
        ),
    )
    return parser.parse_args()


def parse_periods(
    *,
    mode: str,
    month: int | None,
    months: list[str] | None,
    all_periods: bool,
) -> list[int | None]:
    mode, month = normalize_mode_and_month(mode, month)

    if all_periods:
        if mode not in {"mean-daily-insolation", "shading-factor"}:
            raise ValueError(f"--all-periods is not supported for mode={mode}.")
        return [None, *range(1, 13)]

    if months is not None:
        if mode not in {"mean-daily-insolation", "shading-factor"}:
            raise ValueError(f"--months is not supported for mode={mode}.")
        periods: list[int | None] = []
        for token in months:
            lowered = token.lower()
            if lowered in {"annual", "0"}:
                periods.append(None)
                continue
            value = int(token)
            if value < 1 or value > 12:
                raise ValueError(f"Invalid month in --months: {token}")
            periods.append(value)
        if not periods:
            raise ValueError("--months must include at least one period.")
        return periods

    return [month]


def nice_bounds(vmin: float, vmax: float) -> tuple[float, float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        raise ValueError(f"Cannot round non-finite bounds: vmin={vmin}, vmax={vmax}")
    if vmin > vmax:
        raise ValueError(f"vmin ({vmin}) must be <= vmax ({vmax})")

    span = vmax - vmin
    if span == 0:
        magnitude = abs(vmax) if vmax != 0 else 1.0
        step = 10 ** math.floor(math.log10(magnitude))
        return math.floor(vmin / step) * step, math.ceil(vmax / step) * step + step

    raw_step = 10 ** math.floor(math.log10(span))
    candidates = [raw_step, 2 * raw_step, 5 * raw_step, 10 * raw_step]
    step = min(candidates, key=lambda candidate: abs(span / candidate - 5))

    cmin = math.floor(vmin / step) * step
    cmax = math.ceil(vmax / step) * step
    if cmin == cmax:
        cmax = cmin + step
    return float(cmin), float(cmax)


def compute_plot_range(
    conf_path: Path,
    mode: str,
    *,
    month: int | None = None,
) -> tuple[float, float]:
    conf_ds = xr.open_zarr(conf_path)
    field = select_field(
        conf_ds,
        mode=mode,
        month=month,
    )
    vmin = float(field.min().compute())
    vmax = float(field.max().compute())
    return vmin, vmax


def aggregate_plot_range(
    config_paths: dict[int, Path],
    requested_configs: list[tuple[int, int]],
    *,
    mode: str,
    periods: list[int | None],
) -> tuple[float, float]:
    global_min = math.inf
    global_max = -math.inf

    for kestrel_config_number, publication_config_number in requested_configs:
        if kestrel_config_number not in config_paths:
            raise ValueError(
                f"Kestrel configuration {kestrel_config_number} was requested, but only "
                f"{min(config_paths)} through {max(config_paths)} are available."
            )
        conf_path = config_paths[kestrel_config_number]
        for period in periods:
            vmin, vmax = compute_plot_range(conf_path, mode, month=period)
            print(
                "range "
                f"publication_config={publication_config_number} "
                f"kestrel_config={kestrel_config_number} "
                f"month={period if period is not None else 'annual'} "
                f"vmin={vmin} vmax={vmax}"
            )
            global_min = min(global_min, vmin)
            global_max = max(global_max, vmax)

    if not math.isfinite(global_min) or not math.isfinite(global_max):
        raise ValueError("Failed to compute aggregate range across requested configs.")
    return float(global_min), float(global_max)


def load_config_paths(data_dir: Path) -> dict[int, Path]:
    confs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    return {
        config_number: conf_path
        for config_number, conf_path in enumerate(confs, start=1)
    }


def resolve_config_names(
    requested_configs: list[int],
    *,
    config_names: str,
) -> list[tuple[int, int]]:
    if config_names == "publication-names":
        warnings.warn(
            "Interpreting --configs as publication names and mapping them to "
            "Kestrel/on-disk names before loading zarr stores.",
            stacklevel=2,
        )
        return [
            (convert_published_name_to_kestrel_name(config_number), config_number)
            for config_number in requested_configs
        ]

    warnings.warn(
        "Interpreting --configs as Kestrel/on-disk names. Output filenames will "
        "use the corresponding publication names, which may differ. See the "
        "inspire-agrivolt-package README.md deployment section for details.",
        stacklevel=2,
    )
    return [
        (config_number, convert_kestrel_name_to_published_name(config_number))
        for config_number in requested_configs
    ]


def write_colorbar(
    output_path: Path,
    *,
    vmin: float,
    vmax: float,
    cmap: str,
    label: str,
    orientation: str,
) -> None:
    if orientation == "vertical":
        figsize = (2.4, 8.0)
    else:
        figsize = (8.0, 2.0)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.35, right=0.75, bottom=0.08, top=0.98)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ax.set_visible(False)
    cbar = fig.colorbar(sm, ax=ax, orientation=orientation)
    cbar.set_label(label)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)


def get_output_name(
    config_number: int,
    cmap: str,
    mode: str,
    *,
    month: int | None = None,
    fixed_crange: bool = False,
) -> str:
    mode, month = normalize_mode_and_month(mode, month)
    suffix = "-fixed-crange" if fixed_crange else ""
    if mode == "mean-edgetoedge":
        return f"config-{config_number:02d}-{cmap}-colorbar{suffix}.png"
    if mode in {"mean-daily-insolation", "shading-factor"}:
        period = "annual" if month is None else f"m{month:02d}"
        return f"config-{config_number:02d}-{mode}-{period}-{cmap}-colorbar{suffix}.png"
    return f"config-{config_number:02d}-{mode}-{cmap}-colorbar{suffix}.png"


def write_bounds_file(path: Path, *, cmin: float, cmax: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"CMIN={cmin}\nCMAX={cmax}\n", encoding="utf-8")


def default_label_for_mode(mode: str, month: int | None) -> str:
    return get_plot_metadata(mode, mode, month=month)["clabel"]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode, month = normalize_mode_and_month(args.mode, args.month)
    if month is not None and mode not in {"mean-daily-insolation", "shading-factor"}:
        raise ValueError(
            f"--month is only supported for mean-daily-insolation and shading-factor, "
            f"not {mode}."
        )

    if (args.cmin is None) ^ (args.cmax is None):
        raise ValueError("Provide both --cmin and --cmax, or neither.")
    if args.cmin is not None and args.cmin > args.cmax:
        raise ValueError("--cmin must be less than or equal to --cmax.")

    requested_configs = resolve_config_names(
        args.configs,
        config_names=args.config_names,
    )
    periods = parse_periods(
        mode=mode,
        month=month,
        months=args.months,
        all_periods=args.all_periods,
    )
    label = args.label or default_label_for_mode(mode, month)

    config_paths = None
    if args.cmin is None or args.aggregate_range:
        config_paths = load_config_paths(args.data_dir)

    if args.aggregate_range:
        assert config_paths is not None
        vmin, vmax = aggregate_plot_range(
            config_paths,
            requested_configs,
            mode=mode,
            periods=periods,
        )
        if args.nice_bounds:
            vmin, vmax = nice_bounds(vmin, vmax)
        print(f"cmin={vmin} cmax={vmax}")
        if args.bounds_file is not None:
            write_bounds_file(args.bounds_file, cmin=vmin, cmax=vmax)
            print(f"wrote bounds file {args.bounds_file}")

        if args.shared_colorbar_name is not None:
            output_path = args.output_dir / args.shared_colorbar_name
            print(f"writing shared colorbar {output_path.name}: vmin={vmin}, vmax={vmax}")
            write_colorbar(
                output_path,
                vmin=vmin,
                vmax=vmax,
                cmap=args.cmap,
                label=label,
                orientation=args.orientation,
            )
        return

    if args.cmin is not None:
        vmin, vmax = args.cmin, args.cmax
        if args.nice_bounds:
            vmin, vmax = nice_bounds(vmin, vmax)
        if args.bounds_file is not None:
            write_bounds_file(args.bounds_file, cmin=vmin, cmax=vmax)
            print(f"wrote bounds file {args.bounds_file}")
        if args.shared_colorbar_name is not None:
            output_path = args.output_dir / args.shared_colorbar_name
            print(
                f"writing shared colorbar {output_path.name}: "
                f"vmin={vmin}, vmax={vmax}"
            )
            write_colorbar(
                output_path,
                vmin=vmin,
                vmax=vmax,
                cmap=args.cmap,
                label=label,
                orientation=args.orientation,
            )
            return

    for kestrel_config_number, publication_config_number in requested_configs:
        if args.cmin is None:
            assert config_paths is not None
            if kestrel_config_number not in config_paths:
                raise ValueError(
                    f"Kestrel configuration {kestrel_config_number} was requested, but only "
                    f"{min(config_paths)} through {max(config_paths)} are available."
                )

            conf_path = config_paths[kestrel_config_number]
            # Per-config colorbars use a single period (args.month / normalized).
            vmin, vmax = compute_plot_range(
                conf_path,
                mode=mode,
                month=month,
            )
            if args.nice_bounds:
                vmin, vmax = nice_bounds(vmin, vmax)
        else:
            vmin, vmax = args.cmin, args.cmax
            if args.nice_bounds:
                vmin, vmax = nice_bounds(vmin, vmax)
            print(
                "using explicit colorbar bounds "
                f"cmin={vmin}, cmax={vmax} for publication_config={publication_config_number}, "
                f"kestrel_config={kestrel_config_number}"
            )

        output_path = args.output_dir / get_output_name(
            config_number=publication_config_number,
            cmap=args.cmap,
            mode=mode,
            month=month,
            fixed_crange=args.cmin is not None,
        )
        print(
            f"writing {output_path.name}: publication_config={publication_config_number}, "
            f"kestrel_config={kestrel_config_number}, vmin={vmin}, vmax={vmax}"
        )
        write_colorbar(
            output_path,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
            label=label,
            orientation=args.orientation,
        )


if __name__ == "__main__":
    main()
