from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr

from map import select_field


LIVE_DATA_COPY = Path("/projects/inspire/PySAM-MAPS/v1.2/")
DATA_DIR = LIVE_DATA_COPY / "final-backup"
DEFAULT_CMAP = "inferno"
DEFAULT_LABEL = "Mean edge-to-edge value"
DEFAULT_START_CONFIG = 1
DEFAULT_STOP_CONFIG = 11


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
        choices=(
            "mean-edgetoedge",
            "july-shading-factor-setup1",
            "july-21-15-edgetoedge-setup1",
        ),
        default="mean-edgetoedge",
        help=(
            "Field selection to summarize for the colorbar. "
            "'mean-edgetoedge' preserves the current behavior."
        ),
    )
    parser.add_argument(
        "--cmap",
        default=DEFAULT_CMAP,
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="Colorbar label text.",
    )
    parser.add_argument(
        "--orientation",
        choices=("vertical", "horizontal"),
        default="vertical",
        help="Standalone colorbar orientation.",
    )
    parser.add_argument(
        "--start-config",
        type=int,
        default=DEFAULT_START_CONFIG,
        help="First configuration number to render, inclusive.",
    )
    parser.add_argument(
        "--stop-config",
        type=int,
        default=DEFAULT_STOP_CONFIG,
        help="Last configuration number to render, inclusive.",
    )
    return parser.parse_args()


def compute_plot_range(conf_path: Path, config_number: int, mode: str) -> tuple[float, float]:
    conf_ds = xr.open_zarr(conf_path)
    field = select_field(
        conf_ds,
        config_number=config_number,
        mode=mode,
    )
    vmin = float(field.min().compute())
    vmax = float(field.max().compute())
    return vmin, vmax


def load_config_paths(data_dir: Path) -> dict[int, Path]:
    confs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    return {
        config_number: conf_path
        for config_number, conf_path in enumerate(confs, start=1)
    }


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


def get_output_name(config_number: int, cmap: str, mode: str) -> str:
    if mode == "mean-edgetoedge":
        return f"config-{config_number:02d}-{cmap}-colorbar.png"
    return f"config-{config_number:02d}-{mode}-{cmap}-colorbar.png"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_config > args.stop_config:
        raise ValueError("--start-config must be less than or equal to --stop-config")

    if args.mode != "mean-edgetoedge" and (args.start_config != 1 or args.stop_config != 1):
        raise ValueError(
            f"--mode={args.mode} only supports setup/configuration 1. "
            "Use --start-config 1 --stop-config 1."
        )

    config_paths = load_config_paths(args.data_dir)

    requested_configs = range(args.start_config, args.stop_config + 1)

    for config_number in requested_configs:
        if config_number not in config_paths:
            raise ValueError(
                f"Configuration {config_number} was requested, but only "
                f"{min(config_paths)} through {max(config_paths)} are available."
            )

        conf_path = config_paths[config_number]
        vmin, vmax = compute_plot_range(
            conf_path,
            config_number=config_number,
            mode=args.mode,
        )
        output_path = args.output_dir / get_output_name(
            config_number=config_number,
            cmap=args.cmap,
            mode=args.mode,
        )
        print(f"writing {output_path.name}: vmin={vmin}, vmax={vmax}")
        write_colorbar(
            output_path,
            vmin=vmin,
            vmax=vmax,
            cmap=args.cmap,
            label=args.label,
            orientation=args.orientation,
        )


if __name__ == "__main__":
    main()
