from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import warnings

from map import select_field
from name_map import (
    convert_kestrel_name_to_published_name,
    convert_published_name_to_kestrel_name,
)


LIVE_DATA_COPY = Path("/projects/inspire/PySAM-MAPS/v1.2/")
DATA_DIR = LIVE_DATA_COPY / "final-backup"
DEFAULT_CMAP = "inferno"
DEFAULT_LABEL = "Mean edge-to-edge value"


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
            "farmable-land-percent",
            "july-shading-factor",
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


def get_output_name(config_number: int, cmap: str, mode: str) -> str:
    if mode == "mean-edgetoedge":
        return f"config-{config_number:02d}-{cmap}-colorbar.png"
    return f"config-{config_number:02d}-{mode}-{cmap}-colorbar.png"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = load_config_paths(args.data_dir)
    requested_configs = resolve_config_names(
        args.configs,
        config_names=args.config_names,
    )

    for kestrel_config_number, publication_config_number in requested_configs:
        if kestrel_config_number not in config_paths:
            raise ValueError(
                f"Kestrel configuration {kestrel_config_number} was requested, but only "
                f"{min(config_paths)} through {max(config_paths)} are available."
            )

        conf_path = config_paths[kestrel_config_number]
        vmin, vmax = compute_plot_range(
            conf_path,
            config_number=publication_config_number,
            mode=args.mode,
        )
        output_path = args.output_dir / get_output_name(
            config_number=publication_config_number,
            cmap=args.cmap,
            mode=args.mode,
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
            label=args.label,
            orientation=args.orientation,
        )


if __name__ == "__main__":
    main()
