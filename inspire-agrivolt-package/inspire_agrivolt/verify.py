import xarray as xr
import dask.array as da
import numpy as np
from IPython.display import display, Markdown
from collections.abc import Iterable
from typing import Optional

# def verify_dataset_gids(irradiance_ds: xr.Dataset):
#     """
#     verify that a dataset contains valid albedo entries for each gid (lat, lon pair). Raise ValueError if it doesnt and print gids to be recomputed.

#     verified datasets pass silently.
#     """
#     # it will be worth checking all variables, if any are null then we say we need to recompute at that gid.
#     ref_var = "albedo" # comes from nsrdb -> pysam -> outputs

#     mask = merged_with_gids[ref_var].mean(dim="time").isnull()
#     skipped_gids = merged_with_gids.where(mask.compute(), drop=False).gid

#     # visual representation of what was skipped
#     #skipped_gids.plot()

#     gids_arr = skipped_gids.values
#     gids_arr = gids_arr[~np.isnan(gids_arr)].astype(int)

#     if len(gids_arr) > 0:
#         raise ValueError(
#             """Dataset albedo incomplete at the following gids:""",
#             {gids_arr}
#             )


def summarize_nans_nd(
    ds: xr.Dataset,
    profile_dims=("gid", "time", "distance"),
    top_k: int = 10,
):
    """
    Summarize NaNs per variable for arbitrary N-D variables.
    Nicely formatted Markdown tables when run in Jupyter.
    """
    report = {}

    for var in ds.data_vars:
        da = ds[var]
        dims = tuple(da.dims)
        nan_mask = da.isnull()

        # Totals
        total_nans = int(nan_mask.sum().compute().item())
        total_elems = int(np.prod([da.sizes[d] for d in dims])) if dims else 1
        nan_frac = (total_nans / total_elems) if total_elems else 0.0

        entry = {
            "dims_present": list(dims),
            "total_nans": total_nans,
            "nan_frac": nan_frac,
        }

        for dim in profile_dims:
            if dim not in dims:
                continue

            other_dims = tuple(d for d in dims if d != dim)

            # Any NaN along other dims
            if other_dims:
                any_along = nan_mask.any(dim=other_dims)
            else:
                any_along = nan_mask

            affected = int(any_along.sum().compute().item())
            frac = affected / da.sizes[dim] if da.sizes[dim] else 0.0

            # Per-index counts
            if other_dims:
                per_index_nan = nan_mask.sum(dim=other_dims).compute().values
            else:
                per_index_nan = nan_mask.astype("int64").compute().values

            top_list = []
            if per_index_nan.size:
                pos = np.nonzero(per_index_nan > 0)[0]
                if pos.size:
                    k = min(top_k, pos.size)
                    counts_pos = per_index_nan[pos]
                    top_sel = np.argpartition(counts_pos, -k)[-k:]
                    top_sel = top_sel[np.argsort(counts_pos[top_sel])[::-1]]
                    coord_vals = ds[dim].values

                    def _to_py(v):
                        try:
                            return v.item()
                        except Exception:
                            return v

                    top_list = [
                        (_to_py(coord_vals[pos[i]]), int(counts_pos[i]))
                        for i in top_sel
                    ]

            entry[f"by_{dim}"] = {
                "affected": affected,
                "frac": frac,
                "top": top_list,
            }

        report[var] = entry

    # --- Pretty print in Jupyter ---
    for var, r in report.items():
        md = []
        md.append(f"### Variable: `{var}`  *(dims={r['dims_present']})*")
        md.append(f"- **Total NaNs**: {r['total_nans']:,} ({r['nan_frac']:.4%})")

        for dim in profile_dims:
            key = f"by_{dim}"
            if key in r:
                blk = r[key]
                md.append(
                    f"\n**Along `{dim}`**  \n"
                    f"- Affected: {blk['affected']:,} ({blk['frac']:.2%})"
                )

                if blk["top"]:
                    rows = ["| index | NaN count |", "|-------|-----------|"]
                    for idx, cnt in blk["top"][:top_k]:
                        rows.append(f"| {idx} | {cnt:,} |")
                    md.append("\n".join(rows))

        display(Markdown("\n".join(md)))


def gids_with_time_nans(
    ds: xr.Dataset,
    gid_name: str = "gid",
    time_name: str = "time",
    include_vars: Optional[Iterable[str]] = None,
    exclude_vars: Optional[Iterable[str]] = None,
    return_details: bool = True,
) -> tuple[np.ndarray, Optional[dict[str, int]]]:
    """
    Find the superset of gids that have NaNs along the time dimension for any variable.

    For each data variable that contains both `gid` and `time`:
      - reduce `isnull()` over `time` AND all other dims except `gid`
      - this yields a 1-D boolean mask over `gid` for that variable:
          True => this gid has at least one NaN somewhere along time (and any other dims)
    We OR these masks across variables, then return the gids where the union is True.

    Parameters
    ----------
    ds : xr.Dataset
        Dask-backed dataset.
    gid_name : str
        Name of the gid dimension.
    time_name : str
        Name of the time dimension.
    include_vars : Optional[Iterable[str]]
        If provided, limit the check to these variables.
    exclude_vars : Optional[Iterable[str]]
        If provided, skip these variables.
    return_details : bool
        If True, also return a dict {var: num_gids_with_time_nans}.

    Returns
    -------
    affected_gids : np.ndarray
        Array of gid coordinate values where ANY variable has NaNs across time.
    per_var_counts : Optional[Dict[str, int]]
        Count of affected gids per variable (only if return_details=True).
    """
    if gid_name not in ds.dims:
        raise ValueError(f"Dataset has no '{gid_name}' dimension.")
    if time_name not in ds.dims:
        # No global time dim; nothing to do
        return np.array([], dtype=ds[gid_name].dtype), ({} if return_details else None)

    # Choose vars
    vars_to_check = list(ds.data_vars)
    if include_vars is not None:
        include_set = set(include_vars)
        vars_to_check = [v for v in vars_to_check if v in include_set]
    if exclude_vars is not None:
        exclude_set = set(exclude_vars)
        vars_to_check = [v for v in vars_to_check if v not in exclude_set]

    union_mask = None  # dask.DataArray[gid] of bools
    per_var_masks = {}  # store dask bool masks for optional counts

    for var in vars_to_check:
        da = ds[var]
        dims = da.dims

        # Only consider variables that have both gid and time
        if gid_name not in dims or time_name not in dims:
            continue

        # Reduce over time AND all non-gid, non-time dims (e.g., distance)
        other_dims = tuple(d for d in dims if d not in (gid_name, time_name))
        # Any NaN anywhere along time (and other dims) for each gid -> 1D over gid
        var_bad_gids = (
            da.isnull().any(dim=(time_name,) + other_dims)
            if other_dims
            else da.isnull().any(dim=time_name)
        )

        union_mask = var_bad_gids if union_mask is None else (union_mask | var_bad_gids)
        if return_details:
            per_var_masks[var] = var_bad_gids

    if union_mask is None:
        # No vars had both gid and time
        return np.array([], dtype=ds[gid_name].dtype), ({} if return_details else None)

    # Compute superset once
    union_mask_v = union_mask.compute().values  # shape: (n_gid,)
    affected_gids = ds[gid_name].values[union_mask_v]

    if return_details:
        # Count affected gids per var (each a cheap reduction)
        per_var_counts = {
            v: int(m.sum().compute().item()) for v, m in per_var_masks.items()
        }
        return affected_gids, per_var_counts

    return affected_gids, None
