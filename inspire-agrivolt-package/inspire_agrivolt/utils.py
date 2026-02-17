def visualize_empty_data(gid_ds, state: str) -> None:
    """
    plot albedo mean for a state to visualize holes in our data.

    gid_ds: xr.Datset
        dataset with gid and time dimensions and albedo datavariable
    state: str
        state name (title case as appears in NSRDB metadata)
    """
    import pvdeg
    import xarray as xr
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata

    weather_db = "NSRDB"
    weather_arg = {
        "satellite": "Americas",
        "names": "TMY",
        "NREL_HPC": True,
    }

    geo_weather, geo_meta = pvdeg.weather.get(
        weather_db, geospatial=True, **weather_arg
    )

    state_meta = geo_meta[geo_meta["state"] == f"{state.title()}"]

    geo_df = state_meta[["latitude", "longitude"]].copy()
    geo_df.index.name = "gid"
    geo_ds = geo_df.to_xarray()

    lat_long_ds = gid_ds.merge(geo_ds)

    # Step 1: Extract scattered points
    lat = lat_long_ds.latitude.values
    lon = lat_long_ds.longitude.values
    albedo_mean = lat_long_ds.albedo.mean(dim="time").values

    # Step 2: Create a regular grid
    num_points = 300  # adjust resolution based on performance needs
    lon_grid = np.linspace(lon.min(), lon.max(), num_points)
    lat_grid = np.linspace(lat.min(), lat.max(), num_points)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

    # Step 3: Interpolate scattered data onto the grid
    grid_values = griddata(
        points=(lon, lat), values=albedo_mean, xi=(lon_mesh, lat_mesh), method="linear"
    )

    # Step 4: Plot the interpolated image
    plt.figure(figsize=(10, 8))
    pcm = plt.pcolormesh(
        lon_mesh, lat_mesh, grid_values, shading="auto", cmap="viridis"
    )
    plt.colorbar(pcm, label="Mean Albedo")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Interpolated Mean Albedo Over Time")
    plt.show()
