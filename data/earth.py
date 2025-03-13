# earth.py

import torch
import xarray as xr
import numpy as np
from torch.utils.data import TensorDataset


def sph_from_deg(lat_deg, lon_deg):
    """
    Convert latitude and longitude (in degrees) to Cartesian coordinates on the unit sphere.

    Args:
        lat_deg (np.ndarray): Latitudes in degrees.
        lon_deg (np.ndarray): Longitudes in degrees.

    Returns:
        np.ndarray: Array of shape (..., 3) with [x, y, z] coordinates on the unit sphere.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def load_land_mask(data_path):
    """
    Load a land-sea mask from a NetCDF file and extract Cartesian coordinates of land points.

    Assumes the NetCDF file has:
      - A binary mask variable named 'mask' (land=1, sea=0).
      - 1D coordinate arrays 'lat' and 'lon'.
      - If 'mask' has extra dimensions, the first slice is used.

    Args:
        data_path (str): Path to the NetCDF file.

    Returns:
        np.ndarray: Array of shape (N, 3) containing land points on the unit sphere.
    """
    ds = xr.open_dataset(data_path)

    # 'mask' might have shape (extra, n_lat, n_lon) or just (n_lat, n_lon).
    mask = ds["mask"].values
    if mask.ndim > 2:
        mask = mask[0, :, :]  # Take the first slice if there's an extra dimension

    # lat: shape (n_lat,), lon: shape (n_lon,)
    lats = ds["lat"].values
    lons = ds["lon"].values

    # Create 2D arrays of lat/lon
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')  # shape (n_lat, n_lon)

    # Where mask == 1 means land
    land_indices = np.where(mask == 1)
    land_lats = lat_grid[land_indices]
    land_lons = lon_grid[land_indices]

    # Convert lat/lon land coords into cartesian (x, y, z) on the unit sphere
    land_points = sph_from_deg(land_lats, land_lons)  # shape (N, 3)
    return land_points


def sample_earth_land(data_path, num_samples):
    """
    Sample 3D points on Earth's land mass.

    Args:
        data_path (str): Path to the NetCDF land-sea mask file.
        num_samples (int): Number of land points to sample.

    Returns:
        torch.Tensor: (num_samples, 3) float32 points in 3D.
    """
    land_points = load_land_mask(data_path)  # shape: (all_land_points, 3)
    print(f'land_points.shape:{land_points.shape}')
    total_land = len(land_points)
    if total_land == 0:
        raise ValueError(f"No land points found in the provided mask: {data_path}")

    # If num_samples is greater than the number of available land points,
    # sample with replacement, otherwise sample without replacement.
    replace = num_samples > total_land
    sampled_indices = np.random.choice(total_land, size=num_samples, replace=replace)
    sampled_land_points = land_points[sampled_indices]

    return torch.tensor(sampled_land_points, dtype=torch.float32)


class EarthDataset(TensorDataset):
    """
    A PyTorch Dataset that samples points on Earth's landmass (3D) from a netCDF land-sea mask.

    Usage:
        # in your config file/object
        config.data_path = "/path/to/your_landsea_mask.nc"
        config.data.data_samples = 10000
        ...

        dataset = EarthDataset(config)
        # dataset[i] -> returns a single (3,) point on Earth's land.
    """
    def __init__(self, config):
        """
        Args:
            config: A config object or dictionary that must have:
                - config.data_path (str): path to the netCDF land-sea mask
                - config.data.data_samples (int): how many samples to draw
        """
        data_path = getattr(config, "data_path", None)
        if data_path is None:
            raise ValueError("config.data_path is not set. Provide a path to the land-sea mask netCDF.")

        data_samples = getattr(config, "data_samples", 10000)
        if data_samples is None:
            raise ValueError("config.data.data_samples is not set. Provide the number of samples to draw.")

        # Sample land points
        data_tensor = sample_earth_land(data_path, data_samples)

        # Now initialize parent (TensorDataset) with the data
        # TensorDataset usually expects (input, target, ...) but we can just pass one tensor
        super().__init__(data_tensor)

        # If you want to store it for any reason:
        self.data = data_tensor

    def __getitem__(self, index):
        # By default, TensorDataset calls self.tensors[0][index], etc.
        # If you just pass one tensor, you'll get that single point out.
        return super().__getitem__(index)

    def __len__(self):
        # TensorDataset implements this for you, so you could omit this method.
        return super().__len__()
