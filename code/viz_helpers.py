import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pyvista as pv
from matplotlib.colors import ListedColormap, LogNorm
from scipy.interpolate import interp1d
from pathlib import Path

def plot_curtain_slice(df, x1, y1, x2, y2, avg_doi, title=str(None), tol=1.0, dz=0.1, method='linear', cmap='rainbow', vmin=None, vmax=None):
    """
    Plot a vertical curtain slice using 1D interpolation at each horizontal location
    and optionally extrapolate only up to avg_doi.

    pcolormesh is a fast, efficient way to visualize 2D gridded data
    Unlike imshow, which assumes regularly spaced grids in both directions, 
    pcolormesh allows non-uniform grids.
    Provides a continuous surface look instead of a scatter of points.    

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['x', 'y', 'z', 'resistivity'].
    x1, y1, x2, y2 : float
        Coordinates of the slice line.
    avg_doi : float
        Maximum allowed extrapolation depth below measured data.
    tol : float, optional
        Distance tolerance around the line.
    dz : float, optional
        Vertical resolution for plotting grid.
    method : str, optional
        Interpolation method: 'linear', 'nearest', 'cubic', etc.
    cmap : str, optional
        Colormap for pcolormesh.
    vmin, vmax : float, optional
        Min and max for color scale.

    Returns:
    ----------
    None: 
        Displays a 2D curtain plot of resistivity along the slice line.
    """
    # Compute line vector
    dx, dy = x2 - x1, y2 - y1
    line_len = np.sqrt(dx**2 + dy**2)
    ux, uy = dx / line_len, dy / line_len

    # Project points onto line
    rel_x = df['x'] - x1
    rel_y = df['y'] - y1
    dist_along = rel_x * ux + rel_y * uy
    dist_perp = np.abs(-dy*rel_x + dx*rel_y) / line_len

    # Filter points near the line
    mask = (dist_perp <= tol) & (dist_along >= 0) & (dist_along <= line_len)
    slice_df = df.loc[mask].copy()
    slice_df['dist'] = dist_along[mask]

    if slice_df.empty:
        print("No points found along this slice.")
        return

    # Define grid
    unique_dist = np.sort(slice_df['dist'].unique())
    z_min, z_max = slice_df['z'].min(), slice_df['z'].max() + avg_doi
    z_grid = np.arange(z_min, z_max + dz, dz)
    grid_values = np.full((len(z_grid), len(unique_dist)), np.nan)

    # Interpolate at each horizontal location
    for i, d in enumerate(unique_dist):
        col_df = slice_df[slice_df['dist'] == d]
        if len(col_df) >= 2:
            f = interp1d(col_df['z'], col_df['resistivity'],
                         kind=method, bounds_error=False, fill_value=np.nan)
            interp_vals = f(z_grid)
            
            # Limit extrapolation below lowest measured z to avg_doi
            min_z_meas = col_df['z'].min()
            interp_vals[z_grid < min_z_meas - avg_doi] = np.nan
            
            grid_values[:, i] = interp_vals
        elif len(col_df) == 1:
            idx = np.abs(z_grid - col_df['z'].values[0]).argmin()
            grid_values[idx, i] = col_df['resistivity'].values[0]

    # Plot using pcolormesh
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(
        unique_dist, z_grid, grid_values,
        shading='auto', cmap=cmap,
        # vmin=vmin, vmax=vmax
        norm=LogNorm(vmin=vmin, vmax=vmax)
    )
    cbar = fig.colorbar(im, ax=ax, label='Resistivity (Ω·m)')
    ax.set_xlabel('Distance along line (m)')
    ax.set_ylabel('Elevation (m)')
    if title is None:
        ax.set_title('Curtain Slice through 3D Subsurface Model')
    else:
        ax.set_title(title)
    ax.set_ylim(z_grid.min() - 0.5, z_grid.max() - 2)
    ax.set_aspect('equal')
    plt.show()


def voxelise_csv(df, voxel_size, em_survey_date, output_path, origin=None):
    """
    Voxelize a point cloud and optionally align to a common origin.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ['x', 'y', 'z', 'resistivity'].
    voxel_size : float
        Size of each voxel.
    em_survey_date : str
        Identifier used in output filename.
    output_path : str or Path
        Directory where output .vtp file will be saved.
    origin : array-like, optional
        Common origin to align all point clouds. If None, uses the minimum coordinates of this df.

    Returns
    -------
    pv.PolyData
        Voxelized PolyData cloud.
    """
    df = df.copy()

    if origin is None:
        origin = df[['x','y','z']].min().values
    df[['x','y','z']] -= origin

    # Compute voxel indices
    df['vx'] = (df['x'] / voxel_size).astype(int)
    df['vy'] = (df['y'] / voxel_size).astype(int)
    df['vz'] = (df['z'] / voxel_size).astype(int)

    # Aggregate by voxel
    df_voxel = df.groupby(['vx','vy','vz'], as_index=False).agg({
        'x':'mean', 'y':'mean', 'z':'mean', 'resistivity':'mean'
    })

    # Export to VTP
    points = df_voxel[['x','y','z']].to_numpy(dtype=np.float32)
    cloud = pv.PolyData(points)
    cloud['resistivity'] = df_voxel['resistivity'].to_numpy(dtype=np.float32)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{em_survey_date}_pred_ohm_m_voxel.vtp"
    cloud.save(out_file)

    reduction = (1 - len(df_voxel) / len(df)) * 100
    print(f"Voxelized {em_survey_date}: {len(df)} → {len(df_voxel)} points ({reduction:.1f}% reduction). Saved to {out_file}")

    return cloud, origin


def plot_3d_voxel(vtp_filepath, vmin, vmax):
    """
    Load a voxelized VTP file and visualize 3D resistivity values using PyVista.
    
    For values outside the specified range: make gray and black.
    Resistivity values are plotted on a logarithmic scale.

    Parameters
    ----------
    vtp_filepath : str or Path
        Path to the VTP file containing voxelized point cloud data with resistivity vals.
    vmin : float
        Minimum resistivity value for color scaling. 
    vmax : float
        Maximum resistivity value for color scaling. 

    Returns
    -------
    None

    """
    # Load mesh
    loaded_mesh = pv.read(vtp_filepath)

    scalars = loaded_mesh['resistivity'].copy()
    scalars[scalars < vmin] = vmin
    scalars[scalars > vmax] = vmax

    # Plot
    loaded_mesh.plot(
        scalars=scalars,
        cmap='rainbow',
        point_size=5,
        log_scale=True,
        clim=[vmin, vmax],
        scalar_bar_args={
            'title': 'Resistivity (ohm-m)',
            'n_labels': 3,
            'fmt': '%.0e'
        }
    )

def compute_delta_resistivity(cloud_ref, cloud_new, voxel_size=0.5):
    """
    Compute absolute change in resistivity between two voxelized clouds
    using vectorized intersection `np.intersect1d` method.

    Parameters
    ----------
    cloud_ref : pv.PolyData or str/Path
        Reference survey voxelized cloud with 'resistivity'
        Can be a PolyData object or path to a .vtp file.
    cloud_new : pv.PolyData or str/Path
        New survey voxelized cloud with 'resistivity' (e.g., later time).
        Can be a PolyData object or path to a .vtp file.
    voxel_size : float, default 0.5
        Size of each voxel. Must match the voxelization used for both clouds.

    Returns
    -------
    pv.PolyData
        PolyData cloud with 'delta_resistivity' (absolute change) for overlapping voxels.
        Computed as (cloud_new - cloud_ref).
    """

    # Auto-load from file if input is a path
    if isinstance(cloud_ref, (str, Path)):
        cloud_ref = pv.read(cloud_ref)
    if isinstance(cloud_new, (str, Path)):
        cloud_new = pv.read(cloud_new)

    # Snap coordinates to the voxel grid
    # Ensures alignment to the same voxel grid for intersection
    coords_ref = np.round(cloud_ref.points / voxel_size).astype(int)
    coords_new = np.round(cloud_new.points / voxel_size).astype(int)

    # Create structured views of coordinates for fast comparison
    # Each row (x, y, z) becomes a fixed-size binary blob for np.intersect1d 
    dtype = np.dtype((np.void, coords_ref.dtype.itemsize * coords_ref.shape[1]))
    ref_view = np.ascontiguousarray(coords_ref).view(dtype)
    new_view = np.ascontiguousarray(coords_new).view(dtype)

    # Find intersecting voxels and their indices
    common, ref_idx, new_idx = np.intersect1d(ref_view, new_view, return_indices=True)

    # Compute absolute change in resistivity
    delta_vals = cloud_new['resistivity'][new_idx] - cloud_ref['resistivity'][ref_idx]
    delta_points = cloud_new.points[new_idx]

    # Create output PolyData with delta_resistivity
    delta_cloud = pv.PolyData(delta_points)
    delta_cloud['delta_resistivity'] = delta_vals

    print(f"Δ-resistivity (absolute) computed for {len(delta_points)} voxels (intersection of clouds).")
    return delta_cloud