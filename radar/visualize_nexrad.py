import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import zarr
from pyproj import Transformer
import json


def load_preserve_boundary(target_crs, geojson_path='geometries/dangermond-preserve-boundary.geojson'):
    """
    Load preserve boundary from GeoJSON and transform to target CRS (e.g., UTM)
    
    Parameters:
    -----------
    target_crs : pyproj.CRS or str
        Target CRS to transform coordinates to (usually the radar grid CRS)
    
    Returns:
    --------
    boundary_x, boundary_y : arrays
        X and Y coordinates of boundary in target coordinate system (meters)
    """
    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Extract coordinates from first feature
        coords = geojson['features'][0]['geometry']['coordinates'][0]
        
        # GeoJSON coordinates are [lon, lat]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # Transform to target CRS (e.g., UTM)
        from pyproj import CRS
        wgs84 = CRS.from_epsg(4326)
        if isinstance(target_crs, str):
            target_crs = CRS.from_string(target_crs)
        
        transformer = Transformer.from_crs(wgs84, target_crs, always_xy=True)
        
        boundary_x = []
        boundary_y = []
        for lon, lat in zip(lons, lats):
            x, y = transformer.transform(lon, lat)
            boundary_x.append(x)
            boundary_y.append(y)
        
        return np.array(boundary_x), np.array(boundary_y)
        
    except Exception as e:
        print(f"⚠️  Could not load preserve boundary: {e}")
        return None, None


def get_weather_station_locations(target_crs, verbose=True):
    """
    Query database for weather station locations and transform to target CRS
    
    Parameters:
    -----------
    target_crs : pyproj.CRS or str
        Target CRS to transform coordinates to (usually the radar grid CRS)
    
    Returns:
    --------
    stations : list of dict
        Each dict contains: {'name': str, 'x': float, 'y': float, 'lat': float, 'lon': float}
        x, y are in target coordinate system (meters)
    """
    try:
        from database.config import connect, create_session
        from database.models.DendraStations import DendraStation, DendraDatastream
        
        engine = connect()
        session = create_session(engine)
        
        # Query stations that have rainfall data
        stations_with_rain = session.query(
            DendraStation.id,
            DendraStation.name,
            DendraStation.latitude,
            DendraStation.longitude
        ).join(
            DendraDatastream, DendraStation.id == DendraDatastream.station_id
        ).filter(
            DendraDatastream.name == 'Rainfall'
        ).distinct().all()
        
        if verbose:
            print(f"\n📍 Database query returned {len(stations_with_rain)} stations")
        
        if len(stations_with_rain) == 0:
            if verbose:
                print("⚠️  No weather stations with rainfall data found in database")
            return []
        
        # Set up coordinate transformer
        from pyproj import CRS
        wgs84 = CRS.from_epsg(4326)
        if isinstance(target_crs, str):
            target_crs = CRS.from_string(target_crs)
        
        transformer = Transformer.from_crs(wgs84, target_crs, always_xy=True)
        
        if verbose:
            print(f"   Target CRS: {target_crs}")
        
        # Transform station coordinates
        stations = []
        for station_id, name, lat, lon in stations_with_rain:
            if lat is not None and lon is not None:
                # Transform to target coordinates
                x, y = transformer.transform(lon, lat)
                stations.append({
                    'id': station_id,
                    'name': name,
                    'lat': lat,
                    'lon': lon,
                    'x': x,
                    'y': y
                })
                if verbose:
                    print(f"   {name}: ({lat:.5f}°N, {lon:.5f}°W) -> ({x:.0f}m, {y:.0f}m)")
        
        if verbose:
            print(f"\n✅ Transformed {len(stations)} stations to grid coordinates")
        
        return stations
        
    except ImportError:
        print("⚠️  Database module not available - cannot load station locations")
        return []
    except Exception as e:
        print(f"⚠️  Error loading station locations: {e}")
        return []


def check_station_overlap(zarr_path="KVBX_preserve_500m.zarr", patch_size_m=2640, verbose=True):
    """
    Check if multiple stations would be assigned to the same radar grid patch.
    
    Each training sample extracts a patch_size_m × patch_size_m patch around a station.
    If two stations are closer than patch_size_m, their patches overlap.
    
    Parameters:
    -----------
    zarr_path : str
        Path to zarr file
    patch_size_m : float
        Patch size in meters (default: 2640m)
    verbose : bool
        Print detailed information
    
    Returns:
    --------
    overlaps : list of tuples
        List of (station1, station2, distance_m) for stations with overlapping patches
    """
    import xarray as xr
    from scipy.spatial import distance_matrix
    
    # Load radar data to get CRS
    ds = xr.open_zarr(zarr_path, consolidated=False)
    crs_str = ds.reflectivity.attrs.get('crs', 'EPSG:32610')
    
    # Get station locations in UTM
    stations = get_weather_station_locations(crs_str, verbose=False)
    
    if len(stations) == 0:
        print("⚠️  No stations found!")
        return []
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STATION OVERLAP CHECK")
        print(f"{'='*60}")
        print(f"Patch size: {patch_size_m}m")
        print(f"Number of stations: {len(stations)}")
        print(f"Overlap threshold: {patch_size_m}m (patches touch or overlap)")
    
    # Calculate pairwise distances
    coords = np.array([[s['x'], s['y']] for s in stations])
    distances = distance_matrix(coords, coords)
    
    # Find overlapping pairs (distance < patch_size_m)
    overlaps = []
    for i in range(len(stations)):
        for j in range(i+1, len(stations)):
            dist = distances[i, j]
            if dist < patch_size_m:
                overlaps.append((stations[i], stations[j], dist))
    
    if verbose:
        print(f"\n{'='*60}")
        if len(overlaps) == 0:
            print(f"✅ NO OVERLAPS FOUND!")
            print(f"   All stations are > {patch_size_m}m apart")
            print(f"   Each station will have a unique {patch_size_m}m patch")
        else:
            print(f"⚠️  FOUND {len(overlaps)} OVERLAPPING PAIRS:")
            for s1, s2, dist in overlaps:
                print(f"\n   {s1['name']} ↔ {s2['name']}")
                print(f"   Distance: {dist:.0f}m (overlap: {patch_size_m - dist:.0f}m)")
                print(f"   Station 1: ({s1['x']:.0f}m, {s1['y']:.0f}m)")
                print(f"   Station 2: ({s2['x']:.0f}m, {s2['y']:.0f}m)")
            
            print(f"\n   ℹ️  Overlapping patches mean these stations share radar data.")
            print(f"   This is OK for training but be aware when interpreting results.")
        
        # Print distance statistics
        print(f"\n{'='*60}")
        print(f"STATION SPACING STATISTICS:")
        # Get all pairwise distances (excluding diagonal)
        all_dists = []
        for i in range(len(stations)):
            for j in range(i+1, len(stations)):
                all_dists.append(distances[i, j])
        
        if len(all_dists) > 0:
            print(f"  Min distance: {np.min(all_dists):.0f}m")
            print(f"  Max distance: {np.max(all_dists):.0f}m")
            print(f"  Mean distance: {np.mean(all_dists):.0f}m")
            print(f"  Median distance: {np.median(all_dists):.0f}m")
        print(f"{'='*60}\n")
    
    return overlaps

def show_dem_with_stations(dem_path='preserve_dem_10m_utm.tif', show_stations=True, figsize=(12, 10)):
    """
    Visualize Digital Elevation Model with weather stations overlaid
    """
    import rasterio
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"Loading DEM from: {dem_path}...")
    
    try:
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            # Handle NaN values
            valid_mask = ~np.isnan(dem_data)
            n_valid = valid_mask.sum()
            n_total = dem_data.size
            
            print(f"DEM shape: {dem_data.shape}")
            print(f"Valid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")
            
            if n_valid > 0:
                elev_min = np.nanmin(dem_data)
                elev_max = np.nanmax(dem_data)
                print(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m")
            else:
                print("⚠️  No valid elevation data!")
                return
            
            # Create masked array (NaNs will be transparent/white)
            dem_masked = np.ma.masked_invalid(dem_data)
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot DEM with NaN handling
            im = ax.imshow(dem_masked, 
                          origin='upper',
                          extent=extent,
                          cmap='terrain',
                          vmin=elev_min,
                          vmax=elev_max,
                          interpolation='nearest')
            
            # Set NaN color to white (background)
            im.cmap.set_bad(color='white', alpha=0.3)
            
            # Load and plot boundary
            boundary_x, boundary_y = load_preserve_boundary(str(crs))
            if boundary_x is not None:
                ax.plot(boundary_x, boundary_y, 'k-', linewidth=2.5, label='Preserve Boundary')
                ax.plot(boundary_x, boundary_y, 'w--', linewidth=1, alpha=0.7)
            
            # Load and plot stations
            if show_stations:
                stations = get_weather_station_locations(str(crs), verbose=False)
                if len(stations) > 0:
                    station_x = [s['x'] for s in stations]
                    station_y = [s['y'] for s in stations]
                    
                    ax.scatter(station_x, station_y, 
                              c='red', marker='^', s=150, 
                              edgecolor='white', linewidth=2, 
                              zorder=10, label='Weather Stations')
                    
                    # # Add station labels
                    # for s in stations:
                    #     ax.annotate(s['name'].replace('Dangermond_', ''), 
                    #                xy=(s['x'], s['y']), 
                    #                xytext=(10, 10),
                    #                textcoords='offset points', 
                    #                fontsize=8,
                    #                bbox=dict(boxstyle='round,pad=0.3', 
                    #                        facecolor='white', 
                    #                        alpha=0.9,
                    #                        edgecolor='black'),
                    #                arrowprops=dict(arrowstyle='->', 
                    #                              connectionstyle='arc3,rad=0',
                    #                              color='black'))
                    
                    print(f"✓ Plotted {len(stations)} weather stations")
            
            # Formatting
            ax.set_xlabel('Easting (m UTM)', fontsize=12)
            ax.set_ylabel('Northing (m UTM)', fontsize=12)
            ax.set_title('Digital Elevation Model - Dangermond Preserve\n'
                        f'Elevation: {elev_min:.0f} - {elev_max:.0f} m', 
                        fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', 
                              fraction=0.046, pad=0.04)
            
            if show_stations or boundary_x is not None:
                ax.legend(loc='upper right', fontsize=10,
                         fancybox=True, shadow=True)
            
            ax.grid(True, alpha=0.3, linestyle='--', color='gray')
            
            # Set aspect ratio to equal for proper scaling
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
def show_nexrad(zarr_path="KVBX_preserve_500m.zarr", time_idx=0, datetime_target=None, 
                altitudes=[500, 1500, 3000, 6000], show_stations=True, altitude_single=1500):
    """
    Visualize radar reflectivity from the zarr file at multiple altitudes
    
    Parameters:
    -----------
    zarr_path : str
        Path to zarr file (default: "KVBX_preserve_500m.zarr")
    time_idx : int
        Time step to visualize (default: 0 = first scan). Ignored if datetime_target is provided.
    datetime_target : str or datetime, optional
        Target datetime to find. Formats:
        - 'YYYY-MM-DD' → shows first scan of that day at multiple altitudes
        - 'YYYY-MM-DD HH' or 'YYYY-MM-DD HH:00' → shows ALL scans in that hour (single altitude)
    altitudes : list
        List of altitudes (in meters) to display for single-scan view (default: [500, 1500, 3000, 6000])
    show_stations : bool
        Whether to overlay weather station locations (default: True)
    altitude_single : int
        Altitude to use when showing multiple scans in an hour (default: 1500m)
    """
    from datetime import datetime as dt
    import pandas as pd
    
    print(f"Opening zarr file: {zarr_path}...")
    
    try:
        store = zarr.open(zarr_path, mode='r')
        
        ref_array = store['reflectivity']
        time_array = store['time']
        z_array = store['z']
        y_array = store['y']
        x_array = store['x']
        
        print(f"Data shape: {ref_array.shape}")
        print(f"  time: {ref_array.shape[0]} steps")
        
        # Get coordinates
        z_coords = z_array[:]
        x_coords = x_array[:]
        y_coords = y_array[:]
        
        # Get CRS
        try:
            attrs = dict(ref_array.attrs)
            crs_str = attrs.get('crs', 'EPSG:32610')
        except:
            crs_str = 'EPSG:32610'
        
        # Load boundary and stations
        boundary_x, boundary_y = load_preserve_boundary(crs_str) if crs_str else (None, None)
        stations = get_weather_station_locations(crs_str, verbose=False) if show_stations and crs_str else []
        
        # ============================================================
        # DATETIME TARGET MODE
        # ============================================================
        if datetime_target is not None:
            times = pd.to_datetime(time_array[:])
            
            # Check if it's an hour specification (length 13: 'YYYY-MM-DD HH' or 16: 'YYYY-MM-DD HH:00')
            is_hour_query = (len(datetime_target) == 13 or 
                            (len(datetime_target) >= 16 and datetime_target.endswith(':00')))
            
            if is_hour_query:
                # ============================================================
                # HOUR MODE: Plot all scans within the specified hour
                # ============================================================
                hour_str = datetime_target[:13] if len(datetime_target) >= 13 else datetime_target
                target_hour = pd.Timestamp(hour_str.replace(' ', 'T') + ':00:00')
                next_hour = target_hour + pd.Timedelta(hours=1)
                
                # Find all scans in this hour
                hour_mask = (times >= target_hour) & (times < next_hour)
                hour_indices = np.where(hour_mask)[0]
                
                # Find all scans in this hour
                hour_mask = (times >= target_hour) & (times < next_hour)
                hour_indices = np.where(hour_mask)[0]

                if len(hour_indices) == 0:
                    print(f"\n⚠️  No scans found for hour {target_hour}")
                    print(f"   Available range: {times.min()} to {times.max()}")
                    return

                # === SORT INDICES BY ACTUAL TIMESTAMP (most recent change) ===
                hour_times = times[hour_indices]
                sorted_order = np.argsort(hour_times)
                hour_indices = hour_indices[sorted_order]
                # ============================================================

                print(f"\n🔍 Found {len(hour_indices)} scans in hour {target_hour.strftime('%Y-%m-%d %H:00')}")
                for idx in hour_indices:
                    print(f"   idx={idx}: {times[idx].strftime('%H:%M:%S')}")
                
                # Find z index for single altitude
                z_idx = np.argmin(np.abs(z_coords - altitude_single))
                actual_alt = z_coords[z_idx]
                
                # Create grid of subplots for all scans in the hour
                n_scans = len(hour_indices)
                ncols = min(4, n_scans)
                nrows = (n_scans + ncols - 1) // ncols
                
                fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
                if n_scans == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_scans > 1 else [axes]
                
                print(f"\nPlotting {n_scans} scans at {actual_alt:.0f}m altitude...")
                
                for i, tidx in enumerate(hour_indices):
                    ax = axes[i]
                    radar_slice = ref_array[tidx, z_idx, :, :]
                    scan_time = times[tidx]
                    
                    im = ax.imshow(radar_slice, 
                                  origin='lower', 
                                  extent=[x_coords.min(), x_coords.max(), 
                                         y_coords.min(), y_coords.max()],
                                  cmap='turbo',
                                  vmin=-10, vmax=60)
                    
                    if boundary_x is not None:
                        ax.plot(boundary_x, boundary_y, 'k-', linewidth=1.5)
                        ax.plot(boundary_x, boundary_y, 'w-', linewidth=0.5, alpha=0.5)
                    
                    if len(stations) > 0:
                        station_x = [s['x'] for s in stations]
                        station_y = [s['y'] for s in stations]
                        ax.scatter(station_x, station_y, c='lime', marker='^', s=60, 
                                  edgecolor='black', linewidth=1)
                    
                    max_dbz = np.nanmax(radar_slice)
                    ax.set_title(f"{scan_time.strftime('%H:%M:%S')}\nMax: {max_dbz:.1f} dBZ", fontsize=10)
                    ax.set_xlabel('Easting (m)')
                    ax.set_ylabel('Northing (m)')
                    plt.colorbar(im, ax=ax, label='dBZ', fraction=0.046, pad=0.04)
                
                for i in range(n_scans, len(axes)):
                    axes[i].axis('off')
                
                fig.suptitle(f'KVBX Radar - {target_hour.strftime("%Y-%m-%d %H:00")} UTC\n'
                            f'Altitude: {actual_alt:.0f}m | {n_scans} scans', 
                            fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.show()
                
                # Hourly composite
                print(f"\n📊 Creating hourly composite (max of {n_scans} scans)...")
                all_scans = np.stack([ref_array[idx, z_idx, :, :] for idx in hour_indices])
                composite = np.nanmax(all_scans, axis=0)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(composite, origin='lower', 
                              extent=[x_coords.min(), x_coords.max(), 
                                     y_coords.min(), y_coords.max()],
                              cmap='turbo', vmin=-10, vmax=60)
                
                if boundary_x is not None:
                    ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
                
                if len(stations) > 0:
                    for s in stations:
                        ax.scatter(s['x'], s['y'], c='lime', marker='^', s=100, 
                                  edgecolor='black', linewidth=1.5, zorder=10)
                        ax.annotate(s['name'].replace('Dangermond_', ''), 
                                   xy=(s['x'], s['y']), xytext=(5, 5), 
                                   textcoords='offset points', fontsize=8, 
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
                
                ax.set_xlabel('Easting (m UTM)')
                ax.set_ylabel('Northing (m UTM)')
                ax.set_title(f'Hourly Max Composite - {target_hour.strftime("%Y-%m-%d %H:00")} UTC\n'
                            f'Max of {n_scans} scans at {actual_alt:.0f}m', fontsize=12, fontweight='bold')
                plt.colorbar(im, label='Max Reflectivity (dBZ)')
                plt.tight_layout()
                plt.show()
                return
            
            else:
                # ============================================================
                # DAY MODE: Find first scan of day, show at multiple altitudes
                # ============================================================
                if len(datetime_target) == 10:  # 'YYYY-MM-DD'
                    target_date = pd.Timestamp(datetime_target)
                    day_mask = (times.date == target_date.date())
                    if day_mask.any():
                        time_idx = np.where(day_mask)[0][0]
                        print(f"\n🔍 Found day {datetime_target}, using first scan at {times[time_idx]}")
                    else:
                        print(f"\n⚠️  No data for {datetime_target}")
                        return
                else:
                    # Exact datetime match
                    target_dt = pd.Timestamp(datetime_target)
                    time_diffs = np.abs(times - target_dt)
                    time_idx = time_diffs.argmin()
                    print(f"\n🔍 Target: {datetime_target}, closest: {times[time_idx]}")
        
        # ============================================================
        # SINGLE SCAN MODE: Plot at multiple altitudes
        # ============================================================
        if time_idx >= ref_array.shape[0]:
            time_idx = ref_array.shape[0] - 1
        
        radar_slice = ref_array[time_idx, :, :, :]
        
        print(f"\nData diagnostics for time step {time_idx}:")
        print(f"  Valid values: {np.sum(~np.isnan(radar_slice)):,} / {radar_slice.size:,}")
        print(f"  Data range: [{np.nanmin(radar_slice):.2f}, {np.nanmax(radar_slice):.2f}] dBZ")
        
        n_altitudes = len(altitudes)
        ncols = 2
        nrows = (n_altitudes + 1) // 2
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
        axes = [axes] if n_altitudes == 1 else axes.flatten()
        
        for i, z_target in enumerate(altitudes):
            z_idx = np.argmin(np.abs(z_coords - z_target))
            horizontal_slice = radar_slice[z_idx, :, :]
            
            ax = axes[i]
            im = ax.imshow(horizontal_slice, origin='lower', 
                          extent=[x_coords.min(), x_coords.max(), 
                                 y_coords.min(), y_coords.max()],
                          cmap='turbo', vmin=-10, vmax=60)
            
            if boundary_x is not None:
                ax.plot(boundary_x, boundary_y, 'k-', linewidth=2)
                ax.plot(boundary_x, boundary_y, 'w-', linewidth=0.5, alpha=0.5)
            
            if len(stations) > 0:
                station_x = [s['x'] for s in stations]
                station_y = [s['y'] for s in stations]
                ax.scatter(station_x, station_y, c='lime', marker='^', s=100, 
                          edgecolor='black', linewidth=1.5, zorder=10)
                if i == 0:
                    for s in stations:
                        ax.annotate(s['name'].replace('Dangermond_', ''), 
                                   xy=(s['x'], s['y']), xytext=(5, 5),
                                   textcoords='offset points', fontsize=7, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax.set_xlabel('Easting (m UTM)')
            ax.set_ylabel('Northing (m UTM)')
            ax.set_title(f'Altitude: {z_coords[z_idx]:.0f} m')
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)', fraction=0.046, pad=0.04)
        
        for i in range(n_altitudes, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'KVBX Radar Reflectivity - Time step {time_idx}', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()