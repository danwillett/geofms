import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import zarr
from pyproj import Transformer
import json


def load_preserve_boundary(geojson_path='geometries/dangermond-preserve-boundary.geojson'):
    """
    Load preserve boundary from GeoJSON and transform to radar coordinates
    
    Returns:
    --------
    boundary_x, boundary_y : arrays
        X and Y coordinates of boundary in radar coordinate system (meters)
    """
    try:
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        # Extract coordinates from first feature
        coords = geojson['features'][0]['geometry']['coordinates'][0]
        
        # GeoJSON coordinates are [lon, lat]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # Transform to radar coordinates (same as pull_nexrad.py)
        RADAR_LAT = 34.83855 
        RADAR_LON = -120.397917
        
        transformer = Transformer.from_crs(
            "EPSG:4326",
            f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",
            always_xy=True
        )
        
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


def get_weather_station_locations(verbose=True):
    """
    Query database for weather station locations and transform to radar coordinates
    
    Returns:
    --------
    stations : list of dict
        Each dict contains: {'name': str, 'x': float, 'y': float, 'lat': float, 'lon': float}
        x, y are in radar local coordinates (meters from radar)
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
        
        # Radar location (from pull_nexrad.py)
        RADAR_LAT = 34.83855 
        RADAR_LON = -120.397917
        
        if verbose:
            print(f"   Radar location: {RADAR_LAT:.5f}°N, {RADAR_LON:.5f}°W")
        
        # Set up coordinate transformer (same as pull_nexrad.py)
        transformer = Transformer.from_crs(
            "EPSG:4326",  # WGS84 lat/lon
            f"+proj=aeqd +lat_0={RADAR_LAT} +lon_0={RADAR_LON} +units=m +datum=WGS84",  # Radar-centered
            always_xy=True
        )
        
        # Transform station coordinates
        stations = []
        for station_id, name, lat, lon in stations_with_rain:
            if lat is not None and lon is not None:
                # Transform to radar local coordinates
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
            print(f"\n✅ Transformed {len(stations)} stations to radar coordinates")
        
        return stations
        
    except ImportError:
        print("⚠️  Database module not available - cannot load station locations")
        return []
    except Exception as e:
        print(f"⚠️  Error loading station locations: {e}")
        return []


def show_nexrad(time_idx=0, altitudes=[500, 1500, 3000, 6000], show_stations=True):
    """
    Visualize radar reflectivity from the zarr file at multiple altitudes
    
    Parameters:
    -----------
    time_idx : int
        Time step to visualize (default: 0 = first scan)
    altitudes : list
        List of altitudes (in meters) to display (default: [500, 1500, 3000, 6000])
    show_stations : bool
        Whether to overlay weather station locations (default: True)
    """
    
    print("Opening zarr file...")
    
    # Open the zarr - handle both zarr v2 and v3 formats
    try:
        # Try opening with zarr directly first, then wrap in xarray
        store = zarr.open("KVBX_preserve_500m.zarr", mode='r')
        
        # Read the reflectivity array
        ref_array = store['reflectivity']
        time_array = store['time']
        z_array = store['z']
        y_array = store['y']
        x_array = store['x']
        
        print(f"Data shape: {ref_array.shape}")
        print(f"  time: {ref_array.shape[0]} steps")
        print(f"  z: {ref_array.shape[1]} levels")
        print(f"  y: {ref_array.shape[2]} points")
        print(f"  x: {ref_array.shape[3]} points")
        
        # Get the specified time step
        radar_slice = ref_array[time_idx, :, :, :]  # shape: (z, y, x)
        
        # Get coordinate arrays
        z_coords = z_array[:]
        x_coords = x_array[:]
        y_coords = y_array[:]
        
        # Calculate grid layout for subplots
        n_altitudes = len(altitudes)
        ncols = 2
        nrows = (n_altitudes + 1) // 2
        
        # Create figure with subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5*nrows))
        if n_altitudes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        print(f"\nPlotting time step {time_idx}:")
        
        # Load preserve boundary
        boundary_x, boundary_y = load_preserve_boundary()
        
        # Load weather station locations if requested
        stations = []
        if show_stations:
            stations = get_weather_station_locations(verbose=True)
            if len(stations) > 0:
                # Check if stations are within grid bounds
                stations_in_bounds = []
                for s in stations:
                    if (x_coords.min() <= s['x'] <= x_coords.max() and 
                        y_coords.min() <= s['y'] <= y_coords.max()):
                        stations_in_bounds.append(s)
                    else:
                        print(f"   ⚠️  {s['name']} is outside grid bounds!")
                        print(f"      Station: ({s['x']:.0f}, {s['y']:.0f})")
                        print(f"      Grid X: [{x_coords.min():.0f}, {x_coords.max():.0f}]")
                        print(f"      Grid Y: [{y_coords.min():.0f}, {y_coords.max():.0f}]")
                
                if len(stations_in_bounds) > 0:
                    print(f"\n✅ {len(stations_in_bounds)} stations are within grid bounds")
                    stations = stations_in_bounds
                else:
                    print(f"\n⚠️  All {len(stations)} stations are outside the grid!")
                    print(f"   Grid covers: X=[{x_coords.min():.0f}, {x_coords.max():.0f}], Y=[{y_coords.min():.0f}, {y_coords.max():.0f}]")
        
        # Plot each altitude
        for i, z_target in enumerate(altitudes):
            # Find closest z level
            z_idx = np.argmin(np.abs(z_coords - z_target))
            actual_altitude = z_coords[z_idx]
            
            horizontal_slice = radar_slice[z_idx, :, :]  # shape: (y, x)
            
            # Calculate statistics
            valid_data = horizontal_slice[~np.isnan(horizontal_slice)]
            n_valid = len(valid_data)
            n_total = horizontal_slice.size
            
            print(f"  {actual_altitude:.0f} m: {n_valid}/{n_total} valid points, "
                  f"range [{np.nanmin(horizontal_slice):.1f}, {np.nanmax(horizontal_slice):.1f}] dBZ")
            
            # Plot
            ax = axes[i]
            im = ax.imshow(horizontal_slice, 
                          origin='lower', 
                          extent=[x_coords.min(), x_coords.max(), 
                                 y_coords.min(), y_coords.max()],
                          cmap='turbo',
                          vmin=-10, vmax=60)
            
            # Plot preserve boundary
            if boundary_x is not None and boundary_y is not None:
                ax.plot(boundary_x, boundary_y, 'k-', linewidth=2, label='Preserve', zorder=5)
                ax.plot(boundary_x, boundary_y, 'w-', linewidth=0.5, alpha=0.5, zorder=6)
            
            # Plot weather stations
            if len(stations) > 0:
                station_x = [s['x'] for s in stations]
                station_y = [s['y'] for s in stations]
                ax.scatter(station_x, station_y, 
                          c='lime', marker='^', s=100, 
                          edgecolor='black', linewidth=1.5,
                          label='Rain Gauges', zorder=10)
                
                # Optionally label stations (only on first subplot to avoid clutter)
                if i == 0:
                    for s in stations:
                        ax.annotate(s['name'], 
                                   xy=(s['x'], s['y']),
                                   xytext=(5, 5), 
                                   textcoords='offset points',
                                   fontsize=7, 
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', 
                                           edgecolor='black',
                                           alpha=0.7),
                                   zorder=11)
            
            ax.set_xlabel('X (m from radar)')
            ax.set_ylabel('Y (m from radar)')
            ax.set_title(f'Altitude: {actual_altitude:.0f} m')
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Add legend on first subplot (if there are stations or boundary to show)
            if i == 0 and (len(stations) > 0 or boundary_x is not None):
                ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
            
            # Add colorbar to each subplot
            plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)', fraction=0.046, pad=0.04)
        
        # Hide extra subplots if odd number of altitudes
        for i in range(n_altitudes, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'KVBX Radar Reflectivity - Time step {time_idx}', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error opening zarr: {e}")
        print("\nTrying alternative method with xarray...")
        
        try:
            # Fallback to xarray
            da = xr.open_dataarray("KVBX_preserve_500m.zarr", engine='zarr')
            
            print(f"Data shape: {da.shape}")
            
            # Rest of the plotting code
            radar_slice = da.isel(time=0)
            z_target = 1500
            z_idx = np.argmin(np.abs(da.z.values - z_target))
            horizontal_slice = radar_slice.isel(z=z_idx)

            plt.figure(figsize=(10, 8))
            im = horizontal_slice.plot(cmap='turbo', vmin=-10, vmax=60)
            plt.title(f'Radar Reflectivity at {da.z[z_idx].values:.0f} m')
            plt.show()
            
        except Exception as e2:
            print(f"Both methods failed!")
            print(f"  First error: {e}")
            print(f"  Second error: {e2}")
            raise