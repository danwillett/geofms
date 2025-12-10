def show_dem_with_stations(dem_path='dem_preserve.tif', show_stations=True, figsize=(12, 10)):
    """
    Visualize Digital Elevation Model with weather stations overlaid
    
    Parameters:
    -----------
    dem_path : str
        Path to DEM file (GeoTIFF or numpy array)
    show_stations : bool
        Whether to overlay weather station locations
    figsize : tuple
        Figure size
    """
    import rasterio
    from rasterio.plot import show
    import matplotlib.pyplot as plt
    
    print(f"Loading DEM from: {dem_path}...")
    
    try:
        # Try loading as GeoTIFF with rasterio
        with rasterio.open(dem_path) as src:
            dem_data = src.read(1)  # Read first band
            transform = src.transform
            crs = src.crs
            
            # Get extent in projected coordinates
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            print(f"DEM shape: {dem_data.shape}")
            print(f"Elevation range: {dem_data.min():.1f} - {dem_data.max():.1f} m")
            print(f"CRS: {crs}")
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot DEM
            im = ax.imshow(dem_data, 
                          origin='upper',
                          extent=extent,
                          cmap='terrain',
                          vmin=dem_data.min(),
                          vmax=dem_data.max())
            
            # Load and plot boundary
            boundary_x, boundary_y = load_preserve_boundary(str(crs))
            if boundary_x is not None:
                ax.plot(boundary_x, boundary_y, 'k-', linewidth=2.5, label='Preserve Boundary')
                ax.plot(boundary_x, boundary_y, 'w-', linewidth=1, alpha=0.7)
            
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
                    
                    # Add station labels
                    for s in stations:
                        ax.annotate(s['name'].replace('Dangermond_', ''), 
                                   xy=(s['x'], s['y']), 
                                   xytext=(10, 10),
                                   textcoords='offset points', 
                                   fontsize=8,
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                           facecolor='white', 
                                           alpha=0.8,
                                           edgecolor='black'),
                                   arrowprops=dict(arrowstyle='->', 
                                                 connectionstyle='arc3,rad=0'))
                    
                    print(f"✓ Plotted {len(stations)} weather stations")
            
            # Formatting
            ax.set_xlabel('Easting (m UTM)', fontsize=12)
            ax.set_ylabel('Northing (m UTM)', fontsize=12)
            ax.set_title('Digital Elevation Model - Dangermond Preserve', 
                        fontsize=14, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Elevation (m)', 
                              fraction=0.046, pad=0.04)
            
            if show_stations or boundary_x is not None:
                ax.legend(loc='upper right', fontsize=10)
            
            ax.grid(True, alpha=0.3, linestyle='--', color='white')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        print(f"Error loading DEM: {e}")
        print("\nTrying alternative loading method (numpy)...")
        
        # Try loading as numpy array
        try:
            import numpy as np
            dem_data = np.load(dem_path)
            
            print(f"DEM shape: {dem_data.shape}")
            print(f"Elevation range: {dem_data.min():.1f} - {dem_data.max():.1f} m")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            im = ax.imshow(dem_data, cmap='terrain', origin='lower')
            ax.set_title('Digital Elevation Model')
            plt.colorbar(im, label='Elevation (m)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e2:
            print(f"Could not load DEM: {e2}")
            print(f"\nPlease provide DEM as GeoTIFF (.tif) or numpy array (.npy)")
