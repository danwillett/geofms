"""
map_jalama_hydrology.py — Presentation map showing hydrologic features and sensors
in the Jalama Creek watershed at Dangermond Preserve.

Layers:
  1. Preserve boundary (from local GeoJSON)
  2. Jalama watershed polygon (from ArcGIS Feature Service, filtered to Location='Jalama')
  3. Jalama Creek flowlines (from ArcGIS Feature Service)
  4. Rain gauge stations (from PostGIS database — DendraDatastream.name == 'Rainfall')
  5. Groundwater well sensors (from PostGIS — DendraDatastream.name == 'Depth to Groundwater')
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import shape
import contextily as cx
import requests
import json
from pathlib import Path

from database.config import connect, create_session
from database.models.DendraStations import DendraStation, DendraDatastream


# ── 1. LOAD PRESERVE BOUNDARY ────────────────────────────────────────────────

def load_preserve_boundary_gdf(geojson_path='geometries/dangermond-preserve-boundary.geojson'):
    """Load preserve boundary as a GeoDataFrame in WGS84."""
    gdf = gpd.read_file(geojson_path)
    return gdf.to_crs(epsg=4326)


# ── 2. QUERY ARCGIS FEATURE SERVICES ─────────────────────────────────────────

WATERSHED_URL = (
    "https://services.arcgis.com/F7DSX1DSNSiWmOqh/ArcGIS/rest/services/"
    "jldp_major_watersheds_(Public)/FeatureServer/0/query"
)
FLOWLINES_URL = (
    "https://services.arcgis.com/F7DSX1DSNSiWmOqh/ArcGIS/rest/services/"
    "Jalama%20Watershed%20Flowlines/FeatureServer/0/query"
)


def query_arcgis_features(url, where="1=1", out_sr=4326):
    """Query an ArcGIS REST Feature Service and return a GeoDataFrame."""
    params = {
        'where': where,
        'outFields': '*',
        'outSR': out_sr,
        'f': 'geojson',
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return gpd.GeoDataFrame.from_features(resp.json()['features'], crs=f'EPSG:{out_sr}')


def load_jalama_watershed():
    """Load Jalama watershed polygon (filtered by Location='Jalama')."""
    return query_arcgis_features(WATERSHED_URL, where="Location='Jalama'")


def load_jalama_flowlines():
    """Load all Jalama Creek flowlines."""
    return query_arcgis_features(FLOWLINES_URL)


# ── 3. QUERY DATABASE FOR SENSOR LOCATIONS ────────────────────────────────────

def get_stations_by_datastream(datastream_name):
    """
    Query stations that have a specific datastream type.
    Returns GeoDataFrame with station name, lat, lon.
    """
    engine = connect()
    session = create_session(engine)

    results = (
        session.query(
            DendraStation.name,
            DendraStation.latitude,
            DendraStation.longitude,
        )
        .join(DendraDatastream, DendraStation.id == DendraDatastream.station_id)
        .filter(DendraDatastream.name == datastream_name)
        .distinct()
        .all()
    )
    session.close()

    if not results:
        print(f"⚠️  No stations found with datastream '{datastream_name}'")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame(
        [{'name': name, 'latitude': lat, 'longitude': lon} for name, lat, lon in results],
        geometry=gpd.points_from_xy(
            [r[2] for r in results],  # longitude
            [r[1] for r in results],  # latitude
        ),
        crs='EPSG:4326',
    )
    print(f"✓ Found {len(gdf)} stations with '{datastream_name}'")
    return gdf


# ── 4. BUILD THE MAP ──────────────────────────────────────────────────────────

def create_jalama_hydrology_map(output_path='figures/jalama_hydrology_map.png'):
    """Create presentation-quality map of Jalama Creek hydrology."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load all layers
    print("Loading layers...")
    preserve = load_preserve_boundary_gdf()
    watershed = load_jalama_watershed()
    flowlines = load_jalama_flowlines()
    rain_gauges = get_stations_by_datastream('Rainfall')
    gw_wells = get_stations_by_datastream('Depth to Groundwater')

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot layers (back to front)
    preserve.plot(ax=ax, facecolor='#e8e8e8', edgecolor='black',
                  linewidth=2, alpha=0.3, label='Preserve Boundary')

    watershed.plot(ax=ax, facecolor='#a8d5e2', edgecolor='#2b6ca3',
                   linewidth=1.5, alpha=0.4, label='Jalama Watershed')

    flowlines.plot(ax=ax, color='#1a75c4', linewidth=1.5,
                   alpha=0.8, label='Stream Network')

    # Rain gauges
    if not rain_gauges.empty:
        rain_gauges.plot(ax=ax, color='red', marker='^', markersize=120,
                         edgecolor='white', linewidth=1.5, zorder=10,
                         label=f'Rain Gauges (n={len(rain_gauges)})')

    # Groundwater wells
    if not gw_wells.empty:
        gw_wells.plot(ax=ax, color='#2ca02c', marker='s', markersize=100,
                      edgecolor='white', linewidth=1.5, zorder=10,
                      label=f'Groundwater Wells (n={len(gw_wells)})')

    # Zoom to full preserve extent (with generous padding)
    bounds = preserve.total_bounds  # [minx, miny, maxx, maxy]
    pad = 0.02  # degrees
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    # Add basemap
    cx.add_basemap(ax, crs='EPSG:4326', source=cx.providers.CartoDB.Positron, zoom=13)

    # Formatting
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.set_title('Jalama Creek Watershed — Hydrologic Sensors & Features\n'
                 'Jack and Laura Dangermond Preserve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved map to: {output_path}")
    plt.show()


if __name__ == '__main__':
    create_jalama_hydrology_map()