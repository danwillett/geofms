from database.config import connect, create_session
from database.models import DendraStation, DendraDatastream, DendraDatapoint

from sqlalchemy import func, cast, Date
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

def get_rainfall_days():

    engine = connect()
    session = create_session(engine)

    # Find rainfall datastream(s)
    rain_ds = session.query(DendraDatastream).filter(DendraDatastream.name == "Rainfall").all()
    rain_ds_ids = [ds.id for ds in rain_ds]
    print(f"Found {len(rain_ds_ids)} rainfall datastream(s)")

    # Aggregate daily rainfall for each station
    daily_rain = (
        session.query(
            DendraDatapoint.datastream_id,
            cast(DendraDatapoint.timestamp_utc, Date).label("day"),
            func.sum(DendraDatapoint.value).label("rain_mm")
        )
        .filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))
        .group_by(DendraDatapoint.datastream_id, cast(DendraDatapoint.timestamp_utc, Date))
        .order_by(DendraDatapoint.datastream_id, "day")
    ).all()

    # Convert to dictionary for convenience: {station_id: {day: rain_mm}}
    from collections import defaultdict
    station_rain = defaultdict(dict)
    for ds_id, day, rain_mm in daily_rain:
        station_rain[ds_id][day] = rain_mm

    return station_rain

def label_rain_events(station_rain, rain_threshold=0.1):
    """
    Returns deduplicated rain_days and no_rain_days
    missing days are treated as 0 mm
    """
    rain_days_set = set()
    no_rain_days_set = set()

    for station_id, day_dict in station_rain.items():
        # full date range
        min_day = min(day_dict.keys())
        max_day = max(day_dict.keys())
        total_days = [min_day + timedelta(days=i) for i in range((max_day - min_day).days + 1)]

        for day in total_days:
            rain_mm = day_dict.get(day, 0.0)  # missing days treated as 0
            if rain_mm >= rain_threshold:
                rain_days_set.add(day)
            else:
                no_rain_days_set.add(day)

    # No-rain days are the remaining days

    return sorted(rain_days_set), sorted(no_rain_days_set)

def get_hourly_rainfall_intensity(start_date=None, end_date=None):
    """
    Get hourly rainfall rates for each station and rank days by intensity
    
    Parameters:
    -----------
    start_date : datetime.date or str, optional
        Start date for analysis (default: earliest data)
        Can be date object or string in 'YYYY-MM-DD' format
    end_date : datetime.date or str, optional
        End date for analysis (default: latest data)
        Can be date object or string in 'YYYY-MM-DD' format
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns: timestamp_utc, datastream_id, hourly_rain_mm
    """
    
    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    engine = connect()
    session = create_session(engine)
    
    # Find rainfall datastream(s)
    rain_ds = session.query(DendraDatastream).filter(DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])).all()
    rain_ds_ids = [ds.id for ds in rain_ds]
    print(f"Found {len(rain_ds_ids)} rainfall datastream(s)")
    
    # Query all rainfall datapoints
    query = session.query(
        DendraDatapoint.timestamp_utc,
        DendraDatapoint.datastream_id,
        DendraDatapoint.value
    ).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))
    
    if start_date:
        # Convert date to datetime for comparison with timestamp
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DendraDatapoint.timestamp_utc >= start_datetime)
    if end_date:
        # Convert date to datetime and add one day to include the entire end_date
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(DendraDatapoint.timestamp_utc < end_datetime)
    
    query = query.order_by(DendraDatapoint.timestamp_utc)
    
    print("Querying rainfall data...")
    results = query.all()
    print(f"Found {len(results)} rainfall measurements")
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['timestamp_utc', 'datastream_id', 'value'])
    
    if len(df) == 0:
        print("No rainfall data found!")
        return df
    
    # Aggregate to hourly for each station
    df['hour'] = df['timestamp_utc'].dt.floor('H')
    
    hourly = df.groupby(['datastream_id', 'hour'])['value'].sum().reset_index()
    hourly.columns = ['datastream_id', 'timestamp_utc', 'hourly_rain_mm']
    
    return hourly


def rank_days_by_rainfall_intensity(top_n=None, min_rainfall=0.5, start_date=None, end_date=None, 
                                     metric='max_hourly', max_valid_rainfall=None):
    """
    Rank days by rainfall intensity and return the top N rainiest days
    
    Parameters:
    -----------
    top_n : int
        Number of top rainy days to return (default: 100)
    start_date : datetime.date, optional
        Start date for analysis
    end_date : datetime.date, optional
        End date for analysis
    metric : str
        Ranking metric:
        - 'max_hourly': Maximum hourly rainfall rate in the day (default)
        - 'mean_hourly': Mean hourly rainfall rate (when rain > 0)
        - 'daily_total': Total daily rainfall
        - 'peak_intensity': Average of station peak hourly rates
        - 'rain_hours': Number of station-hours with rain above threshold
    max_valid_rainfall : float, optional
        Filter out readings above this value (sensor errors)
    
    Returns:
    --------
    top_days : list of datetime.date
        List of top N rainiest days, sorted by intensity (heaviest first)
    day_stats : pandas.DataFrame
        DataFrame with statistics for each day
    """
    
    print(f"\n{'='*60}")
    print(f"Ranking days by rainfall intensity (metric: {metric})")
    print(f"{'='*60}\n")
    
    # Get hourly rainfall data
    hourly = get_hourly_rainfall_intensity(start_date, end_date)
    
    if len(hourly) == 0:
        return [], pd.DataFrame()
    
    # Filter out extreme values (sensor errors) if specified
    if max_valid_rainfall is not None:
        before_count = len(hourly)
        hourly = hourly[hourly['hourly_rain_mm'] <= max_valid_rainfall]
        after_count = len(hourly)
        if before_count > after_count:
            print(f"⚠️  Filtered out {before_count - after_count} extreme readings (>{max_valid_rainfall} mm/hr)")
    
    # Add date column
    hourly['date'] = hourly['timestamp_utc'].dt.date
    
    # Calculate daily statistics based on metric
    if metric == 'max_hourly':
        # For each day, take max hourly rate across all stations
        daily_stats = hourly.groupby('date').agg({
            'hourly_rain_mm': ['max', 'mean', 'sum', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'max_hourly', 'mean_hourly', 'total_rain', 'n_measurements']
        daily_stats = daily_stats.sort_values('max_hourly', ascending=False)
        
    elif metric == 'mean_hourly':
        # Mean of non-zero hourly rates
        hourly_nonzero = hourly[hourly['hourly_rain_mm'] > 0]
        daily_stats = hourly_nonzero.groupby('date').agg({
            'hourly_rain_mm': ['mean', 'max', 'sum', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'mean_hourly', 'max_hourly', 'total_rain', 'n_measurements']
        daily_stats = daily_stats.sort_values('mean_hourly', ascending=False)
        
    elif metric == 'daily_total':
        # Total daily rainfall
        daily_stats = hourly.groupby('date').agg({
            'hourly_rain_mm': ['sum', 'max', 'mean', 'count']
        }).reset_index()
        daily_stats.columns = ['date', 'total_rain', 'max_hourly', 'mean_hourly', 'n_measurements']
        daily_stats = daily_stats.sort_values('total_rain', ascending=False)
        
    elif metric == 'peak_intensity':
        # Average of each station's peak hourly rate for the day
        station_peaks = hourly.groupby(['date', 'datastream_id'])['hourly_rain_mm'].max().reset_index()
        daily_stats = station_peaks.groupby('date').agg({
            'hourly_rain_mm': ['mean', 'max', 'sum'],
            'datastream_id': 'count'
        }).reset_index()
        daily_stats.columns = ['date', 'avg_station_peak', 'max_station_peak', 'sum_peaks', 'n_stations']
        daily_stats = daily_stats.sort_values('avg_station_peak', ascending=False)
    
    elif metric == 'rain_hours':
        # Count station-hours with rain above threshold (widespread, consistent rain)
        # This finds days with lots of rain coverage, not just spikes
        rain_threshold = min_rainfall  # Use min_rainfall as the threshold
        hourly_with_rain = hourly[hourly['hourly_rain_mm'] >= rain_threshold]
        
        # Count rain-hours per day
        daily_rain_hours = hourly_with_rain.groupby('date').agg({
            'hourly_rain_mm': ['count', 'sum', 'mean', 'max'],
            'datastream_id': 'nunique'
        }).reset_index()
        daily_rain_hours.columns = ['date', 'rain_hours', 'total_rain', 'mean_rain_rate', 'max_rain_rate', 'n_stations']
        daily_rain_hours = daily_rain_hours.sort_values('rain_hours', ascending=False)
        daily_stats = daily_rain_hours
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get top N days
    if top_n is not None:
        top_days = daily_stats.head(top_n)['date'].tolist()
    else:
        top_days = daily_stats[daily_stats['total_rain'] >= min_rainfall]['date'].tolist()
    
    # Print summary
    print(f"Analyzed {len(daily_stats)} days with rainfall data")
    print(f"\nTop {min(top_n, len(top_days))} rainiest days:")
    print("-" * 60)
    
    for i, row in daily_stats.head(10).iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        if metric == 'max_hourly':
            print(f"  {date_str}: {row['max_hourly']:.1f} mm/hr (max), {row['total_rain']:.1f} mm total")
        elif metric == 'mean_hourly':
            print(f"  {date_str}: {row['mean_hourly']:.1f} mm/hr (mean), {row['total_rain']:.1f} mm total")
        elif metric == 'daily_total':
            print(f"  {date_str}: {row['total_rain']:.1f} mm total, {row['max_hourly']:.1f} mm/hr peak")
        elif metric == 'peak_intensity':
            print(f"  {date_str}: {row['avg_station_peak']:.1f} mm/hr (avg peak), {row['n_stations']:.0f} stations")
        elif metric == 'rain_hours':
            print(f"  {date_str}: {row['rain_hours']:.0f} rain-hours, {row['total_rain']:.1f} mm total, {row['max_rain_rate']:.1f} mm/hr peak")
    
    if len(top_days) > 10:
        print(f"  ... and {len(top_days) - 10} more days")
    
    print(f"\n{'='*60}\n")
    
    return top_days, daily_stats

def get_10min_gauge_data(start_date=None, end_date=None, stations=None):
    """
    Get 10-minute interval rain gauge data for alignment with NEXRAD
    
    Parameters:
    -----------
    start_date : datetime.date or str
        Start date for data extraction
    end_date : datetime.date or str
        End date for data extraction
    stations : list of int, optional
        List of station IDs to include (default: all rainfall stations)
    
    Returns:
    --------
    df : pandas.DataFrame
        Columns: timestamp_utc, station_id, station_name, lat, lon, rainfall_mm
        Index: timestamp_utc
    """
    from datetime import datetime, timedelta
    
    # Convert string dates if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    engine = connect()
    session = create_session(engine)
    
    # Find rainfall datastreams
    rain_ds_query = session.query(DendraDatastream).filter(
        DendraDatastream.name == "Rainfall"
    )
    
    if stations:
        rain_ds_query = rain_ds_query.filter(DendraDatastream.station_id.in_(stations))
    
    rain_ds = rain_ds_query.all()
    rain_ds_ids = [ds.id for ds in rain_ds]
    
    print(f"Found {len(rain_ds_ids)} rainfall datastream(s)")
    
    # Get station metadata for lat/lon
    station_info = {}
    for ds in rain_ds:
        station = session.query(DendraStation).filter(
            DendraStation.id == ds.station_id
        ).first()
        if station:
            station_info[ds.id] = {
                'station_id': station.id,
                'name': station.name,
                'lat': station.latitude,
                'lon': station.longitude
            }
    
    # Query rainfall datapoints
    query = session.query(
        DendraDatapoint.timestamp_utc,
        DendraDatapoint.datastream_id,
        DendraDatapoint.value
    ).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))
    
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DendraDatapoint.timestamp_utc >= start_datetime)
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(DendraDatapoint.timestamp_utc < end_datetime)
    
    query = query.order_by(DendraDatapoint.timestamp_utc)
    
    print(f"Querying 10-minute gauge data from {start_date} to {end_date}...")
    results = query.all()
    print(f"Found {len(results)} rainfall measurements")
    
    if len(results) == 0:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['timestamp_utc', 'datastream_id', 'rainfall_mm'])
    
    # Add station metadata
    df['station_id'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('station_id'))
    df['station_name'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('name'))
    df['lat'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lat'))
    df['lon'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lon'))
    
    # Round timestamps to nearest 10 minutes for consistency
    df['timestamp_10min'] = df['timestamp_utc'].dt.round('10min')
    
    # Group by station and 10-min interval (in case of duplicates)
    df_grouped = df.groupby(['timestamp_10min', 'station_id']).agg({
        'rainfall_mm': 'sum',
        'station_name': 'first',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    df_grouped.rename(columns={'timestamp_10min': 'timestamp_utc'}, inplace=True)
    
    print(f"Aggregated to {len(df_grouped)} 10-minute measurements across {df_grouped['station_id'].nunique()} stations")
    
    return df_grouped


def get_hourly_precipitation_by_station(start_date=None, end_date=None, min_rainfall_mm=0.1):
    """
    Get hourly accumulated precipitation for each station (for ML training)
    
    Parameters:
    -----------
    start_date, end_date : datetime.date or str
        Date range
    min_rainfall_mm : float
        Minimum hourly rainfall to include (default: 0.1mm)
        Use to filter out very light/trace amounts
    
    Returns:
    --------
    samples : list of dict
        Each dict contains:
        {
            'hour_start': datetime,
            'station_id': int,
            'station_name': str,
            'lat': float,
            'lon': float,
            'hourly_precip_mm': float
        }
    """
    from datetime import datetime, timedelta
    
    # Convert string dates if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    engine = connect()
    session = create_session(engine)
    
    # Find rainfall datastreams
    rain_ds = session.query(DendraDatastream).filter(DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])).all()
    rain_ds_ids = [ds.id for ds in rain_ds]
    
    print(f"Found {len(rain_ds_ids)} rainfall datastream(s)")
    
    # Get station metadata
    station_info = {}
    for ds in rain_ds:
        station = session.query(DendraStation).filter(
            DendraStation.id == ds.station_id
        ).first()
        if station:
            station_info[ds.id] = {
                'station_id': station.id,
                'name': station.name,
                'lat': station.latitude,
                'lon': station.longitude
            }
    
    # Query rainfall datapoints
    query = session.query(
        DendraDatapoint.timestamp_utc,
        DendraDatapoint.datastream_id,
        DendraDatapoint.value
    ).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))
    
    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DendraDatapoint.timestamp_utc >= start_datetime)
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(DendraDatapoint.timestamp_utc < end_datetime)
    
    query = query.order_by(DendraDatapoint.timestamp_utc)
    
    print(f"Querying precipitation data from {start_date} to {end_date}...")
    results = query.all()
    print(f"Found {len(results)} measurements")
    
    if len(results) == 0:
        return []
    
    # Convert to DataFrame
    df = pd.DataFrame(results, columns=['timestamp_utc', 'datastream_id', 'rainfall_mm'])
    
    # Add station metadata
    df['station_id'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('station_id'))
    df['station_name'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('name'))
    df['lat'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lat'))
    df['lon'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lon'))
    
    # Group by hour and station
    df['hour'] = df['timestamp_utc'].dt.floor('H')
    
    hourly = df.groupby(['hour', 'station_id']).agg({
        'rainfall_mm': ['sum', 'max', 'count', lambda x: (x > 0).sum()],
        'station_name': 'first',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    
    # Flatten multi-level columns
    hourly.columns = ['hour', 'station_id', 'rainfall_mm', 'max_bin_mm', 'n_bins',
                      'n_active_bins', 'station_name', 'lat', 'lon']
    
    # Compute dump ratio: fraction of hourly total in the single largest 10-min bin
    hourly['dump_ratio'] = hourly['max_bin_mm'] / hourly['rainfall_mm'].clip(lower=1e-6)
    
    # Filter by minimum rainfall
    hourly = hourly[hourly['rainfall_mm'] >= min_rainfall_mm]
    
    # Convert to list of dicts
    samples = []
    for _, row in hourly.iterrows():
        samples.append({
            'hour_start': row['hour'],
            'station_id': row['station_id'],
            'station_name': row['station_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'hourly_precip_mm': row['rainfall_mm'],
            'max_bin_mm': row['max_bin_mm'],
            'n_active_bins': int(row['n_active_bins']),
            'dump_ratio': row['dump_ratio'],
        })
    
    print(f"\nFound {len(samples)} hourly samples with rainfall >= {min_rainfall_mm}mm")
    print(f"  Covering {len(set(s['station_id'] for s in samples))} stations")
    if len(samples) > 0:
        print(f"  Time range: {min(s['hour_start'] for s in samples)} to {max(s['hour_start'] for s in samples)}")
        print(f"  Precipitation range: {min(s['hourly_precip_mm'] for s in samples):.2f} - {max(s['hourly_precip_mm'] for s in samples):.2f} mm/hr")
    
    return samples


def get_offset_hourly_precipitation_by_station(start_date=None, end_date=None,
                                                min_rainfall_mm=0.1, offset_minutes=30):
    """
    Get hourly accumulated precipitation using offset windows (e.g. 12:30-1:30).

    Same as get_hourly_precipitation_by_station but with a time offset applied
    to the aggregation windows. This creates additional training samples that
    are partially independent from standard hourly windows.

    Parameters
    ----------
    offset_minutes : int
        Offset in minutes from the top of the hour (default: 30)
    """
    from datetime import datetime, timedelta

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

    engine = connect()
    session = create_session(engine)

    rain_ds = session.query(DendraDatastream).filter(DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])).all()
    rain_ds_ids = [ds.id for ds in rain_ds]

    station_info = {}
    for ds in rain_ds:
        station = session.query(DendraStation).filter(
            DendraStation.id == ds.station_id
        ).first()
        if station:
            station_info[ds.id] = {
                'station_id': station.id,
                'name': station.name,
                'lat': station.latitude,
                'lon': station.longitude
            }

    query = session.query(
        DendraDatapoint.timestamp_utc,
        DendraDatapoint.datastream_id,
        DendraDatapoint.value
    ).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))

    if start_date:
        start_datetime = datetime.combine(start_date, datetime.min.time())
        query = query.filter(DendraDatapoint.timestamp_utc >= start_datetime)
    if end_date:
        end_datetime = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)
        query = query.filter(DendraDatapoint.timestamp_utc < end_datetime)

    query = query.order_by(DendraDatapoint.timestamp_utc)

    print(f"Querying precipitation data for {offset_minutes}-min offset windows...")
    results = query.all()
    print(f"Found {len(results)} measurements")

    if len(results) == 0:
        return []

    df = pd.DataFrame(results, columns=['timestamp_utc', 'datastream_id', 'rainfall_mm'])

    df['station_id'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('station_id'))
    df['station_name'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('name'))
    df['lat'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lat'))
    df['lon'] = df['datastream_id'].map(lambda x: station_info.get(x, {}).get('lon'))

    # Shift timestamps back by offset, then floor to hour, then shift forward
    offset = pd.Timedelta(minutes=offset_minutes)
    df['offset_hour'] = (df['timestamp_utc'] - offset).dt.floor('H') + offset

    hourly = df.groupby(['offset_hour', 'station_id']).agg({
        'rainfall_mm': ['sum', 'max', 'count', lambda x: (x > 0).sum()],
        'station_name': 'first',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()

    # Flatten multi-level columns
    hourly.columns = ['offset_hour', 'station_id', 'rainfall_mm', 'max_bin_mm', 'n_bins',
                      'n_active_bins', 'station_name', 'lat', 'lon']

    # Compute dump ratio
    hourly['dump_ratio'] = hourly['max_bin_mm'] / hourly['rainfall_mm'].clip(lower=1e-6)

    hourly = hourly[hourly['rainfall_mm'] >= min_rainfall_mm]

    samples = []
    for _, row in hourly.iterrows():
        samples.append({
            'hour_start': row['offset_hour'],
            'station_id': row['station_id'],
            'station_name': row['station_name'],
            'lat': row['lat'],
            'lon': row['lon'],
            'hourly_precip_mm': row['rainfall_mm'],
            'max_bin_mm': row['max_bin_mm'],
            'n_active_bins': int(row['n_active_bins']),
            'dump_ratio': row['dump_ratio'],
        })

    print(f"\nFound {len(samples)} offset hourly samples with rainfall >= {min_rainfall_mm}mm")
    return samples
#     """Incomplete function - not currently used"""
#     pass


def get_rainy_hours_set(day_list, min_rainfall_mm=0.01):
    """
    Return a set of naive UTC datetime objects (floored to the hour) where
    at least one gauge recorded rainfall >= min_rainfall_mm.

    Used by pull_nexrad_multi to skip radar scans during dry hours, which
    dramatically reduces the number of files that need to be gridded.

    Parameters
    ----------
    day_list : list of datetime.date
        The days to query (typically your rainy-days filter list).
    min_rainfall_mm : float
        Minimum hourly network-wide rainfall sum to consider the hour "rainy".
        Default is 0.01 mm — effectively "any measurable rain at all."

    Returns
    -------
    rainy_hours : set of datetime
        Naive UTC datetimes, each truncated to the hour.
        Example: {datetime(2024, 9, 8, 6, 0, 0), ...}
    """
    engine  = connect()
    session = create_session(engine)

    rain_ds     = session.query(DendraDatastream).filter(
        DendraDatastream.name.in_(["Rainfall", "Rainfall Sum"])
    ).all()
    rain_ds_ids = [ds.id for ds in rain_ds]

    start_dt = datetime.combine(min(day_list), datetime.min.time())
    end_dt   = datetime.combine(max(day_list), datetime.min.time()) + timedelta(days=1)

    results = (
        session.query(
            DendraDatapoint.timestamp_utc,
            DendraDatapoint.value,
        )
        .filter(DendraDatapoint.datastream_id.in_(rain_ds_ids))
        .filter(DendraDatapoint.timestamp_utc >= start_dt)
        .filter(DendraDatapoint.timestamp_utc <  end_dt)
        .all()
    )
    session.close()

    if not results:
        print("  ⚠ No gauge data found — rainy-hour filter will keep ALL files.")
        return set()

    df = pd.DataFrame(results, columns=['timestamp_utc', 'rainfall_mm'])
    df['hour'] = df['timestamp_utc'].dt.floor('h')

    # Sum across all stations per hour; keep hours with any measurable rain
    hourly_network = df.groupby('hour')['rainfall_mm'].sum()
    rainy_hours    = set(hourly_network[hourly_network >= min_rainfall_mm].index)

    # Strip timezone info so comparisons with naive datetimes (from filenames) work
    rainy_hours = {h.replace(tzinfo=None) if hasattr(h, 'tzinfo') else h
                   for h in rainy_hours}

    print(f"  Gauge DB: {len(rainy_hours)} rainy hours found across {len(day_list)} days "
          f"({len(rainy_hours) / max(len(day_list), 1):.1f} rainy hrs/day on average)")
    return rainy_hours


def save_rainy_days_list(filename='rainy_days_ranked.txt', top_n=100, min_rainfall=None,
                         start_date=None, end_date=None, metric='max_hourly'):
    """
    Save ranked list of rainy days to a text file for use with pull_nexrad
    
    Parameters:
    -----------
    filename : str
        Output filename (default: 'rainy_days_ranked.txt')
    top_n : int or None
        Number of days to save (if None, uses min_rainfall threshold)
    min_rainfall : float or None
        Minimum total rainfall (mm) to include a day (only used if top_n is None)
    start_date, end_date : datetime.date or str
        Date range to analyze
    metric : str
        Ranking metric (see rank_days_by_rainfall_intensity)
    
    Returns:
    --------
    top_days : list
        List of ranked rainy days
    """
    
    # Use keyword arguments to avoid positional argument issues
    top_days, stats = rank_days_by_rainfall_intensity(
        top_n=top_n,
        min_rainfall=min_rainfall,
        start_date=start_date,
        end_date=end_date,
        metric=metric
    )
    
    if len(top_days) == 0:
        print("No rainy days found!")
        return []
    
    # Save to file
    with open(filename, 'w') as f:
        f.write(f"# Top {len(top_days)} rainiest days (metric: {metric})\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Format: YYYY-MM-DD\n")
        f.write("#\n")
        for day in top_days:
            f.write(f"{day.strftime('%Y-%m-%d')}\n")
    
    print(f"✅ Saved {len(top_days)} rainy days to {filename}")
    
    return top_days