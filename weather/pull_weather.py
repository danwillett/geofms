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
    rain_ds = session.query(DendraDatastream).filter(DendraDatastream.name == "Rainfall").all()
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
                                     metric='max_hourly'):
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
    
    if len(top_days) > 10:
        print(f"  ... and {len(top_days) - 10} more days")
    
    print(f"\n{'='*60}\n")
    
    return top_days, daily_stats

def create_rainy_days_dict(top_days):

    engine = connect()
    session = create_session(engine)
    rainy_days_dict = {}

    for day in top_days:
        query = session.query(DendraDatapoint).filter(DendraDatapoint.timestamp_utc.date() == day).filter(DendraDatapoint.datastream_id.in_(rain_ds_ids)).all()





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