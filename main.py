"""
Main entry point for NEXRAD + rain gauge ML pipeline

Run from project root to ensure proper .env loading
"""

import datetime as dt
from radar.pull_nexrad_multi import pull_nexrad_multi
from weather.pull_weather import get_hourly_precipitation_by_station, save_rainy_days_list, get_rainfall_days, label_rain_events, rank_days_by_rainfall_intensity
from radar.pull_nexrad import pull_nexrad
from radar.visualize_nexrad import show_nexrad, show_dem_with_stations
from radar.diagnose_zarr import diagnose_zarr

# from radar.validate_nexrad import validate_nexrad


def find_best_training_days(top_n=20, start_date='2020-01-01', end_date='2026-3-28', max_valid_rainfall=100.0):
    """
    Find the best rainy days for ML training.
    
    Uses 'rain_hours' metric: counts station-hours with rain > threshold.
    This finds days with widespread, consistent rain (not just single spikes).
    Filters out sensor errors (readings > max_valid_rainfall).
    
    Parameters:
    -----------
    top_n : int
        Number of top rainy days to return
    output_file : str
        File to save the list of dates
    max_valid_rainfall : float
        Maximum physically plausible rainfall in mm/hr (default: 100.0)
        Readings above this are likely sensor errors and will be filtered out.
    """
    print("\n" + "="*70)
    print("FINDING BEST RAINY DAYS FOR MODEL TRAINING")
    print("="*70)
    print("\nLooking for days with widespread, consistent rain...")
    print("(Not just days with single extreme spikes!)\n")

    output_file = f"weather/days/top_{top_n}_days_{start_date}_{end_date}.txt"
    
    # Find top N days ranked by number of rain-hours
    top_days, day_stats = rank_days_by_rainfall_intensity(
        top_n=top_n,
        metric='rain_hours',           # Counts station-hours with rain
        min_rainfall=0.5,              # Count hours with > 0.5 mm/hr
        start_date=start_date,       # Extended date range
        end_date=end_date,         # Through end of 2025
        max_valid_rainfall=max_valid_rainfall  # Filter sensor errors
    )
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write("# Top rainy days for model training\n")
        f.write("# Ranked by number of station-hours with rain\n")
        f.write("# Format: YYYY-MM-DD\n\n")
        for day in top_days:
            f.write(f"{day.strftime('%Y-%m-%d')}\n")
    
    print(f"\n✓ Saved top {len(top_days)} days to: {output_file}")
    print(f"\n💡 Next step: Pull NEXRAD radar for these days")
    print(f"   Uncomment STEP 2 in main() and run!\n")
    
    # Show detailed stats
    print("\n" + "="*70)
    print("DETAILED STATISTICS FOR TOP DAYS")
    print("="*70)
    
    for i, row in day_stats.head(min(10, len(day_stats))).iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        print(f"\n{date_str}:")
        print(f"  Rain-hours: {row['rain_hours']:.0f} (station-hours with > 0.1 mm/hr)")
        print(f"  Stations: {row['n_stations']:.0f} stations had rain")
        print(f"  Total rainfall: {row['total_rain']:.1f} mm")
        print(f"  Mean rain rate: {row['mean_rain_rate']:.2f} mm/hr (when raining)")
        print(f"  Max rain rate: {row['max_rain_rate']:.1f} mm/hr")
    
    print("\n" + "="*70)
    print(f"✓ Found {len(top_days)} excellent training days!")
    print("="*70 + "\n")
    
    return output_file


def prepare_training_data(train_years=None, val_years=None, max_valid_rainfall=100.0):
    """
    Prepare aligned radar-gauge dataset for ML training
    
    Parameters:
    -----------
    train_years : list of int, optional
        Years to use for training (e.g., [2023])
        If None, uses random 80/20 split
    val_years : list of int, optional
        Years to use for validation (e.g., [2024])
        Required if train_years is provided
    max_valid_rainfall : float
        Maximum valid rainfall threshold (default: 100.0 mm/hr)
    """
    from deep_learning.prepare_radar_gauge_data import create_training_samples, inspect_dataset
    
    print("\n" + "="*60)
    print("PREPARING TRAINING DATA")
    print("="*60)
    
    # Create dataset
    dataset = create_training_samples(
        radar_zarr_path='KVBX_preserve_500m.zarr',
        output_path='deep_learning/radar_gauge_dataset.pkl',
        train_years=train_years,
        val_years=val_years,
        day_filter_file='my_rainy_days.txt',
        max_valid_rainfall=max_valid_rainfall
    )
    
    # Inspect it
    if dataset:
        print("\n" + "="*60)
        print("INSPECTING DATASET")
        print("="*60)
        inspect_dataset('deep_learning/radar_gauge_dataset.pkl')

def test_precip():
    import datetime
    data = get_hourly_precipitation_by_station(start_date=datetime.date(2024, 4, 12), end_date=datetime.date(2024, 4, 16), min_rainfall_mm=0)
    for dict in data:
        if dict['station_name'] == 'Dangermond_Jalachichi':
            print(dict)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-MODAL PRECIPITATION PREDICTION - DATA PIPELINE")
    print("="*70)
    print("\nFollow these steps in order:\n")
    print("1. Find best rainy days (using 'rain_hours' metric)")
    print("2. Pull NEXRAD radar data for those days")
    print("3. Prepare training dataset (align radar + gauges)")
    print("4. Train model (see train_precipitation_model.ipynb)")
    print("\n" + "="*70 + "\n")
    
    # ============================================================
    # STEP 1: Find best rainy days for training
    # ============================================================
    # Uncomment to find days with widespread, consistent rain
    # (not just days with single sensor spikes!)

    START_DATE = dt.date(2022, 1, 1)
    END_DATE   = dt.date(2026, 4, 4)

    rainy_file = find_best_training_days(
        top_n=100,                              # Find top 100 rainy days
        start_date=START_DATE.strftime('%Y-%m-%d'),
        end_date=END_DATE.strftime('%Y-%m-%d'),
        max_valid_rainfall=100.0               # Filter out sensor errors > 100 mm/hr
    )
    
    # # ============================================================
    # # STEP 2: Pull NEXRAD radar for those days
    # # ============================================================
    # # Uncomment after STEP 1 completes
    
    print("\n🌧️  Pulling NEXRAD radar data for rainy days...")
    zarr_path = pull_nexrad_multi(
        day_filter_file=rainy_file,
        apply_qc=True,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    
    test_precip()

    # ============================================================
    # STEP 3: Build training pickle from zarr
    # ============================================================
    # Uncomment after STEP 2 completes

    # from dataset.create_pickle import create_training_samples
    # # zarr_path = r"radar\outputs\dualpol_500m_2022-01-01_2026-04-04.zarr"
    # # rainy_file = r"weather\days\top_100_days_2022-01-01_2026-04-04.txt"
    # pickle_path = create_training_samples(
    #     radar_zarr_path=zarr_path,
    #     output_path='dataset/outputs/3d/radar_gauge_dataset_tr22_24_26_vl_23_25.pkl',
    #     dem_path='dem/preserve_dem_10m_utm.tif',
    #     train_years=[2022, 2024, 2026],   # adjust to match zarr date range
    #     val_years=[2023, 2025],
    #     day_filter_file=rainy_file,
    #     max_valid_rainfall=100.0,
    #     patch_size_m=4500,
    # )
    
    # ============================================================
    # STEP 4: Visualize radar data (optional)
    # ============================================================
    # Uncomment to see radar reflectivity maps
    
    # show_nexrad(datetime_target="2025-11-15 15")
    # show_dem_with_stations(dem_path='preserve_dem_10m_utm.tif')

    

    # ============================================================
    # TIMEZONE DIAGNOSTIC
    # ============================================================
    # import zarr
    # import pandas as pd
    # import pickle

    # # 1. Check NEXRAD timestamps
    # print("=" * 60)
    # print("NEXRAD RADAR TIMESTAMPS")
    # print("=" * 60)
    # store = zarr.open("KVBX_preserve_500m.zarr", mode='r')
    # radar_times = pd.to_datetime(store['time'][:])
    # print(f"First 5 radar times: {radar_times[:5].tolist()}")
    # print(f"Sample: {radar_times[100]}")
    # print(f"Timezone info: {radar_times[0].tzinfo}")  # None = naive, otherwise shows tz

    # # 2. Check rain gauge timestamps
    # print("\n" + "=" * 60)
    # print("RAIN GAUGE TIMESTAMPS")
    # print("=" * 60)
    # with open('deep_learning/radar_gauge_dataset_50.pkl', 'rb') as f:
    #     data = pickle.load(f)

    # train_samples = data.get('train', [])[:5]
    # for s in train_samples:
    #     print(f"  {s.get('hour_start', s.get('hour', 'N/A'))} - Station: {s.get('station_name', 'N/A')}")

    # # 3. Check if same event matches
    # print("\n" + "=" * 60)
    # print("ALIGNMENT CHECK")
    # print("=" * 60)
    # # Pick a rain event and check if radar times align
    # sample = train_samples[0]
    # gauge_time = pd.Timestamp(sample.get('hour_start', sample.get('hour')))
    # print(f"Gauge timestamp: {gauge_time}")

    # # Find closest radar time
    # time_diffs = abs(radar_times - gauge_time)
    # closest_idx = time_diffs.argmin()
    # print(f"Closest radar: {radar_times[closest_idx]}")
    # print(f"Time difference: {time_diffs[closest_idx]}")

    # if time_diffs[closest_idx] > pd.Timedelta(hours=1):
    #     print("\n⚠️  WARNING: Time difference > 1 hour!")
    #     print("   Possible timezone mismatch!")
    # else:
    #     print("\n✅ Times appear aligned (within 1 hour)")