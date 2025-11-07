"""
Example workflow: Identify rainiest days and pull NEXRAD data for those days only

This demonstrates the complete pipeline:
1. Query weather station data to find rainiest days
2. Save ranked list of days
3. Pull NEXRAD radar data for only those days
"""

from weather.pull_weather import save_rainy_days_list, rank_days_by_rainfall_intensity
from radar.pull_nexrad import pull_nexrad
from datetime import date

def workflow_example():
    """
    Complete workflow example
    """
    
    print("="*80)
    print("RAINY DAY IDENTIFICATION & NEXRAD PROCESSING WORKFLOW")
    print("="*80)
    
    # Step 1: Identify rainiest days from weather station data
    print("\n📊 STEP 1: Analyzing weather station data...")
    print("-"*80)
    
    # Set your date range (should match your available weather data)
    start_date = date(2022, 10, 1)
    end_date = date(2025, 5, 12)
    
    # Rank days by maximum hourly rainfall rate
    # Options: 'max_hourly', 'mean_hourly', 'daily_total', 'peak_intensity'
    top_n_days = 100  # Number of rainiest days to process
    metric = 'max_hourly'  # Rank by maximum hourly rainfall rate
    
    # Get ranked list
    rainy_days, stats = rank_days_by_rainfall_intensity(
        top_n=top_n_days,
        start_date=start_date,
        end_date=end_date,
        metric=metric
    )
    
    if len(rainy_days) == 0:
        print("❌ No rainy days found! Check your database connection.")
        return
    
    # Step 2: Save to file
    print("\n💾 STEP 2: Saving rainy days list...")
    print("-"*80)
    
    output_file = 'rainy_days_top100.txt'
    save_rainy_days_list(
        filename=output_file,
        top_n=top_n_days,
        start_date=start_date,
        end_date=end_date,
        metric=metric
    )
    
    # Step 3: Pull NEXRAD data for only these days
    print("\n📡 STEP 3: Pulling NEXRAD data for rainy days...")
    print("-"*80)
    print(f"This will process NEXRAD data for {len(rainy_days)} days instead of")
    print(f"all {(end_date - start_date).days + 1} days in the date range.")
    print(f"Estimated file reduction: {100 * (1 - len(rainy_days) / ((end_date - start_date).days + 1)):.0f}%")
    print()
    
    response = input("Proceed with NEXRAD download? (y/n): ")
    
    if response.lower() == 'y':
        pull_nexrad(day_filter_file=output_file)
        print("\n✅ Complete! NEXRAD data downloaded for rainiest days only.")
    else:
        print("\n⏸️  NEXRAD download skipped. You can run it later with:")
        print(f"     pull_nexrad(day_filter_file='{output_file}')")
    
    print("\n" + "="*80)
    print("WORKFLOW COMPLETE")
    print("="*80)


def quick_analyze_only():
    """
    Just analyze and show the rainiest days without downloading NEXRAD
    """
    
    print("\n🔍 Quick Analysis: Rainiest Days")
    print("="*60)
    
    # Different metrics to compare
    metrics = {
        'max_hourly': 'Peak hourly rate',
        'daily_total': 'Total daily rainfall',
        'peak_intensity': 'Average station peak'
    }
    
    for metric_key, metric_name in metrics.items():
        print(f"\n{metric_name} ({metric_key}):")
        print("-"*60)
        
        rainy_days, stats = rank_days_by_rainfall_intensity(
            top_n=20,
            start_date=date(2022, 10, 1),
            end_date=date(2025, 5, 12),
            metric=metric_key
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'analyze':
        # Just analyze, don't download
        quick_analyze_only()
    else:
        # Full workflow
        workflow_example()


