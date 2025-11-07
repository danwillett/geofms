from weather.pull_weather import get_rainfall_days, label_rain_events
from radar.pull_nexrad import pull_nexrad
from radar.visualize_nexrad import show_nexrad
# from radar.validate_nexrad import validate_nexrad

from weather.pull_weather import save_rainy_days_list


if __name__ == "__main__":
    # station_rain = get_rainfall_days()
    # rain_days, no_rain_days = label_rain_events(station_rain, 0.1)

    # print(f"rain days: {min(rain_days)}")
    # print(f"no rain days: {min(no_rain_days)}")

    # Create list of top 100 rainiest days
    # save_rainy_days_list(
    #     filename='my_rainy_days.txt',
    #     top_n=2,
    #     metric='max_hourly'
    # )


    # Pull NEXRAD for only those days
    # pull_nexrad(day_filter_file='my_rainy_days.txt', apply_qc=True)
    

    
    
    # print("hey")
    # pull_nexrad()
    show_nexrad(time_idx=258, show_stations=True)