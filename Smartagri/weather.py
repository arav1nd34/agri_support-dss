import requests
import json
from datetime import datetime

def predict_weather(latitude=28.6139, longitude=77.2090, location_name="Noida"):
    print("\n=== WEATHER PREDICTOR ===")
    print(f"Fetching weather data for {location_name}...")
    
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        f"windspeed_10m_max,weathercode"
        f"&timezone=Asia/Kolkata&forecast_days=7"
    )
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        daily = data['daily']
        dates = daily['time']
        temp_max = daily['temperature_2m_max']
        temp_min = daily['temperature_2m_min']
        precipitation = daily['precipitation_sum']
        windspeed = daily['windspeed_10m_max']
        
        print(f"\n7-Day Weather Forecast for {location_name}:")
        print("-" * 65)
        print(f"{'Date':<15}{'Max Temp':>10}{'Min Temp':>10}{'Rain(mm)':>10}{'Wind':>10}")
        print("-" * 65)
        
        for i in range(7):
            print(f"{dates[i]:<15}{temp_max[i]:>9.1f}C{temp_min[i]:>9.1f}C"
                  f"{precipitation[i]:>10.1f}{windspeed[i]:>9.1f}")
        
        print("-" * 65)
        
        avg_rain = sum(precipitation) / 7
        avg_temp = sum(temp_max) / 7
        
        print("\nFarming Advisory:")
        if avg_rain > 10:
            print("  - High rainfall expected. Reduce irrigation this week.")
        elif avg_rain < 2:
            print("  - Low rainfall expected. Irrigation recommended.")
        else:
            print("  - Moderate rainfall. Monitor soil moisture.")
            
        if avg_temp > 35:
            print("  - High temperatures. Ensure adequate watering.")
        elif avg_temp < 15:
            print("  - Cool conditions. Protect sensitive crops from cold.")
        else:
            print("  - Temperature conditions are favorable for most crops.")
        
        return data
        
    except Exception as e:
        print(f"Could not fetch weather data: {e}")
        print("Please check your internet connection.")
        return None