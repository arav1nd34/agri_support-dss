import os
from soilquality import check_soil_quality
from irrigation import analyze_irrigation
from crop import analyze_crop_production
from weather import predict_weather

def main():
   
    os.makedirs('outputs', exist_ok=True)
    
    dataset = 'datasets/Crop_recommendation.csv'
    
    soil_model, soil_acc = check_soil_quality(dataset)
    
    irr_model, irr_acc = analyze_irrigation(dataset)
    
    crop_model, crop_acc = analyze_crop_production(dataset)
    
    weather_data = predict_weather(
        latitude=28.6139,
        longitude=77.2090,
        location_name="Noida, Uttar Pradesh"
    )
    
    print("\n" + "=" * 60)
    print("   SYSTEM SUMMARY")
    print("=" * 60)
    print(f"  Soil Quality Model Accuracy     : {soil_acc*100:.2f}%")
    print(f"  Irrigation Model Accuracy       : {irr_acc*100:.2f}%")
    print(f"  Crop Production Model Accuracy  : {crop_acc*100:.2f}%")
    print(f"  Weather Forecast                : Fetched Successfully")
    print("=" * 60)
    print("\nAll modules executed successfully.")
    print("Output charts saved in /outputs folder.")

if __name__ == "__main__":
    main()