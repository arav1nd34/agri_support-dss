import os
import glob
import streamlit as st
from soilquality import check_soil_quality
from irrigation import analyze_irrigation
from crop import analyze_crop_production
from weather import predict_weather

st.set_page_config(page_title="Smart Agri DSS", page_icon="🌾", layout="wide")

def main():
    os.makedirs('outputs', exist_ok=True)
    dataset = 'Smartagri/datasets/Crop_recommendation.csv'
    
    st.title("🌾 Smart Farming Decision Support System")
    st.caption("SIH 2026 Prototype — Multi-Module Agricultural Analytics Engine")
    st.divider()

    # Sidebar Options
    st.sidebar.header("Location & Settings")
    lat = st.sidebar.number_input("Latitude", value=28.6139, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=77.2090, format="%.4f")
    location = st.sidebar.text_input("Location Name", value="Noida, Uttar Pradesh")
    
    # Model Execution
    with st.spinner("Executing ML Models & Fetching Weather Data..."):
        soil_model, soil_acc = check_soil_quality(dataset)
        irr_model, irr_acc = analyze_irrigation(dataset)
        crop_model, crop_acc = analyze_crop_production(dataset)
        weather_data = predict_weather(latitude=lat, longitude=lon, location_name=location)

    st.success("✅ All ML modules executed successfully!")

    # Top Metric Overview Cards
    st.subheader("📊 System Summary & Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Soil Quality Model", f"{soil_acc * 100:.2f}%")
    col2.metric("Irrigation Model", f"{irr_acc * 100:.2f}%")
    col3.metric("Crop Production Model", f"{crop_acc * 100:.2f}%")
    col4.metric("Weather Status", "Fetched Live")

    st.divider()

    # Tabbed Interface
    tab1, tab2, tab3, tab4 = st.tabs(["🌱 Soil Quality", "💧 Irrigation", "🌾 Crop Analysis", "🌤️ Weather Forecast"])

    with tab1:
        st.subheader("Soil Quality Results")
        st.write(f"**Model Accuracy:** {soil_acc * 100:.2f}%")

    with tab2:
        st.subheader("Irrigation Analysis")
        st.write(f"**Model Accuracy:** {irr_acc * 100:.2f}%")

    with tab3:
        st.subheader("Crop Production Analysis")
        st.write(f"**Model Accuracy:** {crop_acc * 100:.2f}%")

    with tab4:
        st.subheader(f"Weather Forecast for {location}")
        st.json(weather_data) if isinstance(weather_data, dict) else st.write(weather_data)

    # Display Generated Chart Images from /outputs
    st.divider()
    st.subheader("🖼️ Generated Analysis Charts")
    output_images = glob.glob("outputs/*.*")
    
    if output_images:
        cols = st.columns(min(len(output_images), 3))
        for idx, img_path in enumerate(output_images):
            with cols[idx % 3]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info("No generated charts found in /outputs folder.")

if __name__ == "__main__":
    main()
