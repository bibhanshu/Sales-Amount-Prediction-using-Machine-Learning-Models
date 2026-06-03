
import pandas as pd
import streamlit as st

# import functions (we will adapt slightly)
from EV_Status_Model import model_status, x_encoded,EV_Status_Features
from Electric_Price_UP_Model import model_electric, x_price_encoded,Electric_price_feature_Imp

# ===============================
# USER INPUT
# ===============================


st.title("EV Charging Station Predictor 🔌")
st.subheader("Enter Charger Details")
city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"])
charger_type = st.selectbox("Charger Type", ['Fast','Slow'])
power = st.number_input("Power (kW): ")
temperature = st.number_input("Temperature (°C): ")
traffic = st.number_input("Traffic Level (1-10): ")
last_service = st.number_input("Last Service Days: ")
maintenance = st.slider("Maintenance Score", 0.0, 1.0, 0.5)
sessions = st.slider("Sessions Count: ", 0, 10000)
charging_time = st.slider("Charging Time (minutes): ", 0, 800, 300)
time_of_day = st.selectbox('Time Of the Day',['Morning','Afternoon','Evening','Night'])
peak_demand = st.selectbox("Peak Demand", [0, 1])
grid_supply = st.number_input("Grid Supply: ")

# ---- BUTTON + PREDICTION ----
if st.button("🔍 Predict"):
    user_data = pd.DataFrame({
    'City': [city],
    'Charger_Type': [charger_type],
    'Power_kW': [power],
    'Temperature': [temperature],
    'Traffic_Level': [traffic],
    'Maintenance_Score': [maintenance],
    'Last_Service_Days': [last_service],
    'Sessions_Count': [sessions],
    'Charging_Time': [charging_time],
    'Time_of_Day': [time_of_day],
    'Peak_Demand': [peak_demand],
    'Grid_Supply': [grid_supply]})
    user_encoded_status = pd.get_dummies(user_data)
    user_encoded_status = user_encoded_status.reindex(columns=x_encoded.columns, fill_value=0)
    status_prediction = model_status.predict(user_encoded_status)
    if status_prediction[0] == 1:
        st.success("\n✅ Charger Status: UP")
        st.subheader("Feature Importance — Status Model")
        st.dataframe(EV_Status_Features)
        user_encoded_price = pd.get_dummies(user_data)
        user_encoded_price = user_encoded_price.reindex(
        columns=x_price_encoded.columns,
        fill_value=0
        )
        price_prediction = model_electric.predict(user_encoded_price)[0]
        st.metric(label="💰 Predicted Electricity Price", value=f"₹{round(price_prediction, 2)}")
        st.subheader("Feature Importance — Price Model")
        st.dataframe(Electric_price_feature_Imp)
    else:
        st.error("❌ Charger Status: DOWN — Price not calculated")