
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Vehicle Maintenance Prediction", layout="wide")

@st.cache_data
def load_data():
    np.random.seed(42)
    n=500
    data = {
        "Vehicle_ID": [f"VEH{i:04d}" for i in range(1,n+1)],
        "Mileage_km": np.random.randint(5000, 250000, n),
        "Engine_Temp_C": np.random.randint(70, 115, n),
        "Oil_Level_%": np.random.randint(5, 100, n),
        "Brake_Pad_mm": np.round(np.random.uniform(1, 12, n),1),
        "Tire_Pressure_PSI": np.random.randint(25, 44, n),
        "Battery_Voltage": np.round(np.random.uniform(10.5, 14.4, n),1),
        "Last_Service_Days": np.random.randint(10, 300, n),
    }
    df=pd.DataFrame(data)
    # simple rule for target
    df["Maintenance_Needed"] = ((df["Mileage_km"]>100000) | (df["Oil_Level_%"]<20) | (df["Brake_Pad_mm"]<3) | (df["Engine_Temp_C"]>105) | (df["Battery_Voltage"]<11.5) | (df["Last_Service_Days"]>180)).astype(int)
    return df

df = load_data()

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Dataset Preview", "EDA & Insights", "Model Training", "Prediction System"])

if menu == "Dataset Preview":
    st.title("Vehicle Maintenance - Dataset Preview")
    st.dataframe(df.head(100))
    st.download_button("Download CSV", df.to_csv(index=False), "vehicle_data.csv")
elif menu == "EDA & Insights":
    st.title("EDA & Insights")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x="Engine_Temp_C", color="Maintenance_Needed", title="Engine Temp vs Maintenance")
        st.plotly_chart(fig)
    with col2:
        fig2 = px.scatter(df, x="Mileage_km", y="Oil_Level_%", color="Maintenance_Needed", title="Mileage vs Oil Level")
        st.plotly_chart(fig2)
    fig3 = px.pie(df, names="Maintenance_Needed", title="Maintenance Needed Distribution")
    st.plotly_chart(fig3)
elif menu == "Model Training":
    st.title("Model Training")
    X = df[["Mileage_km","Engine_Temp_C","Oil_Level_%","Brake_Pad_mm","Tire_Pressure_PSI","Battery_Voltage","Last_Service_Days"]]
    y = df["Maintenance_Needed"]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    st.metric("Random Forest Accuracy", f"{acc*100:.2f}%")
    st.write("Feature Importance:")
    imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    st.bar_chart(imp.set_index("Feature"))
elif menu == "Prediction System":
    st.title("Vehicle Maintenance Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        mileage = st.number_input("Mileage (km)", 0, 300000, 75000)
        engine_temp = st.number_input("Engine Temp (C)", 60, 130, 90)
        oil = st.number_input("Oil Level %", 0, 100, 60)
    with col2:
        brake = st.number_input("Brake Pad mm", 1.0, 12.0, 6.0)
        tire = st.number_input("Tire Pressure PSI", 20, 50, 32)
    with col3:
        battery = st.number_input("Battery Voltage", 10.0, 15.0, 12.6)
        service = st.number_input("Last Service Days", 0, 500, 90)
    if st.button("Predict Maintenance"):
        risk_score = 0
        reasons=[]
        if mileage>100000: risk_score+=2; reasons.append("High Mileage")
        if engine_temp>105: risk_score+=2; reasons.append("Engine Overheating")
        if oil<20: risk_score+=3; reasons.append("Low Oil Level")
        if brake<3: risk_score+=3; reasons.append("Brake Pad Worn Out")
        if tire<28: risk_score+=1; reasons.append("Low Tire Pressure")
        if battery<11.5: risk_score+=2; reasons.append("Weak Battery")
        if service>180: risk_score+=1; reasons.append("Service Overdue")
        
        if risk_score>=5:
            st.error(f"HIGH RISK - Immediate Service Needed! Reasons: {', '.join(reasons)}")
            st.write("Recommendation: Gaadi ko turant service center le jao. Brake/Oil/Engine check karwao.")
        elif risk_score>=2:
            st.warning(f"MEDIUM RISK - Service Soon. Reasons: {', '.join(reasons) if reasons else 'Minor issues'}")
            st.write("Recommendation: 7 din ke andar servicing karwa lo.")
        else:
            st.success("LOW RISK - Vehicle is Healthy!")
            st.write("Recommendation: Regular checkup continue rakho.")

st.sidebar.markdown("---")
st.sidebar.info("Mini Project - Converted from Student Performance Predictor")
