import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_pipeline import extract_data_from_db
from src.inference import predict_breakdown_risk, predict_transit_duration, predict_fuel_theft_risk, assign_driver_to_truck

# 1. Platform Layout Configuration
st.set_page_config(layout="wide", page_title="Downstream Telematics Tower")
st.title("🛢️ Downstream Oil & Gas Intelligent Fleet Telematics Platform")
st.markdown("Enterprise Operations Control Center mapping Asset Integrity, Fuel Security Anomalies, and Real-Time Predictive Transit Windows.")

# Safely extract dynamic log views from Database Pipeline module
try:
    df_logs = extract_data_from_db()
    
    # FIX 1: Normalize all database columns and string text to lowercase to eliminate matching errors
    df_logs.columns = df_logs.columns.str.lower()
    df_logs["fleet_status"] = df_logs["fleet_status"].astype(str).str.lower().str.strip()
    
    # Apply driver names permanently across the dataset layer based on truck IDs
    df_logs["driver_on_duty"] = df_logs["truck_id"].apply(assign_driver_to_truck)
except Exception as e:
    st.error(f"Platform Data Link Failure: Failed to pull real-time database logs from pgAdmin. Details: {e}")
    st.stop()

# ==========================================
# 2. EXECUTIVE CORE LOGISTICS METRICS (KPIs)
# ==========================================
total_fleet_units = df_logs["truck_id"].nunique()

# FIX 2: Filter matching the lowercase string normalization applied above
active_moving_df = df_logs[df_logs["fleet_status"] == "in transit"]
highway_stalls_df = df_logs[df_logs["fleet_status"] == "in transit (stopped/unscheduled)"]
theft_anomalies_df = df_logs[df_logs["fuel_theft_anomaly"] == 1]

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric(label="Total Tracked Fleet Assets", value=f"{total_fleet_units} Heavy Trucks")
kpi_col2.metric(label="Active Moving Delivery Vectors", value=f"{len(active_moving_df)} Snapshots", delta="Cruising Norm")
kpi_col3.metric(label="Highway Delays / Unscheduled Stops", value=f"{len(highway_stalls_df)} Flagged Logs", delta="Risk Level", delta_color="inverse")
kpi_col4.metric(label="Flagged Fuel Siphoning Incidents", value=f"{len(theft_anomalies_df)} Anomalies", delta="Financial Leakage", delta_color="inverse")

# ==========================================
# 3. INTERACTIVE SECURITY DRILL-DOWN AUDIT BUTTONS
# ==========================================
st.markdown("### 🗃️ Management Incident Investigation Terminal")
st.markdown("Click the options below to instantly inspect blacklisted asset logs, trace vehicle operational contexts, and pull driver IDs for questioning.")

audit_btn_col1, audit_btn_col2 = st.columns(2)

with audit_btn_col1:
    # FIX 3: Make button text dynamic using len() so it accurately represents database states
    if st.button(f"🚨 View Highway Delays ({len(highway_stalls_df)} Logs)", use_container_width=True):
        st.subheader("📋 Roadside Stalls: Active Highway Interceptions Audit")
        if not highway_stalls_df.empty:
            delay_display_df = highway_stalls_df[[
                "timestamp", "truck_id", "driver_on_duty", "origin_depot", 
                "assigned_route", "distance_covered_km", "current_speed_kmh"
            ]].sort_values(by="timestamp", ascending=False).reset_index(drop=True)
            st.dataframe(delay_display_df, use_container_width=True)
        else:
            st.info("No highway stalls recorded in current data frame frame matching filter.")

with audit_btn_col2:
    # FIX 4: Make fuel theft button dynamic using len()
    if st.button(f"⛽ View Fuel Theft Audit ({len(theft_anomalies_df)} Anomalies)", use_container_width=True):
        st.subheader("📋 Security Audit: Flagged Volumetric Fuel Drainage")
        if not theft_anomalies_df.empty:
            theft_display_df = theft_anomalies_df[[
                "timestamp", "truck_id", "driver_on_duty", "fleet_status", 
                "origin_depot", "fuel_volume_lost_liters"
            ]].sort_values(by="fuel_volume_lost_liters", ascending=False).reset_index(drop=True)
            theft_display_df = theft_display_df.rename(columns={"fleet_status": "Location / Status Context"})
            st.dataframe(theft_display_df, use_container_width=True)
        else:
            st.info("No siphoning anomalies found in data source query loops.")

st.markdown("---")

# ==========================================
# 4. INTERACTIVE PLOTLY VISUALIZATIONS
# ==========================================
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.subheader("📊 Macro Fleet Distribution Footprint")
    status_summary = df_logs.groupby("fleet_status")["truck_id"].count().reset_index().rename(columns={"truck_id": "log_count"})
    fig_pie = px.pie(status_summary, names="fleet_status", values="log_count", hole=0.45, color_discrete_sequence=px.colors.sequential.Bluyl_r)
    st.plotly_chart(fig_pie, use_container_width=True)

with viz_col2:
    st.subheader("⛽ Volumetric Product Loss by Distribution Depot")
    theft_by_depot = df_logs[df_logs["fuel_theft_anomaly"] == 1].groupby("origin_depot")["fuel_volume_lost_liters"].sum().reset_index()
    if not theft_by_depot.empty:
        fig_bar = px.bar(theft_by_depot, x="origin_depot", y="fuel_volume_lost_liters", color="origin_depot", labels={"fuel_volume_lost_liters": "Total Refined Liters Stolen"}, color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No lost product volume analytics to track on chart layers.")

st.markdown("---")

# ==========================================
# 5. LIVE MACHINE LEARNING PREDICTIVE WORKSPACE
# ==========================================
st.header("🧠 Live Diagnostic & Dynamic Dispatch Terminal")
st.markdown("Interact directly with underlying edge analytics engines to evaluate vehicle strain profiles and route delivery timetables.")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("🔧 Telemetry Mechanical Breakdown Forecaster")
    mock_truck_select = st.selectbox("Select Target Fleet Asset for Simulation", [f"OG-FLEET-{1000 + i}" for i in range(20)])
    speed_input = st.slider("Current Vehicle Speed (KM/H)", min_value=0, max_value=110, value=80, key="breakdown_speed")
    temp_input = st.slider("Engine Block Thermal Sensor (°C)", min_value=40, max_value=140, value=85)
    service_input = st.number_input("Cumulative Operational Running Hours Since Service Check", value=120)
    route_dist_input = st.number_input("Assigned Route Distance Framework (KM)", value=300)
    covered_dist_input = st.number_input("Current Odometer Distance Covered on Route (KM)", value=140)
    
    if st.button("Run System Component Risk Check"):
        active_driver = assign_driver_to_truck(mock_truck_select)
        st.markdown("#### 📝 Telemetry Input Frame Sent to Model:")
        st.json({
            "Asset ID": mock_truck_select,
            "Driver Assigned": active_driver,
            "Velocity Metrics": f"{speed_input} KM/H",
            "Thermal Reading": f"{temp_input} °C",
            "Engine Wear State": f"{service_input} Run-Hours",
            "Logistics Leg": f"{covered_dist_input} KM / {route_dist_input} KM"
        })
        
        risk = predict_breakdown_risk(speed_input, temp_input, service_input, route_dist_input, covered_dist_input)
        if risk >= 0.50:
            st.error(f"🚨 ALERT: High Predictive Mechanical Breakdown Probability ({round(risk * 100, 2)}%). Halt route transit immediately.")
        else:
            st.success(f"✅ Safe Operating Tolerances Verified. Nominal Breakdown Variance: {round(risk * 100, 2)}%")

with input_col2:
    st.subheader("⏱️ Transit Corridor Duration & Fuel Integrity Engine")
    mock_truck_select_2 = st.selectbox("Select Active Route Asset for Simulation", [f"OG-FLEET-{1200 + i}" for i in range(20)])
    target_corridor = st.selectbox("Assign Downstream Product Transit Corridor Axis", [
        "Lagos-Ibadan Expressway Corridor", 
        "Port Harcourt-Aba Road Corridor", 
        "Sagamu-Benin Express Corridor"
    ])
    live_fleet_status = st.selectbox("Current Operational Vehicle Log Status", [
        "In Transit", 
        "In Transit (Stopped/Unscheduled)", 
        "Idle at Depot", 
        "Under Maintenance", 
        "Off-Duty"
    ])
    
    distance_indexer = {"Lagos-Ibadan Expressway Corridor": 130.0, "Port Harcourt-Aba Road Corridor": 65.0, "Sagamu-Benin Express Corridor": 260.0}
    selected_km = distance_indexer[target_corridor]

    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Calculate Dynamic Estimated Duration"):
            active_driver_2 = assign_driver_to_truck(mock_truck_select_2)
            st.markdown("#### 📝 Logistics Snapshot:")
            eta_hours = predict_transit_duration(selected_km, speed_input, target_corridor)
            st.metric(label="Predicted Dynamic Duration", value=f"{round(eta_hours, 1)} Hours")
            st.json({
                "Asset ID": mock_truck_select_2,
                "Driver Assigned": active_driver_2,
                "Assigned Corridor": target_corridor,
                "Estimated ETA": f"{round(eta_hours, 1)} Hours"
            })
            
    with col_btn2:
        if st.button("Evaluate Live Route Security Risk"):
            theft_risk = predict_fuel_theft_risk(speed_input, temp_input, target_corridor, live_fleet_status)
            if theft_risk >= 0.50:
                st.error(f"🚨 CRITICAL SECURITY EXPOSURE: Fuel theft siphoning anomaly suspected! Risk score: {round(theft_risk * 100, 2)}%")
            else:
                st.success(f"🛡️ Cargo Security Secure. Theft likelihood negligible: {round(theft_risk * 100, 2)}%")
