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
    # Apply driver names permanently across the dataset layer based on truck IDs
    df_logs["driver_on_duty"] = df_logs["truck_id"].apply(assign_driver_to_truck)
except Exception as e:
    st.error(f"Platform Data Link Failure: Failed to pull real-time database logs from pgAdmin. Details: {e}")
    st.stop()

# ==========================================
# 2. EXECUTIVE CORE LOGISTICS METRICS (KPIs)
# ==========================================
total_fleet_units = df_logs["truck_id"].nunique()
active_moving_df = df_logs[df_logs["fleet_status"] == "In Transit"]
highway_stalls_df = df_logs[df_logs["fleet_status"] == "In Transit (Stopped/Unscheduled)"]
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
    if st.button("🚨 View Highway Delays (651 Logs)", use_container_width=True):
        st.subheader("📋 Roadside Stalls: Active Highway Interceptions Audit")
        delay_display_df = highway_stalls_df[[
            "timestamp", "truck_id", "driver_on_duty", "origin_depot", 
            "assigned_route", "distance_covered_km", "current_speed_kmh"
        ]].sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        st.dataframe(delay_display_df, use_container_width=True)

with audit_btn_col2:
    if st.button("⛽ View Fuel Theft Audit (867 Anomalies)", use_container_width=True):
        st.subheader("📋 Security Audit: Flagged Volumetric Fuel Drainage")
        theft_display_df = theft_anomalies_df[[
            "timestamp", "truck_id", "driver_on_duty", "fleet_status", 
            "origin_depot", "fuel_volume_lost_liters"
        ]].sort_values(by="fuel_volume_lost_liters", ascending=False).reset_index(drop=True)
        theft_display_df = theft_display_df.rename(columns={"fleet_status": "Location / Status Context"})
        st.dataframe(theft_display_df, use_container_width=True)

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
    fig_bar = px.bar(theft_by_depot, x="origin_depot", y="fuel_volume_lost_liters", color="origin_depot", labels={"fuel_volume_lost_liters": "Total Refined Liters Stolen"}, color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ==========================================
# 5. SYNCHRONIZED PREDICTIVE WORKSPACE (SMART DROP-DOWNS)
# ==========================================
st.header("🧠 Live Diagnostic & Dynamic Dispatch Terminal")
st.markdown("Select a **flagged blacklisted truck** directly from the audit targets below to dynamically pull its active telemetry state into the machine learning engine.")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("🔧 Telemetry Mechanical Breakdown Forecaster")
    
    # SMART FILTER: Pull ONLY Truck IDs that are actively sitting in the Highway Delays pool
    banned_breakdown_trucks = sorted(highway_stalls_df["truck_id"].unique().tolist())
    
    if banned_breakdown_trucks:
        selected_truck = st.selectbox("Select Target Flagged Asset for Live AI Diagnosis", banned_breakdown_trucks)
        
        # Pull the absolute LATEST database row for this specific truck to auto-populate sliders
        latest_truck_log = highway_stalls_df[highway_stalls_df["truck_id"] == selected_truck].sort_values(by="timestamp", ascending=False).iloc[0]
        
        # Dynamic telemetry variables pulled directly from your PostgreSQL data layer
        default_speed = int(latest_truck_log["current_speed_kmh"])
        default_temp = float(latest_truck_log["engine_temp_c"])
        default_service = int(latest_truck_log["hours_since_service"])
        default_total_dist = float(latest_truck_log["total_route_distance_km"])
        default_covered_dist = float(latest_truck_log["distance_covered_km"])
    else:
        # Fallback values if database is empty
        selected_truck = "OG-FLEET-1000"
        default_speed, default_temp, default_service, default_total_dist, default_covered_dist = 0, 90.0, 100, 130.0, 50.0
        st.selectbox("Select Target Flagged Asset for Live AI Diagnosis", [selected_truck])

    # Sliders now dynamically inherit their starting "value" from the selected blacklisted truck!
    speed_input = st.slider("Current Vehicle Speed (KM/H)", min_value=0, max_value=110, value=default_speed)
    temp_input = st.slider("Engine Block Thermal Sensor (°C)", min_value=40, max_value=140, value=int(default_temp))
    service_input = st.number_input("Cumulative Operational Running Hours Since Service Check", value=default_service)
    route_dist_input = st.number_input("Assigned Route Distance Framework (KM)", value=int(default_total_dist))
    covered_dist_input = st.number_input("Current Odometer Distance Covered on Route (KM)", value=int(default_covered_dist))
    
    if st.button("Run System Component Risk Check"):
        active_driver = assign_driver_to_truck(selected_truck)
        
        st.markdown("#### 📝 Telemetry Input Frame Sent to Model:")
        st.json({
            "Asset ID": selected_truck,
            "Driver Accountable": active_driver,
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
    
    # SMART FILTER: Pull ONLY Truck IDs that have a confirmed Fuel Theft Anomaly flag in the database
    theft_target_trucks = sorted(theft_anomalies_df["truck_id"].unique().tolist())
    
    if theft_target_trucks:
        selected_truck_2 = st.selectbox("Select Flagged Security Target for Integrity Analysis", theft_target_trucks)
        
        # Pull the absolute LATEST database record for this flagged theft asset
        latest_theft_log = theft_anomalies_df[theft_anomalies_df["truck_id"] == selected_truck_2].sort_values(by="timestamp", ascending=False).iloc[0]
        
        default_route = latest_theft_log["assigned_route"]
        default_status = latest_theft_log["fleet_status"]
        default_speed_2 = int(latest_theft_log["current_speed_kmh"])
        default_temp_2 = float(latest_theft_log["engine_temp_c"])
    else:
        selected_truck_2 = "OG-FLEET-1200"
        default_route = "Lagos-Ibadan Expressway Corridor"
        default_status = "In Transit (Stopped/Unscheduled)"
        default_speed_2, default_temp_2 = 0, 45.0
        st.selectbox("Select Flagged Security Target for Integrity Analysis", [selected_truck_2])

    target_corridor = st.selectbox("Assign Downstream Product Transit Corridor Axis", [
        "Lagos-Ibadan Expressway Corridor", 
        "Port Harcourt-Aba Road Corridor", 
        "Sagamu-Benin Express Corridor"
    ], index=["Lagos-Ibadan Expressway Corridor", "Port Harcourt-Aba Road Corridor", "Sagamu-Benin Express Corridor"].index(default_route))
    
    live_fleet_status = st.selectbox("Current Operational Vehicle Log Status", [
        "In Transit", 
        "In Transit (Stopped/Unscheduled)", 
        "Idle at Depot",
                "Under Maintenance", 
        "Off-Duty"
    ], index=["In Transit", "In Transit (Stopped/Unscheduled)", "Idle at Depot", "Under Maintenance", "Off-Duty"].index(default_status))
    
    distance_indexer = {"Lagos-Ibadan Expressway Corridor": 130.0, "Port Harcourt-Aba Road Corridor": 65.0, "Sagamu-Benin Express Corridor": 260.0}
    selected_km = distance_indexer[target_corridor]

    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Calculate Dynamic Estimated Duration"):
            active_driver_2 = assign_driver_to_truck(selected_truck_2)
            
            st.markdown("#### 📝 Logistics Snapshot:")
            st.json({
                "Asset ID": selected_truck_2,
                "Driver Accountable": active_driver_2,
                "Assigned Corridor": target_corridor,
                "Baseline Distance": f"{selected_km} KM",
                "Asset Speed Vector": f"{speed_input} KM/H"
            })
            
            eta = predict_transit_duration(selected_km, speed_input, target_corridor)
            st.info(f"Predicted Travel Window: **{round(eta, 2)} Hours**")
            
    with col_btn2:
        if st.button("Check Fuel Tank Security Integrity"):
            active_driver_2 = assign_driver_to_truck(selected_truck_2)
            
            st.markdown("#### 📝 Security State Snapshot:")
            st.json({
                "Asset ID": selected_truck_2,
                "Driver Accountable": active_driver_2,
                "Corridor Space": target_corridor,
                "Velocity Context": f"{speed_input} KM/H",
                "Reported Status": live_fleet_status
            })
            
            theft_risk = predict_fuel_theft_risk(speed_input, temp_input, target_corridor, live_fleet_status)
            if theft_risk >= 0.75:
                st.error(f"🚨 THEFT HIGH RISK FLAG: {round(theft_risk * 100, 2)}% probability of ongoing fuel siphoning.")
            else:
                st.success(f"✅ Tank Volume Secure. Theft Anomaly Risk: {round(theft_risk * 100, 2)}%")
