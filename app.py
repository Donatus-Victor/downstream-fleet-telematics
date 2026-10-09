import streamlit as st
import plotly.express as px
import pandas as pd
from src.data_pipeline import extract_data_from_db
from src.inference import predict_breakdown_risk, predict_transit_duration, predict_fuel_theft_risk, assign_driver_to_truck

# 1. Platform Layout Configuration
# st.set_page_config(layout="wide", page_title="Downstream Telematics Tower")
# st.title("🛢️ Downstream Oil & Gas Intelligent Fleet Telematics Platform")
# st.markdown("Enterprise Operations Control Center mapping Asset Integrity, Fuel Security Anomalies, and Real-Time Predictive Transit Windows.")

st.set_page_config(layout="wide", page_title="Fleet Telematics Control Center")
st.title("🛢️ Oil & Gas Fleet Telematics Control Center")
st.caption("Monitor asset health, fuel level anomalies, and predicted transit times in one place.")

# Safely extract dynamic log views from Database Pipeline module
try:
    df_logs = extract_data_from_db()
    
    # Normalize all database columns and string text to lowercase to eliminate matching errors
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

# Filter matching the lowercase string normalization applied above
active_moving_df = df_logs[df_logs["fleet_status"] == "in transit"]
highway_stalls_df = df_logs[df_logs["fleet_status"] == "in transit (stopped/unscheduled)"]
theft_anomalies_df = df_logs[df_logs["fuel_theft_anomaly"] == 1]

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
kpi_col1.metric(label="Total Tracked Fleet Assets", value=f"{total_fleet_units} Heavy Trucks")
kpi_col2.metric(label="Active Moving Delivery Vectors", value=f"{len(active_moving_df)} Snapshots", delta="Cruising Norm")
kpi_col3.metric(label="Highway Delays / Unscheduled Stops", value=f"{len(highway_stalls_df)} Flagged Logs", delta="Risk Level", delta_color="inverse")
kpi_col4.metric(label="Sudden Fuel Level Drop Events", value=f"{len(theft_anomalies_df)} Anomalies", delta="Financial Leakage", delta_color="inverse")

# ==========================================
# 3. SMART ASSET VERIFICATION & SEARCH TOOL
# ==========================================
st.markdown("---")
st.markdown("### 🔍 Truck Lookup")
search_query = st.text_input("Validate Truck ID registry existence (e.g., 1003 or OG-FLEET-1003):", placeholder="Type digits or full vehicle key here...").strip().lower()

# Default fallback values for workspace simulations
default_truck = "OG-FLEET-1000"
default_speed, default_temp, default_service, default_total_dist, default_covered_dist = 80, 85, 120, 300, 140
default_route = "Sagamu-Benin Express Corridor"
default_status = "In Transit"

if search_query:
    unique_trucks = df_logs["truck_id"].dropna().unique().tolist()
    matched_trucks = [t for t in unique_trucks if search_query in t.lower()]
    
    if matched_trucks:
        verified_id = matched_trucks[0]
        st.success(f"✅ Valid Asset Verified: Match found for asset reference **'{verified_id.upper()}'** in pgAdmin registry.")
        
        # Isolate latest log entry for the searched vehicle
        truck_history = df_logs[df_logs["truck_id"] == verified_id].sort_values(by="timestamp", ascending=False)
        latest_match = truck_history.iloc[0]
        
        # Overwrite fallback variables to auto-populate the simulation panels lower down
        default_truck = verified_id
        default_speed = int(latest_match["current_speed_kmh"])
        default_temp = int(latest_match["engine_temp_c"])
        default_service = int(latest_match["hours_since_service"])
        default_total_dist = int(latest_match["total_route_distance_km"])
        default_covered_dist = int(latest_match["distance_covered_km"])
        
        # Display the real-time operational lookup card
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.info(f"**Verified ID Key:**\n\n{verified_id.upper()}")
        sc2.info(f"**Assigned Operator:**\n\n{latest_match['driver_on_duty']}")
        sc3.info(f"**Telemetry Status:**\n\n{latest_match['fleet_status'].title()}")
        
        # Pre-calculate asset integrity classification profile score
        risk_check = predict_breakdown_risk(default_speed, default_temp, default_service, default_total_dist, default_covered_dist)
        if risk_check >= 0.50:
            sc4.error(f"**Edge Health Profiler:**\n\n⚠️ HIGH BREAKDOWN RISK ({round(risk_check*100, 1)}%)")
        else:
            sc4.success(f"**Edge Health Profiler:**\n\n🛡️ OPERATIONAL NOMINAL ({round(risk_check*100, 1)}%)")
            
        # Display underlying database record subsets
        st.markdown(f"**Log History Summary Data Frame for Vehicle Context:**")
        st.dataframe(truck_history[["timestamp", "origin_depot", "assigned_route", "current_speed_kmh", "engine_temp_c", "fuel_theft_anomaly"]].reset_index(drop=True), use_container_width=True)
    else:
        st.error(f"❌ Invalid Truck ID: Sequence string '{search_query.upper()}' does not match any operational logs inside current registry matrix.")

# ==========================================
# 4. INTERACTIVE SECURITY DRILL-DOWN AUDIT BUTTONS
# ==========================================
st.markdown("---")
st.markdown("### 🗃️ Incident Review Center")
st.markdown("Review flagged trucks, check their trip context, and find the assigned driver for follow-up.")

audit_btn_col1, audit_btn_col2 = st.columns(2)

with audit_btn_col1:
    if st.button(f"🚨 View Highway Delays ({len(highway_stalls_df)} Logs)", use_container_width=True):
        st.subheader("📋 🚧 Highway Stops & Delays")
        if not highway_stalls_df.empty:
            delay_display_df = highway_stalls_df[[
                "timestamp", "truck_id", "driver_on_duty", "origin_depot", 
                "assigned_route", "distance_covered_km", "current_speed_kmh"
            ]].sort_values(by="timestamp", ascending=False).reset_index(drop=True)
            st.dataframe(delay_display_df, use_container_width=True)
        else:
            st.info("No highway stalls recorded in current data frame matching filter.")

with audit_btn_col2:
    if st.button(f"⛽ View Fuel Drop Audit ({len(theft_anomalies_df)} Anomalies)", use_container_width=True):
        st.subheader("📋 Flagged Fuel Level Drop Events")
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
# 5. INTERACTIVE PLOTLY VISUALIZATIONS
# ==========================================
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    st.subheader("📊 Fleet Status Overviewt")
    status_summary = df_logs.groupby("fleet_status")["truck_id"].count().reset_index().rename(columns={"truck_id": "log_count"})
    fig_pie = px.pie(status_summary, names="fleet_status", values="log_count", hole=0.45, color_discrete_sequence=px.colors.sequential.Bluyl_r)
    st.plotly_chart(fig_pie, use_container_width=True)

with viz_col2:
    st.subheader("⛽ Fuel Level Drops by Depot")
    theft_by_depot = df_logs[df_logs["fuel_theft_anomaly"] == 1].groupby("origin_depot")["fuel_volume_lost_liters"].sum().reset_index()
    if not theft_by_depot.empty:
        fig_bar = px.bar(theft_by_depot, x="origin_depot", y="fuel_volume_lost_liters", color="origin_depot",
        labels={"fuel_volume_lost_liters": "Total Refined Liters Stolen"},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    if not theft_by_depot.empty:
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No volumetric product loss parameters logged to display.")

st.markdown("---")

# ==========================================
# 6. SYNCHRONIZED PREDICTIVE WORKSPACE (SMART DROP-DOWNS)
# ==========================================
st.header("Live Diagnostics & Dispatch")
st.markdown("Select a flagged truck to load its latest telemetry into the prediction models.")

input_col1, input_col2 = st.columns(2)

with input_col1:
    st.subheader("🔧 Breakdown Risk Check")
    
    # Filter ONLY Truck IDs that are actively sitting in the Highway Delays pool
    banned_breakdown_trucks = sorted(highway_stalls_df["truck_id"].unique().tolist())
    
    if banned_breakdown_trucks:
        selected_truck = st.selectbox("Select a flagged truck", banned_breakdown_trucks)
        
        # Pull the absolute LATEST database row for this specific truck to auto-populate sliders
        latest_truck_log = highway_stalls_df[highway_stalls_df["truck_id"] == selected_truck].sort_values(by="timestamp", ascending=False).iloc[0]
        
        # Dynamic telemetry variables pulled directly from your PostgreSQL data layer
        default_speed = int(latest_truck_log["current_speed_kmh"])
        default_temp = float(latest_truck_log["engine_temp_c"])
        default_service = int(latest_truck_log["hours_since_service"])
        default_total_dist = float(latest_truck_log["total_route_distance_km"])
        default_covered_dist = float(latest_truck_log["distance_covered_km"])
    else:
        selected_truck = "OG-FLEET-1000"
        default_speed, default_temp, default_service, default_total_dist, default_covered_dist = 80, 54.0, 400, 200.0, 120.0
        st.selectbox("Select Target Flagged Asset for Live AI Diagnosis", [selected_truck])

    # Sliders inherit starting values from the selected truck!
    speed_input = st.slider("Vehicle Speed (km/h)", min_value=0, max_value=110, value=default_speed)
    temp_input = st.slider("Engine Temperature (°C)", min_value=40, max_value=140, value=int(default_temp))
    service_input = st.number_input("Hours Since Last Service", value=default_service)
    route_dist_input = st.number_input("Total Route Distance (km)", value=int(default_total_dist))
    covered_dist_input = st.number_input("Distance Covered (km)", value=int(default_covered_dist))
    
    if st.button("Run System Component Risk Check"):
        active_driver = assign_driver_to_truck(selected_truck)
        
        st.markdown("#### 📝 Telemetry Input Frame Sent to Model:")
        st.json({
            "Asset ID": selected_truck,
            "Assigned Driver": active_driver,
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
    st.subheader("⏱️ Trip Duration & Fuel Level Check")
    
    # Filter ONLY Truck IDs that have a confirmed Fuel Theft Anomaly flag in the database
    theft_target_trucks = sorted(theft_anomalies_df["truck_id"].unique().tolist())
    
    if theft_target_trucks:
        selected_truck_2 = st.selectbox("Select a truck with a fuel level drop", theft_target_trucks)
        
        # Pull the absolute LATEST database record for this flagged theft asset
        latest_theft_log = theft_anomalies_df[theft_anomalies_df["truck_id"] == selected_truck_2].sort_values(by="timestamp", ascending=False).iloc[0]
        
        default_route = str(latest_theft_log["assigned_route"])
        default_status = str(latest_theft_log["fleet_status"])
        default_speed_2 = int(latest_theft_log["current_speed_kmh"])
        default_temp_2 = float(latest_theft_log["engine_temp_c"])
    else:
        selected_truck_2 = "OG-FLEET-1200"
        default_route = "Lagos-Ibadan Expressway Corridor"
        default_status = "in transit (stopped/unscheduled)"
        default_speed_2, default_temp_2 = 0, 45.0
        st.selectbox("Select Flagged Security Target for Integrity Analysis", [selected_truck_2])

    # Dynamic route list setup matching case format configurations
    corridor_options = ["Lagos-Ibadan Expressway Corridor", "Port Harcourt-Aba Road Corridor", "Sagamu-Benin Express Corridor"]
    try:
        route_index = corridor_options.index(default_route)
    except ValueError:
        route_index = 0

    target_corridor = st.selectbox("Route Corridor", corridor_options, index=route_index)
    
    # Status keys are handled dynamically to account for background string variations
    status_options = ["in transit", "in transit (stopped/unscheduled)", "idle at depot", "under maintenance", "off-duty"]
    try:
        status_index = status_options.index(default_status)
    except ValueError:
        status_index = 1

    live_fleet_status = st.selectbox("Vehicle Status", [s.title() for s in status_options], index=status_index)
    
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
        if st.button("Check Fuel Level Integrity Check"):
            active_driver_2 = assign_driver_to_truck(selected_truck_2)
            
            st.markdown("#### 📝 Security State Snapshot:")
            st.json({
                "Asset ID": selected_truck_2,
                "Driver Accountable": active_driver_2,
                "Corridor Space": target_corridor,
                "Velocity Context": f"{speed_input} KM/H",
                "Reported Status": live_fleet_status
            })
            
            # Map down back to model format expectation constraints
            theft_risk = predict_fuel_theft_risk(speed_input, temp_input, target_corridor, live_fleet_status.lower())
            if theft_risk >= 0.75:
                st.error(f"🚨 HIGH FUEL ANOMALY RISK: {round(theft_risk * 100, 2)}% probability of ongoing fuel siphoning.")
            else:
                st.success(f"✅ Tank Volume Secure. Theft Anomaly Risk: {round(theft_risk * 100, 2)}%")
