import joblib
import numpy as np
import pandas as pd

# Load compiled model artifacts globally for processing efficiency
try:
    model_breakdown = joblib.load("maintenance_model.pkl")
    model_eta = joblib.load("eta_model.pkl")
    model_theft = joblib.load("fuel_theft_model.pkl")
except Exception as e:
    print(f"Artifact Loading Warning: Ensure .pkl models exist in the root folder. Error: {e}")

def assign_driver_to_truck(truck_id):
    """Dynamically maps a permanent, realistic driver name to a truck ID for accountability tracking."""
    first_names = ["Emeka", "Babajide", "Musa", "Chidi", "Tunde", "Ibrahim", "Abiodun", "Nnamdi", "Segun", "Umar"]
    last_names = ["Okonkwo", "Balogun", "Bello", "Adeleke", "Aliyu", "Eke", "Oyinlola", "Abubakar", "Nwachukwu", "Fagbemi"]
    
    try:
        # Extract the numerical part of the truck ID (e.g., 1145 from OG-FLEET-1145) to ensure consistency
        truck_num = int(str(truck_id).split("-")[-1])
    except:
        truck_num = 42 # Fallback seed
        
    # Use deterministic index loops so the same truck always belongs to the same driver
    first = first_names[truck_num % len(first_names)]
    last = last_names[(truck_num + 3) % len(last_names)]
    return f"{first} {last}"

def predict_breakdown_risk(speed, temp, service_hours, total_dist, covered_dist):
    """Calculates mechanical failure risk probability percentage for Class 1."""
    features = np.array([[speed, temp, service_hours, total_dist, covered_dist]])
    prob = model_breakdown.predict_proba(features)
    return float(prob[0][1])  # <--- FIX: Extract Class 1 probability scalar

def predict_transit_duration(total_dist, speed, route_name):
    """Forecasts dynamic corridor trip arrival window durations."""
    route_aba_flag = 1 if route_name == "Port Harcourt-Aba Road Corridor" else 0
    route_benin_flag = 1 if route_name == "Sagamu-Benin Express Corridor" else 0
    
    features = np.array([[total_dist, speed, route_aba_flag, route_benin_flag]])
    duration = model_eta.predict(features)
    return float(duration[0])  # <--- FIX: Ensure scalar conversion

def predict_fuel_theft_risk(speed, temp, route_name, fleet_status):
    """Exposes real-time downstream siphoning anomalies by perfectly aligning feature columns."""
    route_aba_flag = 1 if route_name == "Port Harcourt-Aba Road Corridor" else 0
    route_benin_flag = 1 if route_name == "Sagamu-Benin Express Corridor" else 0
    
    # Reconstruct the EXACT 8 feature columns your model was trained on, in the EXACT same order
    mock_row = {
        "current_speed_kmh": speed,
        "engine_temp_c": temp,
        "assigned_route_Port Harcourt-Aba Road Corridor": route_aba_flag,
        "assigned_route_Sagamu-Benin Express Corridor": route_benin_flag,
        "fleet_status_In Transit": 1 if fleet_status == "In Transit" else 0,
        "fleet_status_In Transit (Stopped/Unscheduled)": 1 if fleet_status == "In Transit (Stopped/Unscheduled)" else 0,
        "fleet_status_Off-Duty": 1 if fleet_status == "Off-Duty" else 0,
        "fleet_status_Under Maintenance": 1 if fleet_status == "Under Maintenance" else 0
    }
    
    mock_df = pd.DataFrame([mock_row])
    
    mock_df = mock_df[[
        "current_speed_kmh", "engine_temp_c", 
        "assigned_route_Port Harcourt-Aba Road Corridor", 
        "assigned_route_Sagamu-Benin Express Corridor",
        "fleet_status_In Transit", 
        "fleet_status_In Transit (Stopped/Unscheduled)", 
        "fleet_status_Off-Duty", 
        "fleet_status_Under Maintenance"
    ]]
    
    prob = model_theft.predict_proba(mock_df)
    return float(prob[0][1])  # <--- FIX: Extract Class 1 probability scalar

# =====================================================================
# LIVE INFERENCE TERMINAL VERIFICATION TRIGGER
# =====================================================================
if __name__ == "__main__":
    print("🧪 Running verification checks on underlying ML models...")
    try:
        # Test Model A (Breakdown)
        breakdown_risk = predict_breakdown_risk(65, 115, 380, 130.0, 85.0)
        print(f"-> Model A (Breakdown Risk): {round(breakdown_risk * 100, 2)}%")
        
        # Test Model B (ETA Regressor)
        eta_hours = predict_transit_duration(130.0, 65, "Lagos-Ibadan Expressway Corridor")
        print(f"-> Model B (Transit Duration): {round(eta_hours, 2)} Hours")
        
        # Test Model C (Fuel Theft)
        theft_risk = predict_fuel_theft_risk(0, 45.0, "Lagos-Ibadan Expressway Corridor", "In Transit (Stopped/Unscheduled)")
        print(f"-> Model C (Fuel Theft Risk): {round(theft_risk * 100, 2)}%")
        print("\n✅ Inference logic working seamlessly across all estimators!")
    except Exception as e:
        print(f"❌ Verification failed. Check feature structures. Details: {e}")
