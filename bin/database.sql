-- 1. Create the Core Telematics Table
CREATE TABLE public.fleet_telematics_logs (
    log_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    truck_id VARCHAR(20) NOT NULL,
    fleet_status VARCHAR(50) NOT NULL,
    origin_depot VARCHAR(100) NOT NULL,
    assigned_route VARCHAR(100) NOT NULL,
    total_route_distance_km NUMERIC(6,2) NOT NULL,
    distance_covered_km NUMERIC(6,2) NOT NULL,
    current_speed_kmh INT NOT NULL,
    fuel_volume_lost_liters NUMERIC(6,1) NOT NULL,
    fuel_theft_anomaly INT NOT NULL CHECK (fuel_theft_anomaly IN (0, 1)),
    engine_temp_c NUMERIC(4,1) NOT NULL,
    hours_since_service INT NOT NULL,
    maintenance_recommendation VARCHAR(100) NOT NULL,
    vibration_g NUMERIC(4,2) NOT NULL,
    breakdown_risk_target INT NOT NULL CHECK (breakdown_risk_target IN (0, 1)),
    actual_trip_duration_hours NUMERIC(5,2) NOT NULL
);

-- 2. Create Indexes to speed up future analytics queries
CREATE INDEX idx_truck_id ON public.fleet_telematics_logs(truck_id);
CREATE INDEX idx_fleet_status ON public.fleet_telematics_logs(fleet_status);
CREATE INDEX idx_fuel_anomaly ON public.fleet_telematics_logs(fuel_theft_anomaly) WHERE fuel_theft_anomaly = 1;
CREATE INDEX idx_breakdown_target ON public.fleet_telematics_logs(breakdown_risk_target) WHERE breakdown_risk_target = 1;

COMMENT ON TABLE public.fleet_telematics_logs IS 'Stores streaming telematics data tracking down-stream fuel fleet anomalies and asset risks.';
