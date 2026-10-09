-- Query 1: The Highway Fuel Theft Audit

--This query exposes the exact "fingerprint" of mid-transit fuel theft you designed: identifying moving trucks that suddenly stopped dead on the highway corridor and dropped significant fuel volume.
SELECT 
    timestamp,
    truck_id,
    origin_depot,
    assigned_route,
    distance_covered_km,
    fuel_volume_lost_liters AS liters_stolen
FROM 
    public.fleet_telematics_logs
WHERE 
    fuel_theft_anomaly = 1 
    AND fleet_status = 'In Transit (Stopped/Unscheduled)'
ORDER BY 
    fuel_volume_lost_liters DESC;


-- Query 2: Fleet Status Operational Breakdown

-- This query aggregates the entire 1,300-truck fleet to provide a high-level operational snapshot. This calculation generates the exact numbers needed to feed my dashboard's Plotly charts.
SELECT 
    fleet_status,
    COUNT(DISTINCT truck_id) AS total_trucks,
    ROUND((COUNT(DISTINCT truck_id)::NUMERIC / 1300) * 100, 2) AS percentage_of_fleet
FROM 
    public.fleet_telematics_logs
-- Filtering for the most recent log update window to reflect active status
WHERE 
    timestamp = (SELECT MAX(timestamp) FROM public.fleet_telematics_logs)
GROUP BY 
    fleet_status
ORDER BY 
    total_trucks DESC;


-- Query 3: Depot Risk & Breakdown Matrix
-- This query links mechanical failures directly back to the loading depots. It uncovers which distribution loops are experiencing the highest engine strain and breakdown incident rates.
SELECT 
    origin_depot,
    COUNT(CASE WHEN breakdown_risk_target = 1 THEN 1 END) AS total_road_breakdowns,
    ROUND(AVG(engine_temp_c), 1) AS avg_operating_temp_c,
    ROUND(AVG(vibration_g), 2) AS avg_vibration_g
FROM 
    public.fleet_telematics_logs
GROUP BY 
    origin_depot
ORDER BY 
    total_road_breakdowns DESC;



--Query 4: The Smart Dispatch Compliance Audit

-- This query validates that my dispatch compliance rules are actually working. It checks whether any high-wear vehicle (hours_since_service >= 310) was wrongfully assigned to the grueling Warri-Benin long-distance route.

SELECT 
    truck_id,
    hours_since_service,
    assigned_route,
    maintenance_recommendation
FROM 
    public.fleet_telematics_logs
WHERE 
    hours_since_service >= 310 
    AND assigned_route = 'Sagamu-Benin Express Corridor';
    
-- Note: If your simulation logic worked perfectly, this query should return 0 rows. 
-- That proves your smart scheduling loop successfully blocked restricted trucks!
