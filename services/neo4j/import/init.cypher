// -------- CONSTRAINTS --------
CREATE CONSTRAINT asset_id IF NOT EXISTS FOR (a:Asset) REQUIRE a.asset_id IS UNIQUE;
CREATE CONSTRAINT endpoint_ip IF NOT EXISTS FOR (e:Endpoint) REQUIRE e.ip IS UNIQUE;
CREATE CONSTRAINT meas_id IF NOT EXISTS FOR (m:Measurement) REQUIRE m.measurement_id IS UNIQUE;
CREATE CONSTRAINT kind_name IF NOT EXISTS FOR (k:MeasurementKind) REQUIRE k.name IS UNIQUE;



// Ensure Reservoir exists
MERGE (:Asset:Reservoir {
  asset_id:'testbed_system_1_Physical_RESERVOIR',
  system_id:'testbed_system_1',
  asset_name:'RESERVOIR',
  asset_type:'Reservoir'
});

// Ensure PressureSensors exists
MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_1',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_1',
  asset_type:'PressureSensor'
});
MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_2',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_2',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_3',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_3',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_4',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_4',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_5',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_5',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_6',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_6',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_7',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_7',
  asset_type:'PressureSensor'
});

MERGE (:Asset:PressureSensor {
  asset_id:'testbed_system_1_Physical_Pressure_sensor_8',
  system_id:'testbed_system_1',
  asset_name:'Pressure_sensor_8',
  asset_type:'PressureSensor'
});

// -------- ASSETS (assets.csv) --------
LOAD CSV WITH HEADERS FROM 'file:///assets.csv' AS row
WITH row WHERE row.system_id = 'testbed_system_1'
MERGE (a:Asset {asset_id: row.asset_id})
  ON CREATE SET a.system_id=row.system_id, a.asset_name=row.asset_name, a.asset_type=row.asset_type
  ON MATCH  SET a.asset_name=row.asset_name, a.asset_type=row.asset_type
WITH a, toLower(a.asset_type) AS t
FOREACH (_ IN CASE WHEN t CONTAINS 'plc' THEN [1] ELSE [] END | SET a:PLC)
FOREACH (_ IN CASE WHEN t CONTAINS 'hmi' THEN [1] ELSE [] END | SET a:HMI)
FOREACH (_ IN CASE WHEN t CONTAINS 'tank' THEN [1] ELSE [] END | SET a:Tank)
FOREACH (_ IN CASE WHEN t CONTAINS 'pump' THEN [1] ELSE [] END | SET a:Pump)
FOREACH (_ IN CASE WHEN t CONTAINS 'valve' THEN [1] ELSE [] END | SET a:Valve)
FOREACH (_ IN CASE WHEN t CONTAINS 'flow sensor' THEN [1] ELSE [] END | SET a:FlowSensor);

// -------- ENDPOINTS (endpoints.csv) --------
LOAD CSV WITH HEADERS FROM 'file:///endpoints.csv' AS row
WITH row WHERE row.system_id='testbed_system_1' AND row.ip IS NOT NULL AND row.ip <> ''
MERGE (a:Asset {asset_id: row.asset_id, system_id: row.system_id})
MERGE (e:Endpoint {ip: row.ip, key: row.ip, system_id: row.system_id})
  ON CREATE SET e.cidr = CASE WHEN row.ip CONTAINS '/' THEN toInteger(split(row.ip,'/')[1]) ELSE null END
MERGE (a)-[:HAS_ENDPOINT]->(e);

// -------- RELATIONS / TOPOLOGY (relations.csv) --------
/* IN_STAGE */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='IN_STAGE'
MATCH (a:Asset {asset_name: row.a})
SET a.stage = row.b;

/* FEEDS_TO */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='FEEDS_TO'
MATCH (u:Asset {asset_name: row.a})
MATCH (v:Asset {asset_name: row.b})
MERGE (u)-[:FEEDS_TO]->(v);

/* FEEDS_THROUGH */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='FEEDS_THROUGH'
MATCH (u:Asset {asset_name: row.a})
MATCH (v:Asset {asset_name: row.b})
MERGE (u)-[:FEEDS_THROUGH]->(v);

/* READS_FROM */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='READS_FROM'
MATCH (u:Asset {asset_name: row.a})
MATCH (v:Asset {asset_name: row.b})
MERGE (u)-[:READS_FROM]->(v);

/* SENSOR_ON */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='SENSOR_ON'
MATCH (ps:PressureSensor {asset_name: row.a})
MATCH (up:Asset {asset_name: row.b})
MERGE (ps)-[:SENSOR_ON]->(up);

/* CONTROLS */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='CONTROLS'
MATCH (u:Asset {asset_name: row.a})
MATCH (v:Asset {asset_name: row.b})
MERGE (u)-[:CONTROLS]->(v);


/* ISSUES_COMMAND_TO */
LOAD CSV WITH HEADERS FROM 'file:///relations.csv' AS row
WITH row WHERE row.type='ISSUES_COMMAND_TO'
MATCH (u:Asset {asset_name: row.a})
MATCH (v:Asset {asset_name: row.b})
MERGE (u)-[:ISSUES_COMMAND_TO]->(v);

// -------- sanity checks --------
MATCH (a:Asset) RETURN count(a) AS assets;
MATCH (e:Endpoint) RETURN count(e) AS endpoints;
MATCH (k:MeasurementKind) RETURN collect(k.name) AS kinds;
