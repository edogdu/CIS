import psycopg2
from psycopg2 import sql, connect
import os
import time

def load_testbed_metadata():
    conn = None
    system_id = "testbed_system_1"
    network_assets = [{
            "asset_type": "PLC",
            "asset_name": "PLC-1",
            "ip": "84.3.251.18/17"
        },{
            "asset_type": "PLC",
            "asset_name": "PLC_2",
            "ip": "84.3.251.101/17"
        },{
            "asset_type": "PLC",
            "asset_name": "PLC_3",
            "ip": "84.3.251.102/17"
        },{
            "asset_type": "PLC",
            "asset_name": "PLC_4",
            "ip": "84.3.251.103/17"
        },{
            "asset_type": "HMI",
            "asset_name": "HMI_1",
            "ip": "84.3.251.20/17"
        },{
            "asset_type": "Field Instrument Controller",
            "asset_name": "FIC_1",
            "ip": "84.3.251.104/17"
        },{
            "asset_type": "Field Instrument Controller",
            "asset_name": "FIC_2",
            "ip": "84.3.251.105/17"
        }] # List of tuples (asset_id, system_id, asset_type, asset_name)
    
    physical_assets = []
    
    for i in range(1, 9):
        
        physical_assets.append({
        "asset_type": "Tank",
        "asset_name": f"Tank_{i}",
        "measurements": [{
            "name": "Pressure",
            "prop_key": "pressure"
        }]
        })
    for i in range(1, 7):
        physical_assets.append({
        "asset_type": "Pump",
        "asset_name": f"Pump_{i}",
        "measurements": [{
            "name": "state",
            "prop_key": "state"
        }]
        })
    for i in range(1, 5):
        physical_assets.append({
        "asset_type": "Flow Sensor",
        "asset_name": f"Flow_sensor_{i}",
        "measurements": [{
            "name": "Flow Sensor Value",
            "prop_key": "value"
        }]
        })
    for i in range(1, 23):
        physical_assets.append({
        "asset_type": "Valve",
        "asset_name": f"Valve_{i}",
        "measurements": [{
            "name": "state",
            "prop_key": "state"
        }]
        })   
    
    try:
        with connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                print("Loading testbed metadata into the database...")
                for asset in network_assets:
                    asset_id = f"{system_id}_{asset['asset_type']}_{asset['asset_name']}"
                    cur.execute("""
                        INSERT INTO assets (asset_id, system_id, asset_type, asset_name, validated)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (asset_id) DO UPDATE 
                        SET last_seen_ts = now(), validated = TRUE
                    """, (asset_id, system_id, asset['asset_type'], asset['asset_name']))
                    cur.execute("""
                        INSERT INTO network_endpoints (endpoint_id, system_id, ip, asset_id, validated)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (endpoint_id) DO UPDATE 
                        SET last_seen_ts = now(), validated = TRUE
                    """, (f"{asset_id}_{asset['ip']}", system_id, asset['ip'], asset_id))

                for asset in physical_assets:
                    asset_id = f"{system_id}_{asset['asset_type']}_{asset['asset_name']}"
                    cur.execute("""
                        INSERT INTO assets (asset_id, system_id, asset_type, asset_name, validated)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (asset_id) DO UPDATE 
                        SET last_seen_ts = now(), validated = TRUE
                    """, (asset_id, system_id, asset['asset_type'], asset['asset_name']))
                    for measurement in asset['measurements']:
                        measurement_id = f"{asset_id}_{measurement['name']}"
                        cur.execute("""
                            INSERT INTO phys_measurements_metadata (measurement_id, asset_id, name, prop_key)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (measurement_id) DO UPDATE 
                            SET last_seen_ts = now()
                        """, (measurement_id, asset_id, measurement['name'], measurement['prop_key']))
                
                
    except Exception as e:
        print(f"Error loading testbed metadata: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    load_testbed_metadata()
    try:
        with connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM assets;")
                asset_count = cur.fetchone()[0]
                print(f"Total assets in database: {asset_count}")
                
                cur.execute("SELECT COUNT(*) FROM phys_measurements_metadata;")
                measurement_count = cur.fetchone()[0]
                print(f"Total physical measurements in database: {measurement_count}")

                cur.execute("SELECT * FROM network_endpoints;")
                print("Network Endpoints:")
                for row in cur.fetchall():
                    print(row)
                
                cur.execute("SELECT * FROM assets;")
                print("Assets:")
                for row in cur.fetchall():
                    print(row)
    finally:
        if conn:
            conn.close()