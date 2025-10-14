from psycopg import sql, connect

def init_db():
    conn = None
    try:
        with connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
                cur.execute("""
                            DROP MATERIALIZED VIEW IF EXISTS phys_agg_30s CASCADE;
                            DROP MATERIALIZED VIEW IF EXISTS scada_agg_30s CASCADE;
                            DROP VIEW IF EXISTS scada_resolved_agg_30s CASCADE;
                            DROP MATERIALIZED VIEW IF EXISTS phys_agg_16s CASCADE;
                            DROP MATERIALIZED VIEW IF EXISTS scada_agg_16s CASCADE;
                            DROP VIEW IF EXISTS scada_resolved_agg_16s CASCADE;
                            DROP MATERIALIZED VIEW IF EXISTS phys_agg_10s CASCADE;
                            DROP MATERIALIZED VIEW IF EXISTS scada_agg_10s CASCADE;
                            DROP VIEW IF EXISTS scada_resolved_agg_10s CASCADE;
                            DROP TABLE IF EXISTS phys_raw CASCADE;
                            DROP TABLE IF EXISTS scada_raw CASCADE;
                            DROP TABLE IF EXISTS scada_staging CASCADE;
                            DROP TABLE IF EXISTS phys_measurements_metadata CASCADE;
                            DROP TABLE IF EXISTS network_endpoints CASCADE;
                            DROP TABLE IF EXISTS assets CASCADE;                            
                            """)
                # Create table if it doesn't exist                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS assets (
                        asset_id text PRIMARY KEY, -- system_id + asset_type + asset_name
                        system_id TEXT NOT NULL,
                        asset_type TEXT NOT NULL, -- pump, sensor, valve, tank, PLC, HMI, ROUTER, etc
                        asset_name TEXT NOT NULL,
                        validated BOOLEAN NOT NULL DEFAULT FALSE,
                        first_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                """)
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS network_endpoints (
                        endpoint_id text PRIMARY KEY, -- asset_id + ip + mac  
                        system_id TEXT NOT NULL,                      
                        ip INET NULL,     
                        cidr SMALLINT NULL,       
                        mac MACADDR NULL,                        
                        asset_id TEXT NOT NULL REFERENCES ASSETS(asset_id) ON DELETE CASCADE,
                        first_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                        validated BOOLEAN NOT NULL DEFAULT FALSE,
                        check (ip IS NOT NULL OR mac IS NOT NULL)
                    );
                """)                
                
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS phys_measurements_metadata (
                        measurement_id text PRIMARY KEY, -- asset_id + name
                        asset_id TEXT NOT NULL REFERENCES ASSETS(asset_id) ON DELETE CASCADE,
                        name TEXT NOT NULL,
                        prop_key TEXT NOT NULL,
                        first_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                        last_seen_ts TIMESTAMPTZ NOT NULL DEFAULT now()                        
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS scada_raw (
                        id text NOT NULL, --source_id + source_mac + destination_id + destination_mac + log_ts
                        system_id TEXT NOT NULL,
                        log_ts TIMESTAMPTZ NOT NULL,
                        source_ip INET,
                        source_port INTEGER,
                        source_mac MACADDR NULL,
                        destination_ip INET NULL,
                        destination_port INTEGER,
                        destination_mac MACADDR NULL,
                        protocol TEXT NOT NULL,
                        modbus_func TEXT NULL,
                        source_number_packets INTEGER DEFAULT 0,
                        destination_number_packets INTEGER DEFAULT 0,
                        total_size INTEGER DEFAULT 0,
                        attributes JSONB,
                            check (source_ip IS NOT NULL OR source_mac IS NOT NULL),
                            check (destination_ip IS NOT NULL OR destination_mac IS NOT NULL),
                            check (source_port BETWEEN 0 AND 65535 OR source_port IS NULL),
                            check (destination_port BETWEEN 0 AND 65535 OR destination_port IS NULL),
                            PRIMARY KEY (id, log_ts)

                    );
                """)
                cur.execute("""
                    CREATE UNLOGGED TABLE IF NOT EXISTS scada_staging (
                        id text NOT NULL, --source_id + source_mac + destination_id + destination_mac + log_ts
                        system_id TEXT NOT NULL,
                        log_ts TIMESTAMPTZ NOT NULL,
                        source_ip INET,
                        source_port INTEGER,
                        source_mac MACADDR NULL,
                        destination_ip INET NULL,
                        destination_port INTEGER,
                        destination_mac MACADDR NULL,
                        protocol TEXT NOT NULL,
                        modbus_func TEXT NULL,
                        source_number_packets INTEGER DEFAULT 0,
                        destination_number_packets INTEGER DEFAULT 0,
                        total_size INTEGER DEFAULT 0,
                        attributes JSONB
                    );
                """)

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS phys_raw (
                        id text NOT NULL, -- system_id + time + name
                        system_id TEXT NOT NULL,
                        log_ts TIMESTAMPTZ NOT NULL,
                        measurement_id TEXT NOT NULL REFERENCES phys_measurements_metadata(measurement_id) ON DELETE CASCADE,
                        measure_value DOUBLE PRECISION NOT NULL,
                        attributes JSONB,
                        PRIMARY KEY (id, log_ts)
                    );
                """)

                
                # Create hypertables if they don't exist
                cur.execute(sql.SQL("SELECT create_hypertable('phys_raw', 'log_ts', if_not_exists => TRUE);"))
                cur.execute(sql.SQL("SELECT create_hypertable('scada_raw', 'log_ts', if_not_exists => TRUE);"))                

                # Create 30-second aggregate views
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS phys_agg_30s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('30 seconds', r.log_ts) AS bucket,
                                r.system_id,
                                m.prop_key,
                                m.asset_id,
                                AVG(r.measure_value) AS avg_value,
                                MAX(r.measure_value) AS max_value,
                                MIN(r.measure_value) AS min_value,
                                COUNT(*) AS num_measurements,
                            COALESCE(SUM(NULLIF(r.attributes->>'Label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT r.attributes->>'Label') FILTER (WHERE r.attributes ? 'Label') AS attack_types
                            FROM phys_raw r
                            JOIN phys_measurements_metadata m ON r.measurement_id = m.measurement_id
                            GROUP BY bucket, r.system_id, m.prop_key, m.asset_id
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_agg_30s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('30 seconds', log_ts) AS bucket,
                                system_id,
                                source_ip,
                                source_port,
                                source_mac,
                                destination_ip,
                                destination_port,
                                destination_mac,
                                protocol,
                                AVG(total_size) AS avg_size,
                                SUM(source_number_packets) AS source_total_packets,
                                SUM(destination_number_packets) AS destination_total_packets,
                                MIN(total_size) AS min_size,
                                MAX(total_size) AS max_size,
                                COUNT(*) AS num_connections,                            
                            COALESCE(SUM(NULLIF(attributes->>'label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT attributes->>'label') FILTER (WHERE attributes ? 'label') AS attack_types,
                                COALESCE(host(source_ip), lower(source_mac::text)) AS source_key,
                                COALESCE(host(destination_ip), lower(destination_mac::text)) AS destination_key
                            FROM scada_raw
                            GROUP BY bucket, system_id, source_ip, source_port, source_mac, destination_ip, destination_port, destination_mac, protocol
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_resolved_agg_30s AS
                            SELECT
                                s.bucket,
                                s.system_id,
                                s.protocol,
                                s.avg_size,
                                s.source_total_packets,
                                s.destination_total_packets,
                                s.min_size,
                                s.max_size,
                                s.num_connections,
                                s.source_ip,
                                s.source_port,
                                s.source_mac,
                                s.destination_ip,
                                s.destination_port,
                                s.destination_mac,
                                s.source_key,
                                s.destination_key,
                            s.num_attacks,
                            s.attack_types,
                                src.asset_id AS source_asset,
                                dst.asset_id AS destination_asset
                            FROM scada_agg_30s s
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.source_ip IS NOT NULL AND ip = s.source_ip)
                                   OR (s.source_mac IS NOT NULL AND mac = s.source_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS src ON TRUE
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.destination_ip IS NOT NULL AND ip = s.destination_ip)
                                   OR (s.destination_mac IS NOT NULL AND mac = s.destination_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS dst ON TRUE;
                            """)
                
                # Create 16 second aggregate views
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS phys_agg_16s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('16 seconds', r.log_ts) AS bucket,
                                r.system_id,
                                m.prop_key,
                                m.asset_id,
                                AVG(r.measure_value) AS avg_value,
                                MAX(r.measure_value) AS max_value,
                                MIN(r.measure_value) AS min_value,
                                COUNT(*) AS num_measurements,
                            COALESCE(SUM(NULLIF(r.attributes->>'Label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT r.attributes->>'Label') FILTER (WHERE r.attributes ? 'Label') AS attack_types
                            FROM phys_raw r
                            JOIN phys_measurements_metadata m ON r.measurement_id = m.measurement_id
                            GROUP BY bucket, r.system_id, m.prop_key, m.asset_id
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_agg_16s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('16 seconds', log_ts) AS bucket,
                                system_id,
                                source_ip,
                                source_port,
                                source_mac,
                                destination_ip,
                                destination_port,
                                destination_mac,
                                protocol,
                                AVG(total_size) AS avg_size,
                                SUM(source_number_packets) AS source_total_packets,
                                SUM(destination_number_packets) AS destination_total_packets,
                                MIN(total_size) AS min_size,
                                MAX(total_size) AS max_size,
                                COUNT(*) AS num_connections,                            
                            COALESCE(SUM(NULLIF(attributes->>'label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT attributes->>'label') FILTER (WHERE attributes ? 'label') AS attack_types,
                                COALESCE(host(source_ip), lower(source_mac::text)) AS source_key,
                                COALESCE(host(destination_ip), lower(destination_mac::text)) AS destination_key
                            FROM scada_raw
                            GROUP BY bucket, system_id, source_ip, source_port, source_mac, destination_ip, destination_port, destination_mac, protocol
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_resolved_agg_16s AS
                            SELECT
                                s.bucket,
                                s.system_id,
                                s.protocol,
                                s.avg_size,
                                s.source_total_packets,
                                s.destination_total_packets,
                                s.min_size,
                                s.max_size,
                                s.num_connections,
                                s.source_ip,
                                s.source_port,
                                s.source_mac,
                                s.destination_ip,
                                s.destination_port,
                                s.destination_mac,
                                s.source_key,
                                s.destination_key,
                            s.num_attacks,
                            s.attack_types,
                                src.asset_id AS source_asset,
                                dst.asset_id AS destination_asset
                            FROM scada_agg_16s s
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.source_ip IS NOT NULL AND ip = s.source_ip)
                                   OR (s.source_mac IS NOT NULL AND mac = s.source_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS src ON TRUE
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.destination_ip IS NOT NULL AND ip = s.destination_ip)
                                   OR (s.destination_mac IS NOT NULL AND mac = s.destination_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS dst ON TRUE;
                            """)


                # Create 10 second aggregate views
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS phys_agg_10s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('10 seconds', r.log_ts) AS bucket,
                                r.system_id,
                                m.prop_key,
                                m.asset_id,
                                AVG(r.measure_value) AS avg_value,
                                MAX(r.measure_value) AS max_value,
                                MIN(r.measure_value) AS min_value,
                                COUNT(*) AS num_measurements,                            
                            COALESCE(SUM(NULLIF(r.attributes->>'Label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT r.attributes->>'Label') FILTER (WHERE r.attributes ? 'Label') AS attack_types
                            FROM phys_raw r
                            JOIN phys_measurements_metadata m ON r.measurement_id = m.measurement_id
                            GROUP BY bucket, r.system_id, m.prop_key, m.asset_id
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_agg_10s 
                            WITH (timescaledb.continuous) AS
                            SELECT
                                time_bucket('10 seconds', log_ts) AS bucket,
                                system_id,
                                source_ip,
                                source_port,
                                source_mac,
                                destination_ip,
                                destination_port,
                                destination_mac,
                                protocol,
                                AVG(total_size) AS avg_size,
                                SUM(source_number_packets) AS source_total_packets,
                                SUM(destination_number_packets) AS destination_total_packets,
                                MIN(total_size) AS min_size,
                                MAX(total_size) AS max_size,
                                COUNT(*) AS num_connections,                            
                            COALESCE(SUM(NULLIF(attributes->>'label_n', '0')::INT), 0) AS num_attacks,
                            array_agg(DISTINCT attributes->>'label') FILTER (WHERE attributes ? 'label') AS attack_types,
                                COALESCE(host(source_ip), lower(source_mac::text)) AS source_key,
                                COALESCE(host(destination_ip), lower(destination_mac::text)) AS destination_key
                            FROM scada_raw
                            GROUP BY bucket, system_id, source_ip, source_port, source_mac, destination_ip, destination_port, destination_mac, protocol
                            WITH NO DATA;
                            """)
                
                cur.execute("""
                            CREATE MATERIALIZED VIEW IF NOT EXISTS scada_resolved_agg_10s AS
                            SELECT
                                s.bucket,
                                s.system_id,
                                s.protocol,
                                s.avg_size,
                                s.source_total_packets,
                                s.destination_total_packets,
                                s.min_size,
                                s.max_size,
                                s.num_connections,
                                s.source_ip,
                                s.source_port,
                                s.source_mac,
                                s.destination_ip,
                                s.destination_port,
                                s.destination_mac,
                                s.source_key,
                                s.destination_key,
                                s.num_attacks,
                                s.attack_types,
                                src.asset_id AS source_asset,
                                dst.asset_id AS destination_asset
                            FROM scada_agg_10s s
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.source_ip IS NOT NULL AND ip = s.source_ip)
                                   OR (s.source_mac IS NOT NULL AND mac = s.source_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS src ON TRUE
                            LEFT JOIN LATERAL (
                                SELECT asset_id
                                FROM network_endpoints
                                WHERE (s.destination_ip IS NOT NULL AND ip = s.destination_ip)
                                   OR (s.destination_mac IS NOT NULL AND mac = s.destination_mac)
                                ORDER BY (ip IS NOT NULL) DESC, (mac IS NOT NULL) DESC
                                LIMIT 1
                            ) AS dst ON TRUE;
                            """)


                print("Database tables and views initialized successfully.")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":    
    init_db()