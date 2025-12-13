import logging
from factories.data import DataFactory
from schemas.ScadaAggregate import ScadaAggregate
from schemas.PhysicalAggregate import PhysicalAggregate
from typing import List
from psycopg.rows import class_row, dict_row

class AggregateRepository:

    async def fetch_physical_aggregates(start_time, end_time, duration, system_id) -> list[PhysicalAggregate]:        
        pool = await DataFactory.get_pg_pool()
        query = f"""
        SELECT 
            bucket,
            {duration} AS duration,
            system_id,
            asset_id,
            prop_key,
            avg_value,
            min_value,
            max_value,
            stddev_value,
            num_measurements
        FROM phys_agg_{duration}s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(PhysicalAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[PhysicalAggregate] = await cur.fetchall()
                logging.info(f"Fetched {len(rows)} physical aggregates for system_id: {system_id} between {start_time} and {end_time}")
                return rows

    async def get_labels_for_snapshot(snapshot_id:str, system_id:str, duration:int) -> List[str]:
        if not snapshot_id:
            raise ValueError("snapshot_id cannot be empty")
        # Example snapshot_id: testbed_system_1_30s_2021-04-09 18:58:00+00:00
        # bucket = substring after last "_"
        bucket = snapshot_id.split('_')[-1]
        logging.info(f"Fetching labels for snapshot_id: {snapshot_id}, bucket: {bucket}, system_id: {system_id}, duration: {duration}")

        pool = await DataFactory.get_pg_pool()
        # query = f"""
        # SELECT attack_types AS labels
        # FROM scada_agg_{duration}s
        # WHERE bucket = %s AND system_id = %s
        # UNION
        # SELECT attack_types AS labels
        # FROM phys_agg_{duration}s
        # WHERE bucket = %s AND system_id = %s
        # ;
        # """
        # async with pool.connection() as conn:
        #     async with conn.cursor(row_factory=dict_row) as cur:
        #         await cur.execute(query, (bucket, system_id, bucket, system_id))
        #         # There should be only one row per snapshot, returns text[] column
        #         # return as string array
        #         rows = await cur.fetchall()                
        #         if rows:
        #             all_labels = []
        #             for row in rows:
        #                 all_labels.extend(row['labels'])
        #             logging.info(f"Fetched labels for snapshot ID: {snapshot_id}, labels: {all_labels}")
        #             # Normalize labels to "normal" and "anomaly"
        #             # for i in range(len(all_labels)):
        #             #     all_labels[i] = all_labels[i] if all_labels[i] == "normal" else "anomaly"
        #             all_labels = list(set(all_labels))
        #             return all_labels
        #         else:
        #             return []
        
        query = f"""
        SELECT attack_type AS labels
        FROM y_labels
        WHERE bucket = %s;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (bucket,))
                # There should be only one row per snapshot, returns text[] column
                # return as string array
                row = await cur.fetchone()
                if row:
                    all_labels = [row['labels']]
                    logging.info(f"Fetched labels for snapshot ID: {snapshot_id}, labels: {all_labels}")
                    return all_labels
                else:
                    return []


    async def fetch_scada_aggregates(start_time, end_time, duration, system_id) -> list[ScadaAggregate]:
        pool = await DataFactory.get_pg_pool()
        query = f"""
        SELECT 
            bucket,
            {duration} AS duration,
            system_id,
            protocol,
            avg_size,
            source_total_packets,
            destination_total_packets,
            min_size,
            max_size,
            num_connections,
            source_ip::text AS source_ip,
            source_port,
            source_mac::text AS source_mac,
            destination_ip::text AS destination_ip,
            destination_port,
            destination_mac::text AS destination_mac,
            source_key,
            destination_key,
            tcp_cwr_count,
            tcp_ece_count,
            tcp_urg_count,
            tcp_ack_count,
            tcp_psh_count,
            tcp_rst_count,
            tcp_syn_count,
            tcp_fin_count,
            tcp_syn_ratio,
            tcp_ack_ratio,
            modbus_response_count,
            modbus_response_ratio,
            COALESCE(avg_modbus_response_code, 0) AS avg_modbus_response_code,
            modbus_response_present
        FROM scada_resolved_agg_{duration}s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        AND source_key IS NOT NULL AND destination_key IS NOT NULL
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(ScadaAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[ScadaAggregate] = await cur.fetchall()
                logging.info(f"Fetched {len(rows)} SCADA aggregates for system_id: {system_id} between {start_time} and {end_time}")
                return rows
            
    async def fetch_physical_data_xgboost(system_id: str, buckets: List[str]) -> list[PhysicalAggregate]:
        
        bucket_list = ",".join([f"'{bucket}'" for bucket in buckets])
        pool = await DataFactory.get_pg_pool()
        query = f"""
        SELECT --bucket, system_id, num_measurements, num_attacks, attack_types,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_1') AS tank_1_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_1') AS Flow_sensor_1_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_2') AS Flow_sensor_2_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_3') AS Flow_sensor_3_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_4') AS Flow_sensor_4_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_1') AS Pump_1_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_2') AS Pump_2_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_3') AS Pump_3_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_4') AS Pump_4_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_5') AS Pump_5_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_6') AS Pump_6_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_2') AS Tank_2_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_3') AS Tank_3_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_4') AS Tank_4_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_5') AS Tank_5_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_6') AS Tank_6_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_7') AS Tank_7_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_8') AS Tank_8_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_1') AS Valv_1_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_2') AS Valv_2_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_3') AS Valv_3_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_4') AS Valv_4_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_5') AS Valv_5_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_6') AS Valv_6_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_7') AS Valv_7_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_8') AS Valv_8_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_9') AS Valv_9_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_10') AS Valv_10_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_11') AS Valv_11_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_12') AS Valv_12_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_13') AS Valv_13_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_14') AS Valv_14_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_15') AS Valv_15_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_16') AS Valv_16_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_17') AS Valv_17_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_18') AS Valv_18_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_19') AS Valv_19_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_20') AS Valv_20_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_21') AS Valv_21_avg,
MAX(avg_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_22') AS Valv_22_avg,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_1') AS tank_1_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_1') AS Flow_sensor_1_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_2') AS Flow_sensor_2_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_3') AS Flow_sensor_3_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_4') AS Flow_sensor_4_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_1') AS Pump_1_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_2') AS Pump_2_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_3') AS Pump_3_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_4') AS Pump_4_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_5') AS Pump_5_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_6') AS Pump_6_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_2') AS Tank_2_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_3') AS Tank_3_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_4') AS Tank_4_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_5') AS Tank_5_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_6') AS Tank_6_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_7') AS Tank_7_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_8') AS Tank_8_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_1') AS Valv_1_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_2') AS Valv_2_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_3') AS Valv_3_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_4') AS Valv_4_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_5') AS Valv_5_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_6') AS Valv_6_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_7') AS Valv_7_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_8') AS Valv_8_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_9') AS Valv_9_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_10') AS Valv_10_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_11') AS Valv_11_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_12') AS Valv_12_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_13') AS Valv_13_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_14') AS Valv_14_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_15') AS Valv_15_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_16') AS Valv_16_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_17') AS Valv_17_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_18') AS Valv_18_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_19') AS Valv_19_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_20') AS Valv_20_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_21') AS Valv_21_max,
MAX(max_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_22') AS Valv_22_max,

MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_1') AS tank_1_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_1') AS Flow_sensor_1_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_2') AS Flow_sensor_2_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_3') AS Flow_sensor_3_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_4') AS Flow_sensor_4_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_1') AS Pump_1_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_2') AS Pump_2_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_3') AS Pump_3_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_4') AS Pump_4_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_5') AS Pump_5_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_6') AS Pump_6_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_2') AS Tank_2_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_3') AS Tank_3_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_4') AS Tank_4_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_5') AS Tank_5_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_6') AS Tank_6_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_7') AS Tank_7_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_8') AS Tank_8_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_1') AS Valv_1_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_2') AS Valv_2_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_3') AS Valv_3_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_4') AS Valv_4_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_5') AS Valv_5_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_6') AS Valv_6_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_7') AS Valv_7_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_8') AS Valv_8_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_9') AS Valv_9_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_10') AS Valv_10_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_11') AS Valv_11_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_12') AS Valv_12_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_13') AS Valv_13_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_14') AS Valv_14_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_15') AS Valv_15_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_16') AS Valv_16_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_17') AS Valv_17_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_18') AS Valv_18_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_19') AS Valv_19_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_20') AS Valv_20_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_21') AS Valv_21_min,
MAX(min_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_22') AS Valv_22_min,


MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_1') AS tank_1_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_1') AS Flow_sensor_1_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_2') AS Flow_sensor_2_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_3') AS Flow_sensor_3_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Flow_sensor_Flow_sensor_4') AS Flow_sensor_4_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_1') AS Pump_1_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_2') AS Pump_2_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_3') AS Pump_3_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_4') AS Pump_4_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_5') AS Pump_5_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Pump_Pump_6') AS Pump_6_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_2') AS Tank_2_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_3') AS Tank_3_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_4') AS Tank_4_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_5') AS Tank_5_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_6') AS Tank_6_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_7') AS Tank_7_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Tank_Tank_8') AS Tank_8_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_1') AS Valv_1_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_2') AS Valv_2_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_3') AS Valv_3_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_4') AS Valv_4_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_5') AS Valv_5_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_6') AS Valv_6_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_7') AS Valv_7_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_8') AS Valv_8_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_9') AS Valv_9_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_10') AS Valv_10_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_11') AS Valv_11_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_12') AS Valv_12_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_13') AS Valv_13_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_14') AS Valv_14_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_15') AS Valv_15_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_16') AS Valv_16_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_17') AS Valv_17_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_18') AS Valv_18_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_19') AS Valv_19_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_20') AS Valv_20_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_21') AS Valv_21_stddev,
MAX(stddev_value) FILTER ( WHERE asset_id = 'testbed_system_1_Valv_Valv_22') AS Valv_22_stddev
FROM  phys_agg_30s
WHERE system_id = '%s' 
AND bucket IN ({bucket_list})
GROUP BY bucket, system_id
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(PhysicalAggregate)) as cur:
                await cur.execute(query, (system_id,))
                rows = await cur.fetchall()
                logging.info(f"Fetched {len(rows)} physical aggregates for XGBoost for system_id: {system_id} for buckets: {buckets}")
                return rows