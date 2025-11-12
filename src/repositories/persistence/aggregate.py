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
            num_measurements
        FROM phys_agg_{duration}s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(PhysicalAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[PhysicalAggregate] = await cur.fetchall()
                return rows

    async def get_labels_for_snapshot(snapshot_id:str, system_id:str, duration:int) -> List[str]:
        if not snapshot_id:
            raise ValueError("snapshot_id cannot be empty")
        # Example snapshot_id: testbed_system_1_30s_2021-04-09 18:58:00+00:00
        # bucket = substring after last "_"
        bucket = snapshot_id.split('_')[-1]
        logging.info(f"Fetching labels for snapshot_id: {snapshot_id}, bucket: {bucket}, system_id: {system_id}, duration: {duration}")

        pool = await DataFactory.get_pg_pool()
        query = f"""
        SELECT attack_types AS labels
        FROM scada_agg_{duration}s
        WHERE bucket = %s AND system_id = %s;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(query, (bucket, system_id))
                # There should be only one row per snapshot, returns text[] column
                # return as string array
                rows = await cur.fetchall()                
                if rows:
                    all_labels = []
                    for row in rows:
                        all_labels.extend(row['labels'])
                    # Normalize labels to "normal" and "anomaly"
                    for i in range(len(all_labels)):
                        all_labels[i] = all_labels[i] if all_labels[i] == "normal" else "anomaly"
                    all_labels = list(set(all_labels))
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
            destination_key
        FROM scada_resolved_agg_{duration}s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        AND source_key IS NOT NULL AND destination_key IS NOT NULL
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(ScadaAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[ScadaAggregate] = await cur.fetchall()
                return rows