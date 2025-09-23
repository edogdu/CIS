from factories.data import DataFactory
from schemas.ScadaAggregate import ScadaAggregate
from schemas.PhysicalAggregate import PhysicalAggregate
from typing import List
from psycopg.rows import class_row, dict_row

class AggregateRepository:

    async def fetch_physical_aggregates(start_time, end_time, system_id) -> list[PhysicalAggregate]:
        return []
        pool = await DataFactory.get_pg_pool()
        query = """
        SELECT 
            bucket,
            30 AS duration,
            system_id,
            asset_id,
            prop_key,
            avg_value,
            min_value,
            max_value,
            num_measurements
        FROM physical_resolved_agg_30s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(PhysicalAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[PhysicalAggregate] = await cur.fetchall()
                return rows

    async def fetch_scada_aggregates(start_time, end_time, system_id) -> list[ScadaAggregate]:
        pool = await DataFactory.get_pg_pool()
        query = """
        SELECT 
            bucket,
            30 AS duration,
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
        FROM scada_resolved_agg_30s
        WHERE bucket >= %s AND bucket < %s AND system_id = %s
        AND source_key IS NOT NULL AND destination_key IS NOT NULL
        ORDER BY bucket;
        """
        async with pool.connection() as conn:
            async with conn.cursor(row_factory=class_row(ScadaAggregate)) as cur:
                await cur.execute(query, (start_time, end_time, system_id))
                rows: list[ScadaAggregate] = await cur.fetchall()
                return rows