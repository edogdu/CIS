from factories.data import DataFactory
from schemas.GenerateGraphRequest import GenerateGraphRequest
from schemas.ScadaAggregate import ScadaAggregate
from schemas.PhysicalAggregate import PhysicalAggregate
import datetime
class SnapshotRepository:
        
    async def create_snapshot(request: GenerateGraphRequest, bucket) -> str:
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            snapshot_id = f"{request.system_id}_{bucket}"
            await session.run(
                """
                MERGE (s:Snapshot {id: $snapshot_id})
                SET s.system_id = $system_id,
                    s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.created_at = datetime()
                """,
                snapshot_id=snapshot_id,
                system_id=request.system_id,
                start_time=bucket,
                end_time=bucket + datetime.timedelta(seconds=30)
            )
            return snapshot_id

    async def add_physical_data_to_snapshot(snapshot_id: str, physical_data: list[PhysicalAggregate]):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            for record in physical_data:
                await session.run(
                    """                    
                    MATCH (a:Asset {asset_id: $asset_id})
                    MERGE (m:Measurement {
                        start_time: datetime($bucket),
                        system_id: a.system_id,
                        asset_id: a.asset_id,
                        snapshot_id: $snapshot_id,
                        measurement_type: $prop_key,
                        duration: $duration                        
                    })
                    ON CREATE SET m.avg_value = $avg_value,
                        m.min_value = $min_value,
                        m.max_value = $max_value,
                        m.num_measurements = $num_measurements
                    MERGE (a)-[:HAS_MEASUREMENT]->(m)
                    """,
                    snapshot_id=snapshot_id,
                    asset_id=record.asset_id,
                    prop_key=record.prop_key,
                    bucket=record.bucket,
                    duration=record.duration,
                    avg_value=record.avg_value,
                    min_value=record.min_value,
                    max_value=record.max_value,
                    num_measurements=record.num_measurements
                )
    async def add_scada_data_to_snapshot(snapshot_id: str, scada_data: list[ScadaAggregate]):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            for record in scada_data:
                await session.run(
                    """
                    MATCH (s:Snapshot {id: $snapshot_id})
                    MATCH (src:Endpoint {key: $source_key})
                    MATCH (dst:Endpoint {key: $destination_key})
                    MERGE (conn:Connection {
                        start_time: datetime($bucket),
                        duration: $duration,
                        protocol: $protocol,
                        avg_size: $avg_size,
                        source_total_packets: $source_total_packets,
                        destination_total_packets: $destination_total_packets,
                        min_size: $min_size,
                        max_size: $max_size,
                        num_connections: $num_connections,
                        source: $source_key,
                        destination: $destination_key,
                        snapshot_id: $snapshot_id
                    })
                    ON CREATE SET conn.source_port = $source_port,
                        conn.destination_port = $destination_port,
                        conn.system_id = src.system_id
                    MERGE (src)-[:INITIATES]->(conn)
                    MERGE (conn)-[:TERMINATES_AT]->(dst)
                    """,
                    snapshot_id=snapshot_id,
                    source_ip=(record.source_ip or None),
                    source_mac=(record.source_mac or None),
                    destination_ip=(record.destination_ip or None),
                    destination_mac=(record.destination_mac or None),
                    bucket=record.bucket,
                    duration=record.duration,
                    protocol=record.protocol,
                    avg_size=record.avg_size,
                    source_total_packets=record.source_total_packets,
                    destination_total_packets=record.destination_total_packets,
                    min_size=record.min_size,
                    max_size=record.max_size,
                    num_connections=record.num_connections,
                    source_port=record.source_port,
                    destination_port=record.destination_port,
                    source_key=record.source_key,
                    destination_key=record.destination_key
                )