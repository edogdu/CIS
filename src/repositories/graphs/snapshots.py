from factories.data import DataFactory
from schemas.GenerateGraphRequest import GenerateGraphRequest
from schemas.ScadaAggregate import ScadaAggregate
from schemas.PhysicalAggregate import PhysicalAggregate
import datetime

class SnapshotRepository:
        
    async def create_snapshot(request: GenerateGraphRequest, bucket) -> str:
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            snapshot_id = f"{request.system_id}_{request.duration}s_{bucket}"
            await session.run(
                """
                MERGE (s:Snapshot {snapshot_id: $snapshot_id})
                SET s.system_id = $system_id,
                    s.start_time = $start_time,
                    s.end_time = $end_time,
                    s.created_at = datetime(),
                    s.duration = $duration
                """,
                snapshot_id=snapshot_id,
                system_id=request.system_id,
                start_time=bucket,
                end_time=(bucket + datetime.timedelta(seconds=request.duration)),
                duration=request.duration
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
                        start_time: $bucket,
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
                    MATCH (s:Snapshot {snapshot_id: $snapshot_id})
                    MATCH (src:Endpoint {key: $source_key})
                    MATCH (dst:Endpoint {key: $destination_key})
                    MERGE (conn:Connection {
                        start_time: $bucket,
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

    async def get_snapshots(start_time: str, end_time: str, duration:int, system_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            #WHERE s.start_time >= datetime($start_time) AND s.end_time <= datetime($end_time) AND s.duration = $duration AND s.system_id = $system_id
            result = await session.run(
                """
                MATCH (s:Snapshot{system_id: $system_id, duration: $duration})
                WHERE datetime(s.start_time) >= datetime($start_time) 
                    AND datetime(s.start_time) <= datetime($end_time)
                OPTIONAL MATCH (c:Connection {snapshot_id: s.snapshot_id})
                WITH s, collect(c) as connection_nodes
                OPTIONAL MATCH (m:Measurement {snapshot_id: s.snapshot_id})
                WITH s, connection_nodes + collect(m) as anchor_nodes
                WHERE size(anchor_nodes) > 0
                CALL apoc.path.subgraphAll(anchor_nodes, {
                    maxLevel: 3,
                    labelFilter: '+Endpoint|+Asset|+Measurement|+Connection',
                    relationshipFilter: 'INITIATES>|TERMINATES_AT>|HAS_MEASUREMENT>|CONNECTED_TO>'
                })
                YIELD nodes, relationships
                WITH s, 
                [n IN nodes | {
                    id: id(n),
                    labels: labels(n),
                    properties: apoc.map.removeKeys(properties(n), ['snapshot_id', 'system_id'])
                }] AS node_list,     
                [r IN relationships | {
                    type: type(r),
                    source: id(startNode(r)),
                    target: id(endNode(r)),
                    properties: properties(r)
                }] AS relationship_list
                ORDER BY s.start_time
                RETURN {
                    snapshot_id: s.snapshot_id,
                    start_time: s.start_time,
                    end_time: s.end_time,
                    duration: s.duration,
                    nodes: node_list,
                    relationships: relationship_list
                } AS snapshot
                """,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                system_id=system_id
            )
            
            # collect all records into a list            
            if result:
                return [record["snapshot"] async for record in result]
            else:
                return []
            
        
    async def get_snapshot(snapshot_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            result = await session.run(
                """
                MATCH (c:Connection {snapshot_id: $snapshot_id})
                WITH collect(c) as connection_nodes
                MATCH (m:Measurement {snapshot_id: $snapshot_id})
                WITH connection_nodes + collect(m) as anchor_nodes
                CALL apoc.path.subgraphAll(anchor_nodes, {
                    maxLevel: 5
                })
                YIELD nodes, relationships
                RETURN {
                    nodes: [node IN nodes | {
                        id: id(node),
                        labels: labels(node),
                        // MODIFICATION: Remove both keys from the properties map
                        properties: apoc.map.removeKeys(properties(node), ['snapshot_id', 'system_id'])
                    }],
                    relationships: [rel IN relationships | {
                        type: type(rel),
                        source: id(startNode(rel)),
                        target: id(endNode(rel)),
                        properties: properties(rel)
                    }]
                }
                """,
                snapshot_id=snapshot_id
            )
            record = await result.single()
            if record:
                return record["value"]
            else:
                return None