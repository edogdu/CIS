from factories.data import DataFactory
from schemas.GenerateGraphRequest import GenerateGraphRequest
from schemas.ScadaAggregate import ScadaAggregate
from schemas.PhysicalAggregate import PhysicalAggregate
import datetime
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnapshotRepository:
        
    async def create_snapshot(request: GenerateGraphRequest, bucket) -> str:
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            snapshot_id = f"{request.system_id}_{request.duration}s_{bucket}"
            await session.run(
                """
                MERGE (s:Snapshot {snapshot_id: $snapshot_id})
                SET s.system_id = $system_id,
                    s.start_time = datetime($start_time),
                    s.end_time = datetime($end_time),
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
                    MERGE (s:Snapshot {snapshot_id: $snapshot_id})
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
                    MERGE (s)-[:CONTAINS]->(m)
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
                    MERGE (s)-[:CONTAINS]->(conn)
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
    async def get_snapshot(snapshot_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            result = await session.run(
                    """
                    MATCH (s:Snapshot {snapshot_id: $snapshot_id})

                    // 2. Find all Connections and Measurements contained in this snapshot
                    MATCH (s)-[:CONTAINS]->(item)
                    WHERE item:Connection OR item:Measurement

                    // 3. From those specific items, find all directly connected Endpoints and Assets
                    OPTIONAL MATCH path = (item)-[]-(related_node)
                    WHERE related_node:Endpoint OR related_node:Asset

                    // 4. Collect all the distinct pieces of the graph
                    WITH collect(DISTINCT item) as items, 
                        collect(DISTINCT related_node) as related_nodes, 
                        collect(DISTINCT relationships(path)) as rels

                    // 5. Combine all nodes and relationships into two lists for a clean return
                    WITH apoc.coll.toSet(
                            // MODIFICATION: The 's' node is removed from this list concatenation.
                            items + related_nodes
                        ) AS all_nodes,
                        apoc.coll.toSet(
                            apoc.coll.flatten(rels)
                        ) AS all_relationships

                    RETURN all_nodes AS nodes, all_relationships AS relationships
                    """,
                    snapshot_id=snapshot_id
                )
            record = await result.single()
            


            # result = await session.run(
            #     """
            #     MATCH (c:Connection {snapshot_id: $snapshot_id})
            #     WITH collect(c) as connection_nodes
            #     MATCH (m:Measurement {snapshot_id: $snapshot_id})
            #     WITH connection_nodes + collect(m) as anchor_nodes
            #     CALL apoc.path.subgraphAll(anchor_nodes, {
            #         maxLevel: 3
            #     })
            #     YIELD nodes, relationships
            #     RETURN {
            #         nodes: [node IN nodes | {
            #             id: id(node),
            #             labels: labels(node),
            #             // MODIFICATION: Remove both keys from the properties map
            #             properties: apoc.map.removeKeys(properties(node), ['snapshot_id', 'system_id'])
            #         }],
            #         relationships: [rel IN relationships | {
            #             type: type(rel),
            #             source: id(startNode(rel)),
            #             target: id(endNode(rel)),
            #             properties: properties(rel)
            #         }]
            #     }
            #     """,
            #     snapshot_id=snapshot_id
            # )            
            if record:
                return record["value"]
            else:
                return None
    async def get_snapshots(start_time: str, end_time: str, duration:int, system_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:            
            snapshots = []
            results = await session.run(
                """
                MATCH (s:Snapshot {system_id: $system_id, duration: $duration})
                WHERE s.start_time >= datetime($start_time)
                AND s.start_time <= datetime($end_time)
                
                CALL{
                    WITH s
                    MATCH (s)-[:CONTAINS]->(item)
                    WHERE item:Connection OR item:Measurement

                    OPTIONAL MATCH path = (item)-[]-(related_node)
                    WHERE related_node:Endpoint OR related_node:Asset

                    WITH s, item, related_node, relationships(path) as rel_list
                    UNWIND rel_list AS relations

                    WITH s,
                    collect(DISTINCT item) + collect(DISTINCT related_node) as nodes,
                    collect(DISTINCT relations) as rels

                    RETURN {
                        snapshot_id: s.snapshot_id,
                        start_time: s.start_time,
                        nodes: [node IN nodes | {
                            id: id(node),
                            labels: labels(node),
                            properties: apoc.map.removeKeys(properties(node), ['snapshot_id', 'system_id'])
                        }],
                        relationships: [rel IN rels | {
                            type: type(rel),
                            source: id(startNode(rel)),
                            target: id(endNode(rel)),
                            properties: properties(rel)
                        }]
                    } AS snapshot
                }
                RETURN collect(snapshot) AS snapshots
                """,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                system_id=system_id
            )
            record = await results.single()
            if record:
                snapshots = record["snapshots"]
            logger.info(f"Fetched {len(snapshots)} snapshots.")
            return snapshots

            #WHERE s.start_time >= datetime($start_time) AND s.end_time <= datetime($end_time) AND s.duration = $duration AND s.system_id = $system_id
            # result = await session.run(
            #     """
            #     MATCH (s:Snapshot{system_id: $system_id, duration: $duration})
            #     WHERE datetime(s.start_time) >= datetime($start_time) 
            #         AND datetime(s.start_time) <= datetime($end_time)
            #     OPTIONAL MATCH (s)-[:CONTAINS]->(c:Connection)
            #     WITH s, collect(c) as connection_nodes
            #     OPTIONAL MATCH (s)-[:CONTAINS]->(m:Measurement)
            #     WITH s, connection_nodes + collect(m) as anchor_nodes
            #     WHERE size(anchor_nodes) > 0
            #     UNWIND anchor_nodes as node
            #     CALL apoc.path.expandConfig([node], {
            #         maxLevel: 3,
            #         bfs: true,
            #         uniqueness: 'NODE_GLOBAL',
            #         labelFilter: '+Endpoint|+Asset|+Measurement|+Connection',
            #         relationshipFilter: 'INITIATES>|TERMINATES_AT>|HAS_MEASUREMENT>|CONNECTED_TO>'
            #     })
            #     YIELD path
            #     WITH s, collect(DISTINCT nodes(path)) as path_nodes, collect(DISTINCT relationships(path)) as path_rels
            #     WITH s,
            #             apoc.coll.toSet(apoc.coll.flatten(path_nodes)) as nodes,
            #             apoc.coll.toSet(apoc.coll.flatten(path_rels)) as relationships
            #     RETURN {
            #         snapshot_id: s.snapshot_id,
            #         start_time: s.start_time,
            #         end_time: s.end_time,
            #         duration: s.duration,
            #         nodes: node_list,
            #         relationships: relationship_list
            #     } AS snapshot
            #     """,
            #     start_time=start_time,
            #     end_time=end_time,
            #     duration=duration,
            #     system_id=system_id
            # )

            # while not is_finished:
            #     logger.info(f"Fetching snapshots with skip={skip} and limit={limit}")
            #     result = await session.run(
            #         """
            #         MATCH (s:Snapshot {system_id: $system_id, duration: $duration})
            #         WHERE s.start_time >= datetime($start_time)
            #         AND s.start_time <= datetime($end_time)
            #         WITH s
            #         ORDER BY s.start_time
            #         SKIP $skip LIMIT $limit

            #         MATCH (s)-[:CONTAINS]->(anchor)
            #         WITH s, anchor WHERE anchor IS NOT NULL and (anchor:Connection OR anchor:Measurement)
            #         CALL apoc.path.expandConfig(anchor, {
            #         maxLevel: 2,
            #         bfs: false,
            #         uniqueness: 'NODE_PATH',
            #         labelFilter: '+Endpoint|+Asset|+Measurement|+Connection',
            #         relationshipFilter: 'INITIATES>|TERMINATES_AT>|HAS_MEASUREMENT>|CONNECTED_TO>'
            #         }) YIELD path
            #         UNWIND nodes(path) AS n
            #         UNWIND relationships(path) AS r
            #         RETURN s.snapshot_id AS snapshot_id, id(n) AS node_id, labels(n) AS node_labels
            #         , properties(n) AS node_props, id(r) AS rel_id, type(r) AS rel_type, properties(r) AS rel_props
            #         """,
            #         start_time=start_time,
            #         end_time=end_time,
            #         duration=duration,
            #         system_id=system_id,
            #         skip=skip,
            #         limit=limit
            #     )


            #     batch_snapshots = {}
            #     logger.info("Processing fetched records...")
            #     records = await result.to_df()
            #     logger.info(f"Records dataframe shape: {records.shape}")
            #     logger.info(f"Records dataframe columns: {records.columns.tolist()}")                
            #     for index, record in records.iterrows():
            #         logger.info(f"Processing record: {record}")
            #         snapshot_id = record["snapshot_id"]                    
            #         if snapshot_id not in batch_snapshots:
            #             batch_snapshots[snapshot_id] = {
            #                 "snapshot_id": snapshot_id,
            #                 "nodes": [],
            #                 "relationships": []
            #             }
            #             node = {
            #             "id": record["node_id"],
            #             "labels": record["node_labels"],
            #             "properties": record["node_props"]
            #             }
            #             rel = {
            #                 "id": record["rel_id"],
            #                 "type": record["rel_type"],
            #                 "properties": record["rel_props"]
            #             }
                    
            #             logger.info(f"Adding node {node['id']} and relationship {rel['id']} to snapshot {snapshot_id}")
            #             batch_snapshots[snapshot_id]["nodes"].append(node)
            #             batch_snapshots[snapshot_id]["relationships"].append(rel)
            #     logger.info(f"Fetched {len(batch_snapshots)} snapshots in this batch.")
            #     if len(records) < limit:
            #         is_finished = True
            #     else:
            #         skip += limit

            # collect all records into a list
            # if snapshots:
            #     return snapshots
            # else:
            #     return []
            
        
