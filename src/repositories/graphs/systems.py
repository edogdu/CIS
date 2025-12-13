from factories.data import DataFactory

class SystemRepository:

    async def create_system(system_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            await session.run(
                """
                MERGE (s:System {id: $system_id})
                """,
                system_id=system_id
            )

    async def get_system(system_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            result = await session.run(
                """
                MATCH (s:System {id: $system_id})
                RETURN s
                """,
                system_id=system_id
            )
            record = await result.single()
            return record["s"] if record else None
    
    async def create_external_endpoint(system_id: str, ip: str, mac: str, key: str, asset_id: str):
        neo4j = await DataFactory.get_neo4j_instance()
        async with neo4j.session() as session:
            await session.run(
                """
                MERGE (a:Asset {asset_id: $asset_id, system_id: $system_id})
                ON CREATE SET a.asset_type = 'External', a.stage = 'Unknown', a.asset_name = $asset_id
                MERGE (e:Endpoint {key: $key})
                ON CREATE SET e.cidr = CASE WHEN e.ip CONTAINS '/' THEN toInteger(split(e.ip,'/')[1]) ELSE null END,
                    e.ip = $ip,
                    e.mac = $mac,
                    e.system_id = $system_id
                ON MATCH SET e.mac = COALESCE(e.mac, $mac), e.system_id = $system_id
                MERGE (a)-[:HAS_ENDPOINT]->(e)
                """,
                asset_id=asset_id,
                system_id=system_id,
                ip=(ip or None),
                mac=(mac or None),
                key=key
            )