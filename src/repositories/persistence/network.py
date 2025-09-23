from factories.data import DataFactory

class NetworkRepository:
    
    async def register_external_endpoint_if_not_exists(system_id: str, ip: str, mac_address: str, port:int):
        pool = await DataFactory.get_pg_pool()
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                if(ip is not None and ip != ""):
                    await cur.execute("""
                    SELECT endpoint_id FROM network_endpoints
                    WHERE system_id = %s AND ip = %s
                """, (system_id, f"{ip}"))
                    current_row = await cur.fetchall()
                    if not current_row or len(current_row) == 0:
                        asset_id = f"{system_id}_external_{ip.replace('.','_')}_{port}"
                        endpoint_id = f"{asset_id}_{ip}"
                        await cur.execute("""
                                    INSERT INTO assets (asset_id, system_id, asset_type, asset_name, validated)
                                    VALUES (%s, %s, %s, %s, TRUE)
                                    ON CONFLICT (asset_id) DO UPDATE 
                                    SET last_seen_ts = now(), validated = FALSE
                                """, (asset_id, system_id, "external", f"external_{ip}_{port}"))
                        await cur.execute("""
                            INSERT INTO network_endpoints (endpoint_id, system_id, ip, asset_id, validated)
                            VALUES (%s, %s, %s, %s, TRUE)
                            ON CONFLICT (endpoint_id) DO UPDATE 
                            SET last_seen_ts = now(), validated = FALSE
                        """, (endpoint_id, system_id, f"{ip}", asset_id))
                        return (endpoint_id, asset_id)
                if((ip is None or ip == "") and mac_address is not None and mac_address != ""):
                    asset_id = f"{system_id}_external_{mac_address.replace(':','_')}_{port}"
                    endpoint_id = f"{asset_id}_{mac_address.replace(':','_')}"
                    await cur.execute("""
                                INSERT INTO assets (asset_id, system_id, asset_type, asset_name, validated)
                                VALUES (%s, %s, %s, %s, TRUE)
                                ON CONFLICT (asset_id) DO UPDATE 
                                SET last_seen_ts = now(), validated = FALSE
                            """, (asset_id, system_id, "external", f"external_{ip}_{port}"))
                    await cur.execute("""
                        INSERT INTO network_endpoints (endpoint_id, system_id, mac, asset_id, validated)
                        VALUES (%s, %s, %s, %s, TRUE)
                        ON CONFLICT (endpoint_id) DO UPDATE 
                        SET last_seen_ts = now(), validated = FALSE
                    """, (endpoint_id, system_id, f"{mac_address}", asset_id))
                    return (endpoint_id, asset_id)
            return None
