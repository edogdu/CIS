from neo4j import AsyncGraphDatabase
import os
from psycopg_pool import AsyncConnectionPool

class DataFactory:
    _neo4j_instance = None
    _pg_pool = None
    PG_DSN = os.getenv("PG_DSN")
    PG_MIN_SIZE = int(os.getenv("PG_MIN_SIZE", 1))
    PG_MAX_SIZE = int(os.getenv("PG_MAX_SIZE", 10))
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    @classmethod
    async def get_neo4j_instance(cls):
        if cls._neo4j_instance is None:
            cls._neo4j_instance = AsyncGraphDatabase.driver(
                cls.NEO4J_URI,
                auth=(cls.NEO4J_USER, cls.NEO4J_PASSWORD)
            )
        return cls._neo4j_instance
    
    @classmethod
    async def get_pg_pool(cls) -> AsyncConnectionPool:
        if cls._pg_pool is None:
            cls._pg_pool = AsyncConnectionPool(
                conninfo=cls.PG_DSN,
                min_size=cls.PG_MIN_SIZE,
                max_size=cls.PG_MAX_SIZE
            )
            await cls._pg_pool.open()
        return cls._pg_pool
    
    @classmethod
    async def close_connections(cls):
        if cls._neo4j_instance:
            await cls._neo4j_instance.close()
            cls._neo4j_instance = None
        if cls._pg_pool:
            await cls._pg_pool.close()
            cls._pg_pool = None