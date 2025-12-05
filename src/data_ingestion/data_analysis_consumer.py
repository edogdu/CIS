import asyncio, os, signal, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from anomalies.anomaly_consumer import SEM
from schemas.ScadaLog import ScadaLog
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Anomaly.Predict')


async def _process_data_analysis_request():
    await _analyze_original_physical_data()
    await _analyze_original_scada_data()
    await _analyze_aggregated_physical_data()
    await _analyze_aggregated_scada_data()
    await _analyze_graph_data()


async def _analyze_original_physical_data():
    pass

async def _analyze_original_scada_data():
    pass

async def _analyze_aggregated_physical_data():
    pass

async def _analyze_aggregated_scada_data():
    pass

async def _analyze_graph_data():
    pass


async def handle_message(request):
    async with SEM:
        await _process_data_analysis_request()


schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')

async def consume_anomaly_predict_request():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='data_analysis_consumer_group',
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        session_timeout_ms=45000,
        max_poll_interval_ms=300000,
        heartbeat_interval_ms=15000,
        request_timeout_ms=60000

    )
    await consumer.start()
    tasks = set()
    try:
        
        async for msg in consumer:

            
            t = asyncio.create_task(handle_message())
            t.add_done_callback(lambda task: logging.info("Request processing task completed with result: %s", task.result() if not task.exception() else f"Error: {task.exception()}"))
            tasks.add(t)
            t.add_done_callback(lambda task: tasks.discard(task))
            logging.info("Request processing task started")

    finally:
        await asyncio.gather(*tasks)  # wait for all tasks to complete
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_anomaly_predict_request())