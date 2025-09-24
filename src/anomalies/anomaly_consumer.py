import asyncio, os, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from schemas.GenerateGraphRequest import GenerateGraphRequest
from pydantic import ValidationError
from repositories.graphs.snapshots import SnapshotRepository
from repositories.persistence.aggregate import AggregateRepository
from schemas.ScadaAggregate import ScadaAggregate
from repositories.persistence.network import NetworkRepository
from repositories.graphs.systems import SystemRepository
from factories.data import DataFactory

schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_anomaly_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("anomaly_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Anomaly.Predict')

async def consume_generate_graph_request():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='anomaly_consumer_group'
    )
    await consumer.start()
    try:
        
        async for msg in consumer:
            body = orjson.loads(msg.value)
            log.info(f"Received message: {body}")
            snapshot_id = body.get("snapshot_id", None)
            
            if snapshot_id:
                # create gnn model

                # train gnn model

                # store results

                #store xai results

                # if anomaly detected, produce to anomaly topic
                log.info(f"Processing existing snapshot_id: {snapshot_id}")
                continue

    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_generate_graph_request())