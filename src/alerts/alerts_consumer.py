import asyncio, os, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_alerts_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("alerts_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Anomaly.Alerts')
pg_dsn = os.getenv("PG_DSN")

async def consume_generate_alert_request():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='alerts_consumer_group'
    )
    await consumer.start()
    try:
        
        async for msg in consumer:
            body = orjson.loads(msg.value)
            log.info(f"Received message: {body}")
            # Create alert in the database

            # fetch Mitre Att&ck techniques based on the alert type
            
            # Generate graph using the fetched techniques
                

    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_generate_alert_request())