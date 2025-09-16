import asyncio
from aiokafka import AIOKafkaConsumer
import os
import logging

logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Data.Raw.Physical')

async def consume_raw_physical():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='raw_physical_consumer_group'
    )
    await consumer.start()
    try:
        async for msg in consumer:
            print(f"Consumed message: {msg.value.decode('utf-8')}")
            logging.info(f"Raw Physical Data: {msg.value.decode('utf-8')}")
    finally:
        await consumer.stop()
        
if __name__ == "__main__":
    asyncio.run(consume_raw_physical())