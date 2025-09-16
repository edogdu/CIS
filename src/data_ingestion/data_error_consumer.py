import asyncio
from aiokafka import AIOKafkaConsumer
import os
import logging

logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_errors.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')


kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_data_errors_topic = os.getenv('KAFKA_DATA_ERROR_TOPIC', 'Data.Error')

async def consume_data_errors():
    consumer = AIOKafkaConsumer(
        kafka_data_errors_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='data_error_consumer_group'
    )
    await consumer.start()
    try:
        async for msg in consumer:
            print(f"Consumed message: {msg.value.decode('utf-8')}")
            logging.error(f"Data Error: {msg.value.decode('utf-8')}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_data_errors())