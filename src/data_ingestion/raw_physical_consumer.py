import asyncio, os, signal, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from schemas.PhysicalLog import PhysicalLog
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError


schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')

logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("scada_raw_physical_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'cis-kafka:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Data.Raw.Physical')
batch_size = int(os.getenv("BATCH_SIZE", "3000"))
linger_ms = int(os.getenv("LINGER_MS", 200))
max_records = int(os.getenv("MAX_RECORDS", "10000"))
pg_table = os.getenv("PG_TABLE", "phys_raw")
pg_dsn = os.getenv("PG_DSN")
stop_event = asyncio.Event()

columns=(
                "id",
                "system_id",
                "log_ts",
                "measurement_id",
                "measure_value",
                "attributes"
            )

async def send_to_error(message):
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    try:
        log.warning(f"Sending Scada Log to Error...")
        await producer.send(topic=kafka_scada_topic, value=message.value)
    finally:
        await producer.stop()
def _graceful_stop():
    stop_event.set()
    
async def process_batch(pool: AsyncConnectionPool, rows: list[tuple]):
    if not rows:
        return
    
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            stmt = f"COPY {pg_table} ({', '.join(columns)}) FROM STDIN"
            async with cur.copy(stmt) as cp:
                for r in rows:                        
                    try:
                        await cp.write_row(r)
                    except Exception as e:
                        log.error(f"Error inserting: {r}")            
            log.info(f"Processed {len(rows)} messages...")


async def consume_raw_physical():
    consumer = AIOKafkaConsumer(
            kafka_scada_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id='raw_physical_consumer_group',
            enable_auto_commit=False,
            auto_offset_reset="latest",
            fetch_min_bytes=1*1024*1024,
            fetch_max_bytes=64*1024*1024,
            fetch_max_wait_ms=linger_ms,
            request_timeout_ms=40000
        )

    await consumer.start()

    try:
        async with AsyncConnectionPool(pg_dsn, max_size=8, kwargs={"autocommit": False}) as pool:
            log.info("ready to consume messages...")
            buffer: list[tuple] = []
            last_flush = asyncio.get_event_loop().time()

            while not stop_event.is_set():
                messages = await consumer.getmany(timeout_ms=linger_ms, max_records=max_records)
                got_messages = False

                for tp, msgs in messages.items():
                    if not msgs:
                        continue
                    got_messages = True
                    for message in msgs:
                        try:
                            data = orjson.loads(message.value)
                            print(f"message: {data}")
                            #TODO: do any validation and cleaning here

                            record = PhysicalLog.model_validate(data)
                            row = (
                                record.id,
                                record.system_id,
                                record.log_ts,
                                record.measurement_id,
                                record.measure_value,
                                json.dumps(record.attributes) if record.attributes is not None else None,
                            )
                            buffer.append(row)
                            if len(buffer) >= batch_size:
                                await process_batch(pool, buffer)
                                buffer.clear()
                                await consumer.commit()
                                last_flush = asyncio.get_event_loop().time()
                        except ValidationError as ve:
                            log.error(f"Validation error for : {message.offset}, {ve}")
                            await send_to_error(message)
                        except Exception as e:
                            log.exception(f"Unexpected errorError {e}")
                            await send_to_error(message)
                now = asyncio.get_event_loop().time()
                if buffer and (not got_messages or (now - last_flush) > (linger_ms / 1000.0) * 5):
                    await process_batch(pool, buffer)
                    buffer.clear()
                    await consumer.commit()
                    last_flush = now
            if buffer:
                await process_batch(pool, buffer)
                buffer.clear()
                await consumer.commit()
    finally:
        await consumer.stop()
        
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _graceful_stop)
        except NotImplementedError:
            pass
    loop.run_until_complete(consume_raw_physical())