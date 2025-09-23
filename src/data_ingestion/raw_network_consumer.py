import asyncio, os, signal, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from schemas.ScadaLog import ScadaLog
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError


schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("scada_raw_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Data.Raw.Scada')
kafka_data_error_topic = os.getenv("KAFKA_ERROR_TOPIC", "Data.Error")
batch_size = int(os.getenv("BATCH_SIZE", "3000"))
linger_ms = int(os.getenv("LINGER_MS", 200))
max_records = int(os.getenv("MAX_RECORDS", "10000"))
pg_table = os.getenv("PG_TABLE", "scada_raw")
pg_stage_table = os.getenv("PG_STAGE_TABLE", "scada_staging")
pg_dsn = os.getenv("PG_DSN")
stop_event = asyncio.Event()
columns=(
                "id",
                "system_id",
                "log_ts",
                "source_ip",
                "source_port",
                "source_mac",
                "destination_ip",
                "destination_port",
                "destination_mac",
                "protocol",
                "modbus_func",
                "source_number_packets",
                "destination_number_packets",
                "total_size",
                "attributes"
            )
merge_sql = f"""
        INSERT INTO {pg_table} AS t (
        id, system_id, log_ts,
        source_ip, source_port, source_mac,
        destination_ip, destination_port, destination_mac,
        protocol, modbus_func,
        source_number_packets, destination_number_packets, total_size,
        attributes
        )
        SELECT
        id, system_id, log_ts,
        source_ip, source_port, source_mac,
        destination_ip, destination_port, destination_mac,
        protocol, modbus_func,
        source_number_packets, destination_number_packets, total_size,
        attributes
        FROM {pg_stage_table}
        ON CONFLICT (id, log_ts) DO NOTHING;

        TRUNCATE {pg_stage_table};
        """
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

def validate_required_field(scada_log: ScadaLog):
    is_src_valid = (scada_log.source_ip is not None) or (scada_log.source_mac is not None)
    is_dest_valid = (scada_log.destination_ip is not None) or (scada_log.destination_ip is not None)

    return is_dest_valid and is_src_valid

async def process_batch(pool: AsyncConnectionPool, rows: list[tuple]):
    if not rows:
        return
    
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            stmt = f"COPY {pg_stage_table} ({', '.join(columns)}) FROM STDIN"
            async with cur.copy(stmt) as cp:
                for r in rows:                        
                    try:
                        await cp.write_row(r)
                    except Exception as e:
                        log.error(f"Error inserting: {r}")
            await cur.execute(merge_sql)
            log.info(f"Processed {len(rows)} messages...")


async def consume_raw_scada():
    async with AsyncConnectionPool(pg_dsn, max_size=8, kwargs={"autocommit": False}) as pool:
        consumer = AIOKafkaConsumer(
            kafka_scada_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            group_id='raw_scada_consumer_group',
            enable_auto_commit=False,
            auto_offset_reset="latest",
            fetch_min_bytes=1*1024*1024,
            fetch_max_bytes=64*1024*1024,
            fetch_max_wait_ms=linger_ms,
            request_timeout_ms=40000
        )
        await consumer.start()
        log.info("ready to consume messages...")
        buffer: list[tuple] = []
        last_flush = asyncio.get_event_loop().time()
        try:
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
                            #TODO: do any validation and cleaning here

                            record = ScadaLog.model_validate(data)

                            if not validate_required_field(record):
                                log.warning(f"Scada Log should have either mac or ip for source/destination: {record.id}")
                                await send_to_error(message)
                                continue
                            row = (
                                record.id,
                                record.system_id,
                                record.log_ts,
                                record.source_ip if record.source_ip != "N/A" else None,
                                record.source_port,
                                record.source_mac,
                                record.destination_ip if record.destination_ip != "N/A" else None,
                                record.destination_port,
                                record.destination_mac,
                                record.protocol,
                                record.modbus_func,
                                record.source_number_packets or 0,
                                record.destination_number_packets or 0,
                                record.total_size or 0,
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
    loop.run_until_complete(consume_raw_scada())