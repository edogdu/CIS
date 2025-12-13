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
                    , filename='./logs/kafka_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("data_graph_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Data.Graph')
pg_table = os.getenv("PG_TABLE", "scada_raw")
pg_stage_table = os.getenv("PG_STAGE_TABLE", "scada_staging")
pg_dsn = os.getenv("PG_DSN")

async def consume_generate_graph_request():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='data_graph_consumer_group'
    )
    await consumer.start()
    try:
        
        async for msg in consumer:
            body = orjson.loads(msg.value)
            

            request = GenerateGraphRequest.model_validate(body)
            if(not request):
                log.error(f"Invalid GenerateGraphRequest: {body}")
                continue
            if(request.end_time <= request.start_time):
                log.error(f"Invalid GenerateGraphRequest: end_time must be after start_time: {body}")
                continue
            if(request.duration not in [10, 16, 30]):
                log.error(f"Invalid GenerateGraphRequest: duration must be one of [10, 16, 30]: {body}")
                continue
            log.info(f"Generate Graph Request: {request.model_dump_json()}")
            scada_records = await AggregateRepository.fetch_scada_aggregates(start_time=request.start_time, end_time=request.end_time, duration=request.duration, system_id=request.system_id)
            physical_records = await AggregateRepository.fetch_physical_aggregates(start_time=request.start_time, end_time=request.end_time, duration=request.duration, system_id=request.system_id)

            log.info(f"Fetched {len(physical_records)} physical records for system_id {request.system_id} between {request.start_time} and {request.end_time}")
            log.info(f"Fetched {len(scada_records)} SCADA records for system_id {request.system_id} between {request.start_time} and {request.end_time}")

            if not scada_records and not physical_records:
                log.warning(f"No SCADA or physical records found for system_id {request.system_id} between {request.start_time} and {request.end_time}")
                continue
            buckets = set([record.bucket for record in scada_records] + [record.bucket for record in physical_records])
            log.info(f"Total unique time buckets to process: {len(buckets)}")

            for bucket in buckets:
                log.info(f"Processing bucket: {bucket}")
                bucket_scada_records = [record for record in scada_records if record.bucket == bucket]
                bucket_physical_records = [record for record in physical_records if record.bucket == bucket]
                print(f"Bucket {bucket} has {len(bucket_scada_records)} SCADA records and {len(bucket_physical_records)} physical records")
                log.info(f"Bucket {bucket} has {len(bucket_scada_records)} SCADA records and {len(bucket_physical_records)} physical records")
                snapshot_id = await SnapshotRepository.create_snapshot(request, bucket)
                for record in bucket_scada_records:
                    log.debug(f"SCADA Record: {record.model_dump_json()}")
                    source_external_ids = await NetworkRepository.register_external_endpoint_if_not_exists(
                        system_id=request.system_id,
                        ip=record.source_ip,
                        port=record.source_port,
                        mac_address=record.source_mac
                    )
                    if source_external_ids:
                        log.info(f"Registered source external endpoint: {source_external_ids}")
                        await SystemRepository.create_external_endpoint(
                            system_id=request.system_id,
                            ip=record.source_ip,
                            mac=record.source_mac,
                            key=record.source_key,
                            asset_id=source_external_ids[1]
                        )

                    destination_external_ids = await NetworkRepository.register_external_endpoint_if_not_exists(
                        system_id=request.system_id,
                        ip=record.destination_ip,
                        port=record.destination_port,
                        mac_address=record.destination_mac
                    )
                    if destination_external_ids:
                        log.info(f"Registered destination external endpoint: {destination_external_ids}")
                        await SystemRepository.create_external_endpoint(
                            system_id=request.system_id,
                            ip=record.destination_ip,
                            mac=record.destination_mac,
                            key=record.destination_key,
                            asset_id=destination_external_ids[1]
                        )

                if bucket_physical_records and len(bucket_physical_records) > 0:
                    await SnapshotRepository.add_physical_data_to_snapshot(snapshot_id, bucket_physical_records)
                if bucket_scada_records and len(bucket_scada_records) > 0:
                    await SnapshotRepository.add_scada_data_to_snapshot(snapshot_id, bucket_scada_records)

    finally:
        await consumer.stop()
        await DataFactory.close_connections()
        


if __name__ == "__main__":
    asyncio.run(consume_generate_graph_request())