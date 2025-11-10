import asyncio, os, json, logging, orjson
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import ValidationError
from repositories.graphs.snapshots import SnapshotRepository
from repositories.persistence.aggregate import AggregateRepository
from schemas.ScadaAggregate import ScadaAggregate
from schemas.DetectAnomalyRequest import DetectAnomalyRequest
from repositories.persistence.network import NetworkRepository
from repositories.graphs.systems import SystemRepository
from factories.data import DataFactory
from factories.models import ModelRepositoryFactory
from schemas.ModelTypes import ModelTypes
from schemas.XaiTypes import XaiTypes
from repositories.graphs.pyg_builder import to_pyg_data, global_schema, ylabel_to_index, index_to_ylabel, y_labels
from repositories.persistence.anomaly import AnomalyRepository
import torch
from models.gnn import GNNClassifierModel, build_data_loaders, evaluate_model, fit_model, test_model, validate_dataset_integrity
import numpy as np
logging.info("Imported y_labels in anomaly_consumer.py: %s", y_labels)
SEM = asyncio.Semaphore(2)  # limit to 2 concurrent processing

schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_anomaly_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("anomaly_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Anomaly.Predict')

def _snapshots_to_dataset(snapshots):
    dataset = []
    write_name = True
    for snapshot in snapshots:
        try:
            pyg_data = to_pyg_data(snapshot, schema=global_schema, write_name=write_name)
            dataset.append(pyg_data)
            write_name = False  # only write feature names once
        except Exception as e:
            log.error(f"Error converting snapshot to PyG data: {e}")
    return dataset

async def handle_message(request):
    async with SEM:
        await _process_request(request)

async def _process_request(request: DetectAnomalyRequest):
    """Train model based on src/models/gnn.py GNNAnomalyDetector class which uses GNNModel"""
    # get all snapshots
    snapshots = await SnapshotRepository.get_all_snapshots(request.duration, request.system_id)

    # Convert each snapshot to PyG data
    temp_data = _snapshots_to_dataset(snapshots)
    dataset = []
    y_counts = np.zeros(len(y_labels), dtype=int)

    for data in temp_data:

        # get y Labels from AggregateRepository
        log.info(f"Getting labels for snapshot ID: {data.snapshot_id}")
        labels = await AggregateRepository.get_labels_for_snapshot(data.snapshot_id, request.system_id, request.duration)
        log.info(f"Snapshot ID: {data.snapshot_id} has labels: {labels}")

        is_normal = all(label.lower() == 'normal' for label in labels)
        i = ylabel_to_index('normal') if is_normal else ylabel_to_index('anomaly')
        data.y = torch.tensor([i], dtype=torch.float32)
        y_counts[i] += 1
        dataset.append(data)

    validate_dataset_integrity(dataset)
    log.info(f"Prepared dataset with {len(dataset)} graphs")
    log.info(f"Label distribution: " + ", ".join([f"{index_to_ylabel(i)}: {count}" for i, count in enumerate(y_counts)]))
    config = {
        "hidden_dim": 92,
        "dropout": 0.7,
        "learning_rate": 3e-4,
        "weight_decay": 0.001,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0001,
        "max_epochs": 20,
    }

    # create model object
    #self, config: Dict[str, Any], in_channels: int=51, out_channels: int=6
    model = GNNClassifierModel(config=config,
                               in_channels=dataset[0].num_node_features)
    
    # split dataset into train/val/test
    train_loader, val_loader, test_loader = build_data_loaders(dataset)
    

    # train model
    model, criterion, metrics = fit_model(model, train_loader, val_loader, config)

    # validate model
    val_metrics = evaluate_model(model, val_loader, criterion)
    log.info(f"Validation Metrics: {val_metrics}")

    # test model
    test_metrics = test_model(model, test_loader, criterion)
    log.info(f"Test Metrics: {test_metrics}")
    # persist results to export folder for analysis as csv file


async def _process_request_OLD(request: DetectAnomalyRequest):
    # run in a new thread to avoid blocking        
    snapshots = []
    if request.snapshot_id and request.snapshot_id.strip() != "":
        snapshots.append(await SnapshotRepository.get_snapshot(request.snapshot_id))
    else:
        snapshots = await SnapshotRepository.get_snapshots(request.start_time, request.end_time, request.duration, request.system_id)
    if len(snapshots) == 0:
        log.error(f"No snapshots found between {request.start_time} and {request.end_time}")
        return  []
    results = []
    dataset = _snapshots_to_dataset(snapshots)
    if not dataset:
        log.error("No valid PyG data could be created from snapshots")
        return []

    config = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 100,
        "dropout": 0.5,
        "val_frac": 0.1,
        "test_frac": 0.1,
        "is_undirected": True,
        "xai_topk": 20,
        "xai_loss_min": None,
        "export_path": "./exports/models",
        "bucket_duration": request.duration,
        "system_id": request.system_id,
        "model_type": request.model_type.value,
        "xai_type": request.xai_type.value if request.xai_type else XaiTypes.NONE.value,
    }
    if request.model_config:
        config.update(request.model_config)
    log.info(f"Model config: {config}")
    
    # create model runner based on model type
    if request.model_type == ModelTypes.GNN:
        model_runner = ModelRepositoryFactory.get_model_runner(
            model_type=request.model_type,                    
            input_dim=dataset[0].num_node_features,
            hidden_dim=16,
            output_dim=16,
            xai_type=request.xai_type,
            device=None,
            config=config,
        )
        if request.is_train:
            model_runner.prepare_splits(dataset)
            log.info(f"Training model: {request.model_type}")
            def _train():
                return model_runner.train_epochs(epochs=config['epochs'])
            # run in a new thread to avoid blocking
            loop = asyncio.get_event_loop()
            train_loss = await loop.run_in_executor(None, _train)
            log.info(f"Training completed with final loss: {train_loss}")
            #system_id, model_name, history, path
            model_runner.save_history_csv(request.system_id,
                                            request.model_type.value,
                                            train_loss,
                                            './logs')
            edge_percentile = request.edge_threshold_percent if request.edge_threshold_percent else 0.05
            snapshot_percentile = request.snapshot_threshold_percent if request.snapshot_threshold_percent else 0.01
            log.info(f"Determining anomaly thresholds at edge percentile: {edge_percentile}, snapshot percentile: {snapshot_percentile}")
            model_runner.determine_anomaly_thresholds(edge_percentile=edge_percentile, snapshot_percentile=snapshot_percentile)
            results = model_runner.evaluate_test()
            log.info(f"Evaluation results: {results}")
            model_runner.save_test_csv(request.system_id,
                                        request.model_type.value,
                                        results,
                                        './logs')
            model_runner.save_model()
            #for data in dataset:
            #    log.info(f"Explaining data with {data.num_nodes} nodes and {data.num_edges} edges")
            #    model_runner.explain(data, topk=config.get('xai_topk', 20))
        elif request.is_threshold_only:
            log.info(f"Loading model for threshold determination: {request.model_type}")
            model_runner = ModelRepositoryFactory.get_model_runner(
            model_type=request.model_type,                    
            input_dim=dataset[0].num_node_features,
            hidden_dim=16,
            output_dim=16,
            xai_type=request.xai_type,
            device=None,
            config=config,
            load_from_path=True
            )
            model_runner.prepare_splits(dataset)
            log.info("Model loaded, determining thresholds")
            model_runner.determine_anomaly_thresholds(edge_percentile=request.edge_threshold_percent, snapshot_percentile=request.snapshot_threshold_percent)
            model_runner.save_model()
            results = {
                "edge_threshold": model_runner.edge_threshold,
                "snapshot_threshold": model_runner.snapshot_threshold
            }
            log.info(f"Determined thresholds: {results}")
        else:
            log.info(f"Loading model for inference: {request.model_type}")
            model_runner = ModelRepositoryFactory.get_model_runner(
            model_type=request.model_type,                    
            input_dim=dataset[0].num_node_features,
            hidden_dim=16,
            output_dim=16,
            xai_type=request.xai_type,
            device=None,
            config=config,
            load_from_path=True
            )
            log.info("Model loaded, starting inference")
            anomalies = []
            for i, data in enumerate(dataset):
                log.info(f"Processing graph #{i} (Snapshot ID: {getattr(data, 'snapshot_id', 'N/A')}) for anomalies...")
                data_anomalies = model_runner.detect_anomalies(data)
                if data_anomalies:
                    log.info(f"Detected anomalies for snapshot ID: {getattr(data, 'snapshot_id', 'N/A')}")
                    tensorid_to_graphid = getattr(data, 'tensorid_to_graphid', {})
                    for anomaly in data_anomalies:
                        anomaly['snapshot_id'] = getattr(data, 'snapshot_id', 'N/A')
                        anomaly['src_graph_id'] = tensorid_to_graphid.get(anomaly['src_tensorid'], anomaly['src_tensorid'])
                        anomaly['dst_graph_id'] = tensorid_to_graphid.get(anomaly['dst_tensorid'], anomaly['dst_tensorid'])
                    anomalies.extend(data_anomalies)
            #log.info(f"Anomalies detected: {anomalies}")
            if anomalies:
                await AnomalyRepository().save_anomalies(request.system_id,
                                                         request.model_type.value,
                                                         request.duration,                                                         
                                                        anomalies)
                model_runner.save_anomalies_csv(request.system_id,
                                            request.model_type.value,
                                            anomalies,
                                            './logs')
            results = anomalies
        return results

    elif request.model_type == ModelTypes.XGBOOST:
        raise NotImplementedError("XGBoost model runner not implemented yet")
    
    else:
        log.error(f"Unsupported model type: {request.model_type}")
        return []

async def consume_anomaly_predict_request():
    consumer = AIOKafkaConsumer(
        kafka_scada_topic,
        bootstrap_servers=kafka_bootstrap_servers,
        group_id='anomaly_consumer_group',
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
            body = orjson.loads(msg.value)
            log.info(f"Received message: {body}")
            try:
                request = DetectAnomalyRequest.model_validate(body)                
            except ValidationError as e:
                log.error(f"Validation error: {e.json()}")
                continue
            # process request to avoid blocking
            log.info(f"Processing request: {request.model_type}, {request.xai_type}, {request.system_id}, {request.start_time} to {request.end_time}")
            t = asyncio.create_task(handle_message(request))
            t.add_done_callback(lambda task: log.info("Request processing task completed with result: %s", task.result() if not task.exception() else f"Error: {task.exception()}"))
            tasks.add(t)
            t.add_done_callback(lambda task: tasks.discard(task))
            log.info("Request processing task started")

    finally:
        await asyncio.gather(*tasks)  # wait for all tasks to complete
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(consume_anomaly_predict_request())