import asyncio, os, json, logging, orjson
from concurrent.futures import ThreadPoolExecutor
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import ValidationError
from repositories.graphs.snapshots import SnapshotRepository
from repositories.persistence.aggregate import AggregateRepository
from schemas.ScadaAggregate import ScadaAggregate
from schemas.DetectAnomalyRequest import DetectAnomalyRequest
from repositories.persistence.network import NetworkRepository
from repositories.graphs.systems import SystemRepository
from factories.data import DataFactory
from schemas.ModelTypes import ModelTypes
from schemas.XaiTypes import XaiTypes
from repositories.graphs.pyg_builder import to_pyg_hetero_data, ylabel_to_index, index_to_ylabel, y_labels, write_hetero_feature_mappings, visualize_features_distribution
from repositories.persistence.anomaly import AnomalyRepository
import torch
import models.gnn_het_single as gnn_het_single
import numpy as np
import pandas as pd
import models.gnn_het as gnn_het
import models.gnn_het_bin as gnn_het_bin
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.info("Imported y_labels in anomaly_consumer.py: %s", y_labels)
SEM = asyncio.Semaphore(2)  # limit to 2 concurrent processing
executor = ThreadPoolExecutor(max_workers=1)

schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
logging.basicConfig(level=logging.INFO
                    , filename='./logs/kafka_anomaly_consumers.log'
                    ,format='%(asctime)s - %(levelname)s - %(message)s'
                    , filemode='a')
log = logging.getLogger("anomaly_consumer")
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
kafka_scada_topic = os.getenv('KAFKA_TOPIC', 'Anomaly.Predict')

def _train_and_eval_detector_classifier_sync(dataset, y_counts):
    """
    This function runs in a separate thread. 
    It handles the blocking CPU-bound tasks: Training, Eval, Plotting.
    """
    log.info(f"Label distribution: " + ", ".join([f"{index_to_ylabel(i)}: {count}" for i, count in enumerate(y_counts)]))
    
    config = {
        "hidden_dim": 84,
        "dropout": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 1e-5,
        "num_layers": 3,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 5e-4,
        "max_epochs": 300,
        "num_heads": 4
    }

    # Initialize Models
    detector_model = gnn_het_bin.GNNHeteroAnomalyDetectionModel(config=config, metadata=dataset[0].metadata())
    classify_model = gnn_het.GNNHeteroClassifierModel(config=config,
                                   metadata=dataset[0].metadata())

    # split dataset into train/val/test
    bin_data, anom_data, final_test_data = detector_model.build_data_loaders(dataset)

    # train and validate detector model
    _, criterion, _ = detector_model.fit_model(bin_data[0], bin_data[1], config)
    val_metrics = detector_model.evaluate_model(bin_data[1])
    log.info(f"Binary Detector Validation Metrics: {val_metrics}")

    # train and validate classifier model
    logging.info("Training Anomaly Classifier Model...")
    _, anom_criterion, _ = classify_model.fit_model(anom_data[0], anom_data[1], config)
    logging.info("Anomaly Classifier Model Training Completed.")
    val_metrics = classify_model.evaluate_model(anom_data[1])
    log.info(f"Anomaly Classifier Validation Metrics: {val_metrics}")

    # perform final test on combined model
    y_all, y_pred_all = [], []
    y_all_bin, y_pred_all_bin = detector_model.test_model(final_test_data, test_description="Final Test Set - Binary Detector")

    # filter final_test_data and create new DataLoader for only detected anomalies
    anomaly_test_data = []
    anomaly_test_idx = []
    for i, label in enumerate(y_pred_all_bin):
        if label == 1 and y_all_bin[i] > 0:  # only test samples predicted as anomaly and true label is anomaly
            d = final_test_data.dataset[i].clone()
            # subtract 1 from label to match classifier labels (0-4)
            if d.y.item() > 0:
                d.y = d.y - 1
            anomaly_test_data.append(d)
            # shift label to match classifier labels (0-4)
            anomaly_test_idx.append(i)

def _train_and_eval_single_sync(dataset, y_counts):
    """
    This function runs in a separate thread. 
    It handles the blocking CPU-bound tasks: Training, Eval, Plotting.
    """
    log.info(f"Label distribution: " + ", ".join([f"{index_to_ylabel(i)}: {count}" for i, count in enumerate(y_counts)]))
    
    config = {
        "hidden_dim": 84,
        "dropout": 0.5,
        "learning_rate": 0.001,
        "weight_decay": 5e-4,
        "num_layers": 3,
        "early_stopping_patience": 40,
        "early_stopping_min_delta": 5e-4,
        "max_epochs": 150,
        "num_heads": 4
    }

    # Initialize Model
    m = gnn_het_single.GNNHeteroClassifierModel(config=config, metadata=dataset[0].metadata())
    
    # Build Loaders (Note: Use num_workers=0 inside here as it's now in a thread, safe from asyncio)
    (train_loader, val_loader), final_test_loader = m.build_data_loaders(dataset)

    # Train
    _, criterion, _ = m.fit_model(train_loader, val_loader, config)
    
    # Validate
    val_metrics = m.evaluate_model(val_loader)
    log.info(f"Validation Metrics: {val_metrics}")
    
    # Test
    y_all, y_pred_all = [], []
    y_all, y_pred_all = m.test_model(final_test_loader, test_description="Final Test Set - GNNHeteroSingleModel")
    
    # Metrics & Export
    m.get_label_metrics(y_all, y_pred_all, None, export_results=True, is_final_test=True)
    final_report_dict = classification_report(y_all, y_pred_all, target_names=y_labels, zero_division=0, output_dict=True)
    report_df = pd.DataFrame(final_report_dict).transpose()
    report_df.to_csv(f"./exports/results/classification_report_singlemodel_GNNHET_FINAL.csv")
    
    # Plotting
    cm = confusion_matrix(y_all, y_pred_all)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"./exports/images/gnn_het_singlemodel_confusion_matrix_FINAL.png")
    plt.close()
    
    log.info("Final classification report and confusion matrix saved.")
    return True

async def handle_message(request):
    async with SEM:
        await _process_request_gnn_het(request)





async def _process_request_gnn_het(request: DetectAnomalyRequest):
    """Train model based on src/models/gnn_het.py GNNHeteroAnomalyDetector class which uses GNNHeteroModel"""
    
    # get all snapshots
    snapshots = await SnapshotRepository.get_all_snapshots(request.duration, request.system_id)
    #snapshots = await SnapshotRepository.get_all_snapshots_phys_only(request.duration, request.system_id)

    # Convert each snapshot to PyG data
    temp_data = _het_snapshots_to_dataset(snapshots)
    dataset = []
    y_counts = np.zeros(len(y_labels), dtype=int)
    _write_name = True
    #visualize_features_distribution(temp_data)
    
    for data in temp_data:
        
        write_hetero_feature_mappings(data, data.snapshot_id, write_name=_write_name)
        _write_name = False
        # get y Labels from AggregateRepository
        log.info(f"Getting labels for snapshot ID: {data.snapshot_id}")
        labels = await AggregateRepository.get_labels_for_snapshot(data.snapshot_id, request.system_id, request.duration)
        log.info(f"Snapshot ID: {data.snapshot_id} has labels: {labels}")
        mapped_labels = get_mapped_anomaly_labels(labels)
        logging.info(f"Mapped labels for snapshot ID: {data.snapshot_id} are: {mapped_labels}")
        
        data.y = torch.tensor([max(mapped_labels)], dtype=torch.long)
        y_counts[max(mapped_labels)] += 1
        dataset.append(data)

        # need_copy = mapped_labels and len(mapped_labels) > 1
        # if need_copy:
            
        #     # for each mapped label except 0, create a new data copy if needed
        #     for label in mapped_labels:
        #         if label != 0:                
        #             if need_copy:
        #                 newdata = data.clone()
        #             else:
        #                 newdata = data
        #             newdata.y = torch.tensor([label], dtype=torch.float32)
        #             y_counts[label] += 1
        #             dataset.append(newdata)
        #             need_copy = True
        # else:
        #     # single label case, add directly
        #     data.y = torch.tensor([mapped_labels[0]], dtype=torch.float32)
        #     y_counts[mapped_labels[0]] += 1
        #     dataset.append(data)
        
    
    log.info(f"Prepared dataset with {len(dataset)} graphs")
    log.info(f"Label distribution: " + ", ".join([f"{index_to_ylabel(i)}: {count}" for i, count in enumerate(y_counts)]))
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, _train_and_eval_single_sync, dataset, y_counts)
    # config = {
    #     "hidden_dim": 128,
    #     "dropout": 0.4,
    #     "learning_rate": 0.0005,
    #     "weight_decay": 1e-5,
    #     "num_layers": 3,
    #     "early_stopping_patience": 50,
    #     "early_stopping_min_delta": 5e-4,
    #     "max_epochs": 300,
    #     "num_heads": 4
    # }

    # m = gnn_het_single.GNNHeteroClassifierModel(config=config, metadata=dataset[0].metadata())
    # # split dataset into train/val/test
    # (train_loader, val_loader), final_test_loader = m.build_data_loaders(dataset)

    # # train and validate detector model
    # _, criterion, _ = m.fit_model(train_loader, val_loader, config)
    # val_metrics = m.evaluate_model(val_loader)
    # log.info(f"Validation Metrics: {val_metrics}")
    # # perform final test on combined model
    # y_all, y_pred_all = [], []
    # y_all, y_pred_all = m.test_model(final_test_loader, test_description="Final Test Set - GNNHeteroSingleModel")
    # m.get_label_metrics(y_all, y_pred_all, None, export_results=True, is_final_test=True)
    # final_report_dict = classification_report(y_all, y_pred_all, target_names=y_labels, zero_division=0, output_dict=True)
    # report_df = pd.DataFrame(final_report_dict).transpose()
    # report_df.to_csv(f"./exports/results/classification_report_singlemodel_GNNHET_FINAL.csv")
    # # save confusion matrix image
    # cm = confusion_matrix(y_all, y_pred_all)
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.savefig(f"./exports/images/gnn_het_singlemodel_confusion_matrix_FINAL.png")
    # plt.close()
    # log.info("Final classification report saved.")

    # create model object
    #self, config: Dict[str, Any], in_channels: int=51, out_channels: int=6

    # detector_model = gnn_het_bin.GNNHeteroAnomalyDetectionModel(config=config, metadata=dataset[0].metadata())
    # classify_model = gnn_het.GNNHeteroClassifierModel(config=classify_config,
    #                                metadata=dataset[0].metadata())

    # # split dataset into train/val/test
    # bin_data, anom_data, final_test_data = detector_model.build_data_loaders(dataset)
    

    # # train and validate detector model
    # _, criterion, _ = detector_model.fit_model(bin_data[0], bin_data[1], config)
    # val_metrics = detector_model.evaluate_model(bin_data[1])
    # log.info(f"Binary Detector Validation Metrics: {val_metrics}")

    # # train and validate classifier model
    # logging.info("Training Anomaly Classifier Model...")
    # _, anom_criterion, _ = classify_model.fit_model(anom_data[0], anom_data[1], config)
    # logging.info("Anomaly Classifier Model Training Completed.")
    # val_metrics = classify_model.evaluate_model(anom_data[1])
    # log.info(f"Anomaly Classifier Validation Metrics: {val_metrics}")

    # # perform final test on combined model
    # y_all, y_pred_all = [], []
    # y_all_bin, y_pred_all_bin = detector_model.test_model(final_test_data, test_description="Final Test Set - Binary Detector")

    # # filter final_test_data and create new DataLoader for only detected anomalies
    # anomaly_test_data = []
    # anomaly_test_idx = []
    # for i, label in enumerate(y_pred_all_bin):
    #     if label == 1 and y_all_bin[i] > 0:  # only test samples predicted as anomaly and true label is anomaly
    #         d = final_test_data.dataset[i].clone()
    #         # subtract 1 from label to match classifier labels (0-4)
    #         if d.y.item() > 0:
    #             d.y = d.y - 1
    #         anomaly_test_data.append(d)
    #         # shift label to match classifier labels (0-4)
    #         anomaly_test_idx.append(i)

    # y_all_anom, y_pred_all_anom = [], []
    # if anomaly_test_data:
    #     anomaly_test_loader = DataLoader(anomaly_test_data, batch_size=32, shuffle=False)
    #     # Run the classifier model on the anomaly test set
    #     y_all_anom, y_pred_all_anom = classify_model.test_model(anomaly_test_loader, test_description="Final Test Set - Anomaly Classifier")
    
    # y_all = [data.y.item() for data in final_test_data.dataset]
    # # combine binary and anomaly predictions
    # # for y_pred_all_bin, if 0 then final is 0, if 1 then final is from y_pred_all_anom
    # anom_idx = 0
    # for i, label in enumerate(y_pred_all_bin):
    #     if label == 0 or (label == 1 and y_all_bin[i] == 0):
    #         y_pred_all.append(label)
    #     else:
    #         y_pred_all.append(y_pred_all_anom[anom_idx] + 1)  # shift by 1 to match classifier labels
    #         anom_idx += 1
    # logging.info(f"Combined final predictions: {y_pred_all}")
    # logging.info("Generating final classification report for combined model...")
    # classify_model.get_label_metrics(y_all, y_pred_all, None, export_results=True, is_final_test=True)
    # logging.info("Generated final classification report and saved to CSV.")
    # final_report_dict = classification_report(y_all, y_pred_all, target_names=y_labels, zero_division=0, output_dict=True)
    # report_df = pd.DataFrame(final_report_dict).transpose()
    # report_df.to_csv(f"./exports/results/classification_report_FINAL_COMBINED.csv")
    # # save confusion matrix image
    # logging.info("Generating confusion matrix for final combined model.")
    # cm = confusion_matrix(y_all, y_pred_all)
    # # add timestamp to filename to avoid overwriting
    # plt.figure(figsize=(10, 7))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.savefig(f"./exports/images/gnn_het_detection_confusion_matrix_FINAL_COMBINED.png")
    # plt.close()
    # logging.info("Final combined confusion matrix saved.")

    # #xai results
    # logging.info("Generating XAI explanations for final test set...")
    # xai_results = []
    # # explain only the detected anomalies
    # for i, data in enumerate(final_test_data.dataset):
    #     is_anomaly = y_pred_all_bin[i] == 1
    #     if is_anomaly:
    #         log.info(f"Generating explanation for snapshot ID: {data.snapshot_id} (Index: {i})")
    #         explanation = classify_model.explain_with_captum(data, target_class_idx=int(data.y.item()))
    #         xai_results.append(explanation)

    #         # now add explanations for each anomaly class predicted
    #         for class_idx in range(1, 5):  # Assuming 4 anomaly classes
    #             log.info(f"Generating explanation for snapshot ID: {data.snapshot_id} for class index: {class_idx} (Index: {i})")
    #             explanation = classify_model.explain_with_captum(data, target_class_idx=class_idx)
    #             xai_results.append(explanation)
    # log.info(f"XAI explanations generated for {len(xai_results)} samples.")
    # with open(f"./exports/results/gnn_het_xai_explanations_final_test.json", "w") as f:
    #     json.dump(xai_results, f, indent=4)
    



    # persist results to export folder for analysis as csv file


def _het_snapshots_to_dataset(snapshots):
    dataset = []
    write_name = True
    for snapshot in snapshots:
        try:
            pyg_data = to_pyg_hetero_data(snapshot, write_name=write_name)
            dataset.append(pyg_data)
            write_name = False  # only write feature names once
        except Exception as e:
            log.error(f"Error converting snapshot to PyG hetero data: {e}")
    return dataset

def get_mapped_anomaly_labels(labels,is_binary: bool=False):
    """Map string labels to indices based on y_labels"""
    if is_binary:
            is_normal = all(label.lower() == 'normal' for label in labels)
            return [0 if is_normal else 1]
    else:
        mapped = []
        for label in labels:
            logging.info(f"Mapping label: {label}")
            if label.lower() in y_labels:
                mapped.append(ylabel_to_index(label.lower()))
            else:
                mapped.append(ylabel_to_index('anomaly'))  # default to anomaly if unknown
        return mapped  # return highest severity label only


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