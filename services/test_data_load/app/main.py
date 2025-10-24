import asyncio
from aiokafka import AIOKafkaProducer
from fastapi import FastAPI
import pandas as pd
import glob
import os
from schemas.PhysicalLog import PhysicalLog
from schemas.ScadaLog import ScadaLog
from schemas.GenerateGraphRequest import GenerateGraphRequest
from psycopg import sql, connect
from schemas.DetectAnomalyRequest import DetectAnomalyRequest
from schemas.XaiTypes import XaiTypes
from fastapi.responses import RedirectResponse

app = FastAPI()
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS','kafka:9092')
kafka_phys_topic = os.getenv('KAFKA_PHYS_TOPIC','Data.Raw.Physical')
kafka_scada_topic = os.getenv('KAFKA_SCADA_TOPIC','Data.Raw.Scada')
kafka_anomaly_topic = os.getenv('KAFKA_ANOMALY_TOPIC','Anomaly.Predict')
kafka_generate_graph_topic = os.getenv('KAFKA_GENERATE_GRAPH_TOPIC','Data.Graphs')
data_dir = os.getenv('DATA_DIR','/app/exports/data')
schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')
aggregate_table_names = ['phys_agg_30s', 'phys_agg_16s', 'phys_agg_10s', 'scada_agg_30s'
                   , 'scada_agg_16s', 'scada_agg_10s', 'scada_resolved_agg_30s'
                   , 'scada_resolved_agg_16s', 'scada_resolved_agg_10s']
@app.get("/")
async def root():
    return RedirectResponse(url="/docs", status_code=302)


@app.post("/generate_graph")
async  def generate_graph(request: GenerateGraphRequest):
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    msg = request
    await producer.send(topic=kafka_generate_graph_topic, value=msg.json().encode('utf-8'))
    await producer.flush()
    await producer.stop()
    return {"message": "Graph generation request sent"}

@app.post("/train_gnn_model")
async  def train_gnn_model(request: DetectAnomalyRequest):
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    msg = request
    await producer.send(topic=kafka_anomaly_topic, value=msg.json().encode('utf-8'))
    await producer.flush()
    await producer.stop()
    return {"message": "GNN training request sent"}
    
@app.get("/export_aggregate_data")
async def export_aggregate_data():
    with connect() as conn:            
        for table in aggregate_table_names:
            query = f"SELECT * FROM {table} ORDER BY bucket;"
            df = pd.read_sql_query(query, conn)
            df.to_csv(f"{data_dir}/{table}.csv",index=False)
async def export_sys_config():
    with connect() as conn:            
        q1 = "SELECT * FROM assets"
        q2 = "SELECT * FROM network_endpoints"
        q3 = "SELECT * FROM phys_measurements_metadata"
        df = pd.read_sql_query(q1, conn)
        df.to_csv(f"{data_dir}/assets.csv",index=False)

        df = pd.read_sql_query(q2, conn)
        df.to_csv(f"{data_dir}/endpoints.csv",index=False)

        df = pd.read_sql_query(q3, conn)
        df.to_csv(f"{data_dir}/measurement_types.csv",index=False)

@app.get("/refresh_materialized_views")
async def refresh_materialized_views():
    for table in aggregate_table_names:
        with connect() as conn:   
            conn.autocommit = True         
            cur = conn.cursor()
        
            if table.startswith('scada_resolved_'):
                # Refresh materialized view
                cur.execute(sql.SQL(f"REFRESH MATERIALIZED VIEW {table};"))                
            else:
                # Refresh continuous aggregates
                cur.execute(sql.SQL(f"CALL refresh_continuous_aggregate('{table}', NULL, NULL);"))
                
        
        