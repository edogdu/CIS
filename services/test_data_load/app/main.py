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

app = FastAPI()
kafka_bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS','kafka:9092')
kafka_phys_topic = os.getenv('KAFKA_PHYS_TOPIC','Data.Raw.Physical')
kafka_scada_topic = os.getenv('KAFKA_SCADA_TOPIC','Data.Raw.Scada')
kafka_generate_graph_topic = os.getenv('KAFKA_GENERATE_GRAPH_TOPIC','Data.Graphs')
data_dir = os.getenv('DATA_DIR','/app/data')
schema_dir = os.getenv('SCHEMA_DIR','/app/schemas')

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/aggregate_logs")
async  def aggregate_logs():
    print("Starting log aggregation...")
    await asyncio.gather(
        aggregate_scada_logs(),
        aggregate_physical_logs()
    )
    return {"message": "completed"}

async def aggregate_scada_logs():
       
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    try:
        path = f"{data_dir}/testbed_system_1/Network/csv/*.csv"
        print(f"Looking for SCADA files in path: {path}")
        all_files = glob.glob(path)
        print(f"Found {len(all_files)} files.")
        for filename in all_files:
            print(f"Processing file: {filename}")
            df = pd.read_csv(filename)            
            df.columns = df.columns.str.strip()  # Strip whitespace from column names                        
            df.fillna({
                'sport': 0,
                'dport': 0,
                'n_pkt_src': 0,
                'n_pkt_dst': 0,
                'size': 0
            }, inplace=True)  # Check for NaN in critical columns
            df = df.where(pd.notnull(df), None)  # Replace NaN with None
            #Time,mac_s,mac_d,ip_s,ip_d,sport,dport,proto,flags,size,
            # modbus_fn,n_pkt_src,n_pkt_dst,modbus_response,label_n,label
            
            messages = ScadaLog.load_from_dataframe(system_id='testbed_system_1', df=df)
            print(f"Generated {len(messages)} messages from file {filename}")
            for msg in messages:
                await producer.send(topic=kafka_scada_topic, value=msg.json().encode('utf-8'))
                #await producer.flush()
    except Exception as e:
        print(f"Error occurred: {e}")        
    finally:
        await producer.stop()
    return {"message": "completed"}

async def aggregate_physical_logs():
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    try:
        path = f"{data_dir}/testbed_system_1/Physical/csv/*.csv"
        print(f"Looking for files in path: {path}")
        all_files = glob.glob(path)
        for filename in all_files:
            print(f"Processing file: {filename}")
            df = pd.read_csv(filename)
            df.columns = df.columns.str.strip()  # Strip whitespace from column names
            df = df.where(pd.notnull(df), None)  # Replace NaN with None
            messages = PhysicalLog.load_from_dataframe(system_id='testbed_system_1', df=df)
            print(f"Generated {len(messages)} messages from file {filename}")
            for msg in messages:
                await producer.send(topic=kafka_phys_topic, value=msg.json().encode('utf-8'))
                #await producer.flush()
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        await producer.stop()
    return {"message": "completed"}

@app.post("/generate_graph")
async  def generate_graph(request: GenerateGraphRequest):
    producer = AIOKafkaProducer(bootstrap_servers=[kafka_bootstrap_servers])
    await producer.start()
    msg = request
    await producer.send(topic=kafka_generate_graph_topic, value=msg.json().encode('utf-8'))
    await producer.flush()
    await producer.stop()
    return {"message": "Graph generation request sent"}
    
@app.get("/export_aggregate_data")
async def export_aggregate_data():
    with connect() as conn:            
        query = "SELECT * FROM scada_resolved_agg_30s ORDER BY bucket;"
        df = pd.read_sql_query(query, conn)
        df.to_csv(f"{data_dir}/testbed_system_1/aggregated.csv",index=False)
