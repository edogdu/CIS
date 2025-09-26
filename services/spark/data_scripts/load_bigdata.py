import os, glob
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, TimestampType, BooleanType
)
from pyspark.sql.functions import (
    col, concat, lit, when, to_timestamp, coalesce, create_map, date_format, trim
    , regexp_replace, split, element_at
)

kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "cis-kafka:9092")
num_partitions = os.getenv("KAFKA_NUM_PARTITIONS", 16)
phys_data_path = os.getenv("PHYS_DATA_DIR","/data/testbed system_1/Physical")
phys_utf16_data_path = os.getenv("PHYS_UTF16_DATA_DIR","/data/testbed system_1/Physical/utf16")
phys_utf16_data_dest_path = os.getenv("PHYS_UTF16_DATA_DEST_DIR","/data/testbed system_1/Physical")
network_data_path = os.getenv("SCADA_DATA_DIR","/data/testbed system_1/Network/csv")
system_id = "testbed_system_1"
network_topic = os.getenv("KAFKA_SCADA_TOPIC","Data.Raw.Scada")
phys_topic = os.getenv("KAFKA_PHYS_TOPIC","Data.Raw.Physical")

network_data_schema = StructType([
    StructField("Time", TimestampType(), True),
    StructField("mac_s", StringType(), True),
    StructField("mac_d", StringType(), True),
    StructField("ip_s", StringType(), True),
    StructField("ip_d", StringType(), True),
    StructField("sport", IntegerType(), True),
    StructField("dport", IntegerType(), True),
    StructField("proto", StringType(), True),
    StructField("flags", StringType(), True),
    StructField("size", IntegerType(), True),
    StructField("modbus_fn", StringType(), True),
    StructField("n_pkt_src", IntegerType(), True),
    StructField("n_pkt_dst", IntegerType(), True),
    StructField("modbus_response", StringType(), True),
    StructField("label_n", IntegerType(), True),
    StructField("label", StringType(), True)
])

phys_data_schema = StructType([
    #Time	Tank_1	Tank_2	Tank_3	Tank_4	Tank_5	Tank_6	Tank_7	Tank_8	Pump_1	Pump_2	Pump_3	Pump_4	Pump_5	Pump_6	Flow_sensor_1	Flow_sensor_2	Flow_sensor_3	Flow_sensor_4	Valv_1	Valv_2	Valv_3	Valv_4	Valv_5	Valv_6	Valv_7	Valv_8	Valv_9	Valv_10	Valv_11	Valv_12	Valv_13	Valv_14	Valv_15	Valv_16	Valv_17	Valv_18	Valv_19	Valv_20	Valv_21	Valv_22	Label_n	Label
    StructField("Time", TimestampType(), True),
    StructField("Tank_1", IntegerType(), True), #8
    StructField("Tank_2", IntegerType(), True),
    StructField("Tank_3", IntegerType(), True),
    StructField("Tank_4", IntegerType(), True),
    StructField("Tank_5", IntegerType(), True),
    StructField("Tank_6", IntegerType(), True),
    StructField("Tank_7", IntegerType(), True),
    StructField("Tank_8", IntegerType(), True),
    StructField("Pump_1", BooleanType(), True), #6
    StructField("Pump_2", BooleanType(), True),
    StructField("Pump_3", BooleanType(), True),
    StructField("Pump_4", BooleanType(), True),
    StructField("Pump_5", BooleanType(), True),
    StructField("Pump_6", BooleanType(), True),
    StructField("Flow_sensor_1", IntegerType(), True), #4
    StructField("Flow_sensor_2", IntegerType(), True),
    StructField("Flow_sensor_3", IntegerType(), True),
    StructField("Flow_sensor_4", IntegerType(), True),
    StructField("Valv_1", BooleanType(), True), #22
    StructField("Valv_2", BooleanType(), True),
    StructField("Valv_3", BooleanType(), True),
    StructField("Valv_4", BooleanType(), True),
    StructField("Valv_5", BooleanType(), True),
    StructField("Valv_6", BooleanType(), True),
    StructField("Valv_7", BooleanType(), True),
    StructField("Valv_8", BooleanType(), True),
    StructField("Valv_9", BooleanType(), True),
    StructField("Valv_10", BooleanType(), True),
    StructField("Valv_11", BooleanType(), True),
    StructField("Valv_12", BooleanType(), True),
    StructField("Valv_13", BooleanType(), True),
    StructField("Valv_14", BooleanType(), True),
    StructField("Valv_15", BooleanType(), True),
    StructField("Valv_16", BooleanType(), True),
    StructField("Valv_17", BooleanType(), True),
    StructField("Valv_18", BooleanType(), True),
    StructField("Valv_19", BooleanType(), True),
    StructField("Valv_20", BooleanType(), True),
    StructField("Valv_21", BooleanType(), True),
    StructField("Valv_22", BooleanType(), True),
    StructField("Label_n", IntegerType(), True),
    StructField("Label", StringType(), True)
])

def preprocess_phys_data(df):
    double_cols = ["Tank_1","Tank_2","Tank_3","Tank_4","Tank_5","Tank_6","Tank_7","Tank_8"
                   ,"Flow_sensor_1","Flow_sensor_2","Flow_sensor_3","Flow_sensor_4"]
    
    bool_cols =   ["Pump_1","Pump_2","Pump_3","Pump_4","Pump_5","Pump_6"
                   ,"Valv_1","Valv_2","Valv_3","Valv_4","Valv_5","Valv_6"
                   ,"Valv_7","Valv_8","Valv_9","Valv_10","Valv_11","Valv_12"
                   ,"Valv_13","Valv_14","Valv_15","Valv_16","Valv_17","Valv_18"
                   ,"Valv_19","Valv_20","Valv_21","Valv_22"]
        
    all_cols = double_cols + bool_cols
    df2 = df.select("Time", "Label_n", "Label",
                    *[col(c).cast("double").alias(c) for c in double_cols],
                    *[when(col(c) == True, lit(1.0))
                      .when(col(c) == False, lit(0.0))
                      .otherwise(lit(None).cast("double")).alias(c) for c in bool_cols])
    
    pairs = ", ".join([f"'{c}',`{c}`" for c in all_cols])
    measure_types =  {"Tank": "Pressure", "Pump": "state", "Flow_sensor": "Value", "Valv": "state"}
    asset_type = split(col("measurement_id"), "_").getItem(0)

    unpivot_df = df2.selectExpr(
        "Time",
        "Label_n",
        "Label",
        f"stack({len(all_cols)}, {pairs}) as (measurement_id, measure_value)"
    )
    
    
    #testbed_system_1_Tank_Tank_1_Pressure
    return unpivot_df.withColumn("id", concat(lit(f"{system_id}_"), date_format(to_timestamp(regexp_replace(trim(col("Time")),"^\uffef",""), "dd/MM/yyyy HH:mm:ss"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"), lit("_"), col("measurement_id"))
                                 ).withColumn("system_id", lit(system_id)
                                 ).withColumn("log_ts", date_format(to_timestamp(regexp_replace(trim(col("Time")),"^\uffef",""), "dd/MM/yyyy HH:mm:ss"), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")
                                 ).withColumn("measurement_id", concat(lit(f"{system_id}_")
                                                                       ,when(asset_type == "Flow", lit("Flow_sensor"))
                                                                          .otherwise(asset_type)
                                                                       , lit("_")
                                                                       , col("measurement_id")
                                                                       , lit("_")
                                                                       , when(asset_type == "Tank", lit("Pressure"))
                                                                        .when(asset_type == "Pump", lit("State"))
                                                                        .when(asset_type == "Flow", lit("Value"))
                                                                        .when(asset_type == "Valv", lit("State"))
                                                                        .otherwise(lit("unknown"))
                                                                       )
                                 ).withColumn("measure_value", col("measure_value").cast("double")
                                 ).withColumn("attributes", create_map(lit("Label_n"), col("Label_n"), lit("Label"), col("Label"))
                                 ).drop("Label_n").drop("Label").drop("Time")



def preprocess_network_data(df):
    print("processing network data...")
    return df.withColumn("id", 
                         concat(lit(f"{system_id}_")
                                ,col("Time")
                                ,lit("_")
                                ,col("ip_s")
                                ,lit("_")
                                ,col("mac_s")
                                ,lit("_")
                                ,col("ip_d")
                                ,lit("_")
                                ,col("mac_d"))
            ).withColumn("system_id"
                         , lit(system_id)    
            ).withColumn("log_ts"
                         , date_format(to_timestamp(col("Time")), "yyyy-MM-dd'T'HH:mm:ss.SSSSSS")
            ).withColumn("source_ip"
                         , col("ip_s")
            ).withColumn("source_port"
                         , col("sport")
            ).withColumn("source_mac"
                         , col("mac_s")
            ).withColumn("destination_ip"
                         , col("ip_d")
            ).withColumn("destination_port"
                         , col("dport")
            ).withColumn("destination_mac"
                         , col("mac_d")
            ).withColumn("protocol"
                         , coalesce(col("proto"), lit(""))
            ).withColumn("modbus_func"
                         , col("modbus_fn")
            ).withColumn("source_number_packets"
                         , coalesce(col("n_pkt_src"), lit(0))
            ).withColumn("destination_number_packets"
                         , coalesce(col("n_pkt_dst"), lit(0))
            ).withColumn("total_size"
                         , coalesce(col("size"), lit(0))
            ).withColumn("attributes"
                         , create_map(
                             lit("flags"), col("flags"),
                             lit("modbus_response"), col("modbus_response"),
                             lit("label_n"), col("label_n"),
                             lit("label"), col("label")
                         ))

def send_to_kafka(df, topic):
    print(f"sending to kafka.: {df.take(100)}")
    df.selectExpr("CAST(id AS STRING) AS key", "to_json(struct(*)) AS value") \
    .write \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
    .option("topic", topic) \
    .save()
    print("complete...")

def convert_phys_utf16_to_utf8():
    print("converting to utf-8...")
    path = f"{phys_utf16_data_path}/*.csv"
    print(f"looking for files in path: {path}")
    files = glob.glob(path)
    print(f"found {len(files)} files to convert...")
    for filename in files:        
        print(f"processing file: {filename}")
        src = f"{phys_utf16_data_path}/{os.path.basename(filename)}"
        dest = f"{phys_utf16_data_dest_path}/{os.path.basename(filename)}"
        with open(src, "r", encoding="utf-16") as fsrc:
            with open(dest, "w", encoding="utf-8") as fdest:
                print(f"writing to file: {dest}")
                for line in fsrc:
                    fdest.write(line)
        

if __name__ == "__main__":
    spark = SparkSession.builder.appName("CISBulkDataLoader").getOrCreate()
    try:
        #network
        network_df = spark.read.csv(f"{network_data_path}/*.csv"
                                    ,header=True
                                    ,schema=network_data_schema
                                    ,nullValue="NaN")
        final_network_df = preprocess_network_data(network_df)
        send_to_kafka(final_network_df, network_topic)
        print("Network data load complete.")

        #physical
        # TODO: Need to fix mapping before uncommenting and loading physical data
        convert_phys_utf16_to_utf8()
        
        phys_df_utf8 = spark.read.options(delimiter="\t", header=True, schema=phys_data_schema, nullValue="NaN", encoding="UTF-8").csv(f"{phys_data_path}/phy_*.csv")
        final_utf8phys_df = preprocess_phys_data(phys_df_utf8)
        #print(f"final_utf8phys_df: {final_utf8phys_df.take(100)}")
        send_to_kafka(final_utf8phys_df, phys_topic)
        print("Physical data load complete.")
    finally:
        spark.stop()
        print("Spark session stopped.")
        print("Data load process complete.")