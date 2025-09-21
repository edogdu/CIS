# CIS
Critical Infrastructure Security

## Software Requirements 
* Docker Desktop - https://www.docker.com/products/docker-desktop/

## Usage
* First, pull the repository into your file system 
```bash
$  git clone https://github.com/edogdu/CIS.git ./CIS
```

* Next, switch to root directory of project
```bash
$  cd CIS
```

* Lastly, run docker-compose to start project
```bash
$  docker-compose up --build
```

* To shutdown the UCKG, follow these steps
    - type Ctrl+C to stop server
    - use docker-compose to clean up images
```bash
$  docker-compose down
```

## Notes about Ports
* The following have had their ports mapped to non-standard ports to avoid collision with UCKG
  * **Neo4j**
    * 7485 for web ui
    * 7698 for bolt
  * **Postgres**
    * 5445

## Notes about Configuring Spark and Kafka
Depending on the hosting machine's available resources, the following services in docker-compose.yml may need to be adjusted accordingly.
* Spark - https://spark.apache.org/docs/latest/hardware-provisioning.html
* Kafka - https://docs.redhat.com/en/documentation/red_hat_streams_for_apache_kafka/2.9/html/kafka_configuration_tuning/con-broker-config-properties-str#improving_request_handling_throughput_by_increasing_i_o_threads

```yaml
cis-spark-worker:
	SPARK_WORKER_MEMORY: "16G"
	SPARK_WORKER_CORES: "8"
	
cis-load-bigdata:
	entrypoint:
    - "spark.executor.instances=2"
    - "--conf"
    - "spark.executor.memory=8g"
    - "--conf"
    - "spark.executor.cores=4"
    - "--conf"
    - "spark.driver.memory=8g"

cis-kafka:
	environment:  
		KAFKA_HEAP_OPTIONS: "-Xms6g -Xmx6g"
		KAFKA_CFG_NUM_NETWORK_THREADS: "6"
		KAFKA_CFG_NUM_IO_THREADS: "16"
		KAFKA_CFG_SOCKET_SEND_BUFFER_BYTES: "1048576"
		KAFKA_CFG_MESSAGE_MAX_BYTES: "2000000"
		KAFKA_CFG_REPLICA_SOCKET_RECEIVE_BUFFER_BYTES: "1048576"

```

## Resources
* testbed_system_1 assumes that you have downloaded A hardware-in-the-loop water distribution testbed (WDT) dataset for cyber-physical security testing dataset from IEEE Dataport. See /data/testbed_system_1/CITATIONS.md for more information.
* Docker has a tendency to have hanging resources that can take up alot of diskspace, I found the following commands useful
    - In Docker Desktop Application, ensure no containers are running
    - In Command Line or Bash run Docker prune commands
    ```bash
    $  docker image prune -f
    ```
    ```bash
    $  docker builder prune
    ```
    - In Docker Desktop Application, navigate to Troubleshoot section (bug button) and select Clean/Purge data
    - If diskspace utilization is not changing after running these steps, try restarting your computer


## License

This project is licensed under the [MIT License](LICENSE.md).
