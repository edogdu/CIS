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
