/* System Flow Graph */
MATCH p=()-[r]->() WHERE type(r) IN ['FEEDS_TO', 'FEEDS_THROUGH'] RETURN p;

/* PLC Controls & Communication */
MATCH p=()-[r]->() WHERE type(r) IN [':ISSUES_COMMAND_TO', 'CONTROLS'] RETURN p

/* Network Communication */
MATCH p=()-[r:HAS_ENDPOINT]->() RETURN p

/* Get Snapshot Graph for Model */
MATCH p=(a:Asset)-[r:HAS_MEASUREMENT]->(m:Measurement{snapshot_id:'testbed_system_1_2021-04-09 12:19:00+00:00'}) 
MATCH n=(src:Endpoint)-[:INITIATES]->(c:Connection{snapshot_id: 'testbed_system_1_2021-04-09 12:19:00+00:00'})
MATCH n2=(c)-[:TERMINATES_AT]->(e2:Endpoint)
OPTIONAL MATCH ctr=(a:Asset)-[:HAS_ENDPOINT]->(e)
OPTIONAL MATCH ctr2=(a2:Asset)-[:HAS_ENDPOINT]->(e2)
OPTIONAL MATCH rf=(a)-[:READS_FROM]->()
RETURN p,n,n2,ctr,ctr2, rf

/* Clear The Database */
MATCH (n) DETACH DELETE n


MATCH (n)-[r]->(n2) WHERE type(r) IN ['ISSUES_COMMAND_TO', 'CONTROLS','HAS_ENDPOINT','FEEDS_THROUGH', 'FEEDS_TO','READS_FROM', 'SENSOR_ON'] 
MATCH (n2)-[r2]-(n3) WHERE type(r2) IN ['TERMINATES_AT','INITIATES','HAS_MEASUREMENT']
AND n3.snapshot_id = 'testbed_system_1_30s_2022-02-21 15:06:00+00:00'
RETURN n,r,n2,r2,n3