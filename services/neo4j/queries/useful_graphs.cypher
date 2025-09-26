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
RETURN p,n,n2,ctr,ctr2

/* Clear The Database */
MATCH (n) DETACH DELETE n