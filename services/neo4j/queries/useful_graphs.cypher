/* System Flow Graph */
MATCH p=()-[r]->() WHERE type(r) IN ['FEEDS_TO', 'FEEDS_THROUGH'] RETURN p;

/* PLC Controls & Communication */
MATCH p=()-[r]->() WHERE type(r) IN [':ISSUES_COMMAND_TO', 'CONTROLS'] RETURN p

/* Network Communication */
MATCH p=()-[r:HAS_ENDPOINT]->() RETURN p