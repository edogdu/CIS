from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pandas import DataFrame

class ScadaLog(BaseModel):
    id: str = Field(..., description="Unique identifier for the log entry")
    system_id: str = Field(..., description="Identifier for the system generating the log")
    log_ts: datetime = Field(..., description="Timestamp of the log entry")
    source_ip: Optional[str] = Field(None, description="Source IP address")
    source_port: Optional[int] = Field(None, description="Source port number")
    source_mac: Optional[str] = Field(None, description="Source MAC address")
    destination_ip: Optional[str] = Field(None, description="Destination IP address")
    destination_port: Optional[int] = Field(None, description="Destination port number")
    destination_mac: Optional[str] = Field(None, description="Destination MAC address")
    protocol: str = Field(..., description="Network protocol used")
    modbus_func: Optional[Any] = Field(None, description="Modbus function code if applicable")
    source_number_packets: Optional[int] = Field(0, description="Number of packets sent from source")
    destination_number_packets: Optional[int] = Field(0, description="Number of packets sent to destination")
    total_size: Optional[int] = Field(0, description="Total size of the packets")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes as a JSON object")

    @staticmethod
    def load_from_dataframe(system_id:str, df: DataFrame) -> list["ScadaLog"]:
        logs = []
        for _, row in df.iterrows():
            print(row.head())
            log = ScadaLog(
                id=f"{system_id}_{row['Time']}_{row['mac_s']}_{row['mac_d']}",
                system_id=system_id,
                log_ts=row['Time'],
                source_ip= None if row.get('ip_s') == 'NaN' else row.get('ip_s'),
                source_port=row.get('sport'),
                source_mac=row.get('mac_s'),
                destination_ip=None if row.get('ip_d') == 'NaN' else row.get('ip_d'),
                destination_port=row.get('dport'),
                destination_mac=row.get('mac_d'),
                protocol=row.get('proto',''),
                modbus_func=row.get('modbus_fn'),
                source_number_packets=row.get('n_pkt_src',0),
                destination_number_packets=row.get('n_pkt_dst',0),
                total_size=row.get('size',0),
                attributes={
                    "flags": row.get('flags'),
                    "modbus_response": row.get('modbus_response'),
                    "label_n": row.get('label_n'),
                    "label": row.get('label')
                }
            )
            logs.append(log)
            print(f"Created log: {log}")
        return logs
    

                        #     id text NOT NULL, --source_id + source_mac + destination_id + destination_mac + log_ts
                        # system_id TEXT NOT NULL,
                        # log_ts TIMESTAMPTZ NOT NULL,
                        # source_ip INET,
                        # source_port INTEGER,
                        # source_mac MACADDR NULL,
                        # destination_ip INET NULL,
                        # destination_port INTEGER,
                        # destination_mac MACADDR NULL,
                        # protocol TEXT NOT NULL,
                        # modbus_func TEXT NULL,
                        # source_number_packets INTEGER DEFAULT 0,
                        # destination_number_packets INTEGER DEFAULT 0,
                        # total_size INTEGER DEFAULT 0,
                        # attributes JSONB,


    #Time, mac_s, mac_d, ip_s, ip_d, sport, dport, proto, flags, size, 
    # modbus_fn, n_pkt_src, n_pkt_dst, modbus_response, label_n, label