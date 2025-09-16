from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pandas import DataFrame

class PhysicalLog(BaseModel):
    id: str = Field(..., description="Unique identifier for the log entry")
    system_id: str = Field(..., description="Identifier for the system generating the log")
    log_ts: datetime = Field(..., description="Timestamp of the log entry")
    measurement_id: str = Field(..., description="Identifier for the measurement type")
    measure_value: float = Field(..., description="Value of the measurement")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Additional attributes as a JSON object")

    @staticmethod
    def load_from_dataframe(system_id:str, df: DataFrame) -> list["PhysicalLog"]:
        logs = []
        for _, row in df.iterrows():
            
            vals = row.to_dict(orient='records')
            for key, val in vals.items():
                if key in ['Time','Label_n','Label']:
                    continue
                log = PhysicalLog(
                    id=f"{system_id}_{row['Time']}_{key}",
                    system_id=system_id,
                    log_ts=row['Time'],
                    measurement_id=key,
                    measure_value=val,
                    attributes={
                        "label_n": row.get('Label_n'),
                        "label": row.get('Label')
                    }
                )
                logs.append(log)
        return logs


        
    

    # id text NOT NULL, -- system_id + time + name
    #                     system_id TEXT NOT NULL,
    #                     log_ts TIMESTAMPTZ NOT NULL,
    #                     measurement_id TEXT NOT NULL REFERENCES phys_measurements_metadata(measurement_id) ON DELETE CASCADE,
    #                     measure_value DOUBLE PRECISION NOT NULL,
    #                     attributes JSONB,
    
    #: ,Tank_1,Tank_2,Tank_3,Tank_4,Tank_5,Tank_6,Tank_7,Tank_8,Pump_1,Pump_2,Pump_3,Pump_4,Pump_5,Pump_6,Flow_sensor_1,Flow_sensor_2,Flow_sensor_3,Flow_sensor_4,Valv_1,Valv_2,Valv_3,Valv_4,Valv_5,Valv_6,Valv_7,Valv_8,Valv_9,Valv_10,Valv_11,Valv_12,Valv_13,Valv_14,Valv_15,Valv_16,Valv_17,Valv_18,Valv_19,Valv_20,Valv_21,Valv_22,Label_n,Label