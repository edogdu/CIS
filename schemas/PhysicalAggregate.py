from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class PhysicalAggregate(BaseModel):
    bucket: datetime = Field(..., description="Start time of the aggregation bucket")
    duration: Optional[int] = Field(..., description="Duration of the aggregation bucket in seconds")
    system_id: str = Field(..., description="Identifier for the system")
    asset_id: str = Field(..., description="Identifier for the physical device")
    prop_key: str = Field(..., description="Type of property being measured (e.g., flow rate, pressure)")
    avg_value: float = Field(..., description="Average sensor reading in the bucket")
    min_value: float = Field(..., description="Minimum sensor reading in the bucket")
    max_value: float = Field(..., description="Maximum sensor reading in the bucket")
    num_measurements: int = Field(..., description="Number of readings in the bucket")
    