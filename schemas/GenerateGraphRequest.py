from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class GenerateGraphRequest(BaseModel):
    begin_ts: datetime = Field(..., description="Start timestamp for the graph data")
    end_ts: datetime = Field(..., description="End timestamp for the graph data")
    system_id: str = Field(..., description="Identifier for the system")