from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class GenerateGraphRequest(BaseModel):
    start_time: datetime = Field(..., description="Start timestamp for the graph data")
    end_time: datetime = Field(..., description="End timestamp for the graph data")
    system_id: str = Field(..., description="Identifier for the system")