
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class ScadaAggregate(BaseModel):
    bucket: datetime = Field(..., description="Start time of the aggregation bucket")
    duration: Optional[int] = Field(..., description="Duration of the aggregation bucket in seconds")
    system_id: str = Field(..., description="Identifier for the system")
    protocol: str = Field(..., description="Network protocol used")
    avg_size: float = Field(..., description="Average size of packets in the bucket")
    source_total_packets: int = Field(..., description="Total packets sent from source")
    destination_total_packets: int = Field(..., description="Total packets sent to destination")
    min_size: int = Field(..., description="Minimum packet size in the bucket")
    max_size: int = Field(..., description="Maximum packet size in the bucket")
    num_connections: int = Field(..., description="Number of unique connections in the bucket")
    source_ip: Optional[str] = Field(..., description="Source IP address")
    source_port: Optional[int] = Field(..., description="Source port number")
    source_mac: Optional[str] = Field(..., description="Source MAC address")
    destination_ip: Optional[str] = Field(..., description="Destination IP address")
    destination_port: Optional[int] = Field(..., description="Destination port number")
    destination_mac: Optional[str] = Field(..., description="Destination MAC address")
    source_key: str = Field(None, description="Unique key for the source endpoint")
    destination_key: str = Field(None, description="Unique key for the destination endpoint")
