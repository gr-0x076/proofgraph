from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import datetime

class ProvEntity(BaseModel):
    id: str
    type: str  # e.g., "claim", "evidence_passage", "document", "verdict"
    attributes: Dict[str, Any] = Field(default_factory=dict)

class ProvActivity(BaseModel):
    id: str
    type: str  # e.g., "decomposition", "hybrid_retrieval", "nli_verification"
    started_at: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    ended_at: Optional[datetime.datetime] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ProvAgent(BaseModel):
    id: str
    type: str  # e.g., "model", "search_engine", "user"
    name: str
    version: Optional[str] = None

class ProvRelation(BaseModel):
    subject_id: str
    predicate: str  # e.g., "used", "wasGeneratedBy", "wasDerivedFrom", "wasAttributedTo", "contradicts", "disagrees_with"
    object_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
