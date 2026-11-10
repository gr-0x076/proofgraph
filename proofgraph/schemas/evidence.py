from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

class EvidencePassage(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}")
    text: str = Field(..., description="Exact quoted text of the retrieved evidence passage")
    source_url: Optional[str] = Field(None, description="Source URL or document identifier")
    source_title: Optional[str] = Field("Unknown Source", description="Title or publisher of the source")
    source_domain: Optional[str] = Field("unknown", description="Web domain or authority")
    credibility_score: float = Field(0.5, ge=0.0, le=1.0, description="Domain credibility score (0=unreliable, 1=high authority)")
    relevance_score: float = Field(0.0, ge=0.0, le=1.0, description="Retrieval similarity score")
    metadata: Dict[str, Any] = Field(default_factory=dict)
