from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class VeracityLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

class ClaimVerdict(BaseModel):
    id: str = Field(default_factory=lambda: f"verdict_{uuid.uuid4().hex[:8]}")
    claim_id: str
    label: VeracityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    justification: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    refnli_passed: bool = Field(True, description="Whether evidence passed the context alignment filter")
    credibility_adjusted: bool = Field(False, description="Whether source credibility altered the raw verdict")
