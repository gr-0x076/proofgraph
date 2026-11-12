from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class VeracityLabel(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    ENTITY_UNDERSPECIFIED = "ENTITY_UNDERSPECIFIED"

class EvidenceTier(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    UNDERSPECIFIED = "UNDERSPECIFIED"
    CONFLICTING = "CONFLICTING"

class ClaimVerdict(BaseModel):
    id: str = Field(default_factory=lambda: f"verdict_{uuid.uuid4().hex[:8]}")
    claim_id: str
    label: VeracityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence_tier: EvidenceTier = Field(default=EvidenceTier.WEAK, description="Discrete qualitative evidence quality tier")
    justification: str
    supporting_evidence_ids: List[str] = Field(default_factory=list)
    contradicting_evidence_ids: List[str] = Field(default_factory=list)
    refnli_passed: bool = Field(True, description="Whether evidence passed the context alignment filter")
    credibility_adjusted: bool = Field(False, description="Whether source credibility altered the raw verdict")

def determine_evidence_tier(
    label: VeracityLabel,
    confidence: float,
    is_conflicting: bool = False,
    is_underspecified: bool = False
) -> EvidenceTier:
    """Maps continuous confidence and verifier stance to a qualitative evidence tier."""
    if is_underspecified or label == VeracityLabel.ENTITY_UNDERSPECIFIED:
        return EvidenceTier.UNDERSPECIFIED
    if is_conflicting:
        return EvidenceTier.CONFLICTING
    if label in (VeracityLabel.SUPPORTED, VeracityLabel.CONTRADICTED):
        if confidence >= 0.85:
            return EvidenceTier.STRONG
        elif confidence >= 0.65:
            return EvidenceTier.MODERATE
        else:
            return EvidenceTier.WEAK
    return EvidenceTier.WEAK
