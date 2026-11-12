from typing import List, Optional
from pydantic import BaseModel, Field
import uuid

class AtomicClaim(BaseModel):
    id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:8]}")
    text: str = Field(..., description="The self-contained, decontextualized atomic claim statement")
    parent_sentence: str = Field(..., description="The original sentence from which this claim was extracted")
    entities: List[str] = Field(default_factory=list, description="Named entities identified in the claim")
    context_brackets: Optional[str] = Field(None, description="Context inferred from surrounding text (e.g. '[in 2024]')")
    is_verifiable: bool = Field(True, description="Whether this claim is factually verifiable (vs subjective opinion)")
    ambiguity_flag: bool = Field(False, description="Flagged if claim cannot be safely disambiguated")
    is_underspecified: bool = Field(False, description="Flagged if subject/entity is an underspecified mononym or missing context")
    ambiguity_reason: Optional[str] = Field(None, description="Explanation of why claim cannot be verified without disambiguation")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
