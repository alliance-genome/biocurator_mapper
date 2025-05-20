from typing import List, Optional

from pydantic import BaseModel, Field


class ResolveRequest(BaseModel):
    passage: str
    ontology_name: str = Field(..., description="Ontology short name, e.g. GO")


class OntologyTerm(BaseModel):
    id: str
    name: str
    definition: Optional[str] = None


class ResolveResponse(BaseModel):
    best_match: OntologyTerm | None = None
    confidence: Optional[float] = None
    reason: Optional[str] = None
    alternatives: Optional[List[OntologyTerm]] = None
    error: Optional[str] = None


class OntologyUpdateRequest(BaseModel):
    ontology_name: str
    source_url: str
