from pydantic import Field
from typing import List, Optional

from core.domain.models.base import Base


class Snapshot(Base):
    """Where was I? Keep this stuff in context when I come back."""

    body: str
    label: Optional[str] = None
    session: Optional[str] = None
    questions: List[str] = Field(default_factory=list)
    designs: List[str] = Field(default_factory=list)
    offers: List[str] = Field(default_factory=list)
    engagements: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    builds: List[str] = Field(default_factory=list)
    state_machines: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    pipelines: List[str] = Field(default_factory=list)
