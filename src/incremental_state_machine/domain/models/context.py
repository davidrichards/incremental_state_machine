from pydantic import Field
from typing import List, Optional

from core.domain.models.base import Base


class Context(Base):
    """Where was I? Keep this stuff in context when I come back."""

    body: str
    label: Optional[str] = None
    session: Optional[str] = None
    claim: Optional[str] = None
    interest: Optional[str] = None
    questions: List[str] = Field(default_factory=list)
    doc_chat: Optional[str] = None
    design: Optional[str] = None
    offer: Optional[str] = None
    engagement: Optional[str] = None
    tasks: List[str] = Field(default_factory=list)
    build: Optional[str] = None
    state_machine: Optional[str] = None
    agents: List[str] = Field(default_factory=list)
    peek: Optional[str] = None
