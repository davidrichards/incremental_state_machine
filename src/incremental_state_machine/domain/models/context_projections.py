from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Literal, Dict, Any
import subprocess
import json
import logging


class Default(BaseModel):
    """Default projection for Context - shows key fields in a clean format"""

    id: Optional[str] = None
    label: Optional[str] = None
    body: str
    session: Optional[str] = None
    claim: Optional[str] = None
    interest: Optional[str] = None
    doc_chat: Optional[str] = None
    design: Optional[str] = None
    offer: Optional[str] = None
    engagement: Optional[str] = None
    build: Optional[str] = None
    state_machine: Optional[str] = None
    peek: Optional[str] = None

    # Lists with non-empty items only
    questions: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)

    @classmethod
    def from_context(cls, context_data: Dict[str, Any]) -> "Default":
        """Create a default projection from context data"""
        # Filter out empty lists to keep output clean
        filtered_data = {}
        for key, value in context_data.items():
            if isinstance(value, list) and len(value) == 0:
                continue  # Skip empty lists
            filtered_data[key] = value

        return cls(**filtered_data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to exclude empty lists"""
        data = super().model_dump(**kwargs)
        # Remove empty lists for cleaner output
        return {
            k: v for k, v in data.items() if not (isinstance(v, list) and len(v) == 0)
        }


class Discussion(Default):
    """Discussion projection for Context - shows questions with detailed answers"""

    def _question_with_answers(self, question_label: str) -> Dict[str, List[str]]:
        """Retrieve question text and its answers for a given question label"""
        # Use the question command to get answers for a question
        try:
            cmd = ["question", "get", "--label", question_label, "--format", "json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            question_data = json.loads(result.stdout)
            return {
                question_data.get("question", question_label): question_data.get(
                    "answers", []
                )
            }
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logging.warning(
                f"Error retrieving question data for '{question_label}': {e}"
            )
            return {question_label: []}

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include question details"""
        data = super().model_dump(**kwargs)

        # Transform questions into detailed format if they exist
        if "questions" in data and data["questions"]:
            question_details = {}
            for q_label in data["questions"]:
                question_details[q_label] = self._question_with_answers(q_label)
            data["questions"] = question_details

        return data
