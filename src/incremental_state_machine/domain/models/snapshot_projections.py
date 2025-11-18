from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Literal, Dict, Any
import subprocess
import json
import logging


class Default(BaseModel):
    """Default projection for Snapshot - shows key fields in a clean format"""

    label: Optional[str] = None
    body: str
    session: Optional[str] = None

    # Lists with non-empty items only
    questions: List[str] = Field(default_factory=list)
    designs: List[str] = Field(default_factory=list)
    offers: List[str] = Field(default_factory=list)
    engagements: List[str] = Field(default_factory=list)
    tasks: List[str] = Field(default_factory=list)
    builds: List[str] = Field(default_factory=list)
    state_machines: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    pipelines: List[str] = Field(default_factory=list)

    @classmethod
    def from_snapshot(cls, snapshot_data: Dict[str, Any]) -> "Default":
        """Create a default projection from snapshot data"""
        # Filter out empty lists to keep output clean
        filtered_data = {}
        for key, value in snapshot_data.items():
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

    def _question_with_answers(self, question_label: str) -> Dict[str, List[str]]:
        """Retrieve question text and its answers for a given question label"""
        try:
            command = [
                "question",
                "get",
                "--label",
                question_label,
                "--format",
                "json",
            ]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )
            question_data = json.loads(proc.stdout)
            question_text = question_data.get("text", question_label)
            answers = []
            if "validations_by_claim" in question_data:
                for claim_id, validations in question_data[
                    "validations_by_claim"
                ].items():
                    for validation in validations:
                        if "answers" in validation:
                            for answer in validation["answers"]:
                                if "text" in answer:
                                    answers.append(answer["text"])
            return {question_text: answers}
        except Exception as e:
            logging.warning(
                f"Error retrieving question data for '{question_label}': {e}"
            )
            return {question_label: []}  # Fallback to label with empty answers

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        import subprocess

        result = super().model_dump(**kwargs)
        if "questions" in result:
            questions_with_answers = {}
            for question_label in result["questions"]:
                questions_with_answers[question_label] = self._question_with_answers(
                    question_label
                )
            result["questions"] = questions_with_answers
        return result
