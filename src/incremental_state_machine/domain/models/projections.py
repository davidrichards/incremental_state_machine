from pydantic import BaseModel, Field, computed_field
from typing import List, Optional, Literal, Dict, Any
import subprocess
import json
import logging


class DatabricksEntry(BaseModel):
    command: str
    category: str
    description: str
    status: Literal["todo", "in_process", "addressed", "blocked", "unknown"] = "todo"
    label: Optional[str] = None
    questions: Optional[List[str]] = Field(default_factory=list)


class Compact(BaseModel):
    command: str
    status: str


class Discussion(BaseModel):
    command: str
    label: Optional[str] = None
    questions: List[str] = Field(default_factory=list)

    def _get_question_data(self, question_label: str) -> tuple[str, List[str]]:
        """Get question text and answers for a specific question label using subprocess"""
        try:
            result = subprocess.run(
                ["question", "get", "--label", question_label, "--format", "json"],
                capture_output=True,
                text=True,
                check=True,
            )

            question_data = json.loads(result.stdout)
            answers = []
            question_text = question_data.get(
                "text", question_label
            )  # Fallback to label if no text

            # Extract answers from validations_by_claim
            if "validations_by_claim" in question_data:
                for claim_id, validations in question_data[
                    "validations_by_claim"
                ].items():
                    for validation in validations:
                        if "answers" in validation:
                            for answer in validation["answers"]:
                                if "text" in answer:
                                    answers.append(answer["text"])

            # Also check validations_by_interest if it exists
            if "validations_by_interest" in question_data:
                for interest_id, validations in question_data[
                    "validations_by_interest"
                ].items():
                    for validation in validations:
                        if "answers" in validation:
                            for answer in validation["answers"]:
                                if "text" in answer:
                                    answers.append(answer["text"])

            return question_text, answers

        except subprocess.CalledProcessError as e:
            logging.warning(f"Failed to get data for question '{question_label}': {e}")
            return question_label, []  # Fallback to label as text
        except json.JSONDecodeError as e:
            logging.warning(
                f"Failed to parse JSON response for question '{question_label}': {e}"
            )
            return question_label, []  # Fallback to label as text
        except Exception as e:
            logging.warning(
                f"Unexpected error getting data for question '{question_label}': {e}"
            )
            return question_label, []  # Fallback to label as text

    @computed_field
    @property
    def questions_dict(self) -> Dict[str, List[str]]:
        """Transform questions list into a dict of {question_text: [answers]}"""
        result = {}
        for question_label in self.questions:
            question_text, answers = self._get_question_data(question_label)
            result[question_text] = answers
        return result

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to return questions as dict instead of list"""
        # Get the base model data, excluding the computed field
        data = super().model_dump(exclude={"questions_dict"}, **kwargs)
        # Replace questions list with questions dict
        data["questions"] = self.questions_dict
        return data


class SnapshotDefault(BaseModel):
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
    def from_snapshot(cls, snapshot_data: Dict[str, Any]) -> "SnapshotDefault":
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


class SnapshotDiscussion(SnapshotDefault):

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
