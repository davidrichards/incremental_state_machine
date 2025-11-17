# db_learn.py

import json
import yaml
import typer
import importlib
import re
from typing import List, Optional
from pathlib import Path
from incremental_state_machine.domain.models.projections import DatabricksEntry

app = typer.Typer()

JSONL_FILE = (
    Path(__file__).parent.parent.parent.parent.parent
    / "var/state/databricks_todo.jsonl"
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def load_entries() -> List[dict]:
    if not JSONL_FILE.exists():
        return []
    with open(JSONL_FILE, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Clean up any entries that have empty labels arrays
    for entry in entries:
        if "labels" in entry and (not entry["labels"] or entry["labels"] == []):
            del entry["labels"]

    return entries


def save_entries(entries: List[dict]):
    with open(JSONL_FILE, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def find_record(entries: List[dict], command: Optional[str], label: Optional[str]):
    for record in entries:
        if command and record["command"] == command:
            return record
        if label and record.get("label") == label:
            return record
    return None


def find_record_by_command_or_label(entries: List[dict], identifier: str):
    """Find a record by either command name or label"""
    # First try to find by command
    for record in entries:
        if record["command"] == identifier:
            return record

    # If not found by command, try by label
    for record in entries:
        if record.get("label") == identifier:
            return record

    return None


def apply_projection(record: dict, format_arg: str):
    """
    format_arg may be:
      - "json"
      - "yaml"
      - "json:ProjectionName"
      - "yaml:ProjectionName"
    """
    if ":" in format_arg:
        fmt, proj_name = format_arg.split(":", 1)
        module = importlib.import_module(
            "incremental_state_machine.domain.models.projections"
        )
        Projection = getattr(module, proj_name)
        projected = Projection(**record).model_dump()
        return fmt, projected
    else:
        return format_arg, record


def emit_output(data, fmt: str):
    if fmt == "json":
        typer.echo(json.dumps(data, indent=2))
    elif fmt == "yaml":
        typer.echo(yaml.safe_dump(data, sort_keys=False))
    else:
        raise typer.BadParameter(f"Unknown format: {fmt}")


# ------------------------------------------------------------
# attach
# ------------------------------------------------------------


@app.command()
def attach(
    identifier: str = typer.Argument(
        ..., help="Command name or label to find and modify"
    ),
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Set a single label for this record (when identifier is a command)",
    ),
    question: Optional[str] = typer.Option(None, "--question"),
    status: Optional[str] = typer.Option(
        None, "--status", help="todo, in_process, addressed, blocked, unknown"
    ),
):
    """
    Modify a record in two modes:
    1. Update mode: <command> --label <label> → Find by command and set label
    2. Find mode: <label> [other options] → Find by label and make other updates
    """
    entries = load_entries()

    # Find the record by command or label
    rec = find_record_by_command_or_label(entries, identifier)
    if not rec:
        typer.echo(f"No record found for: {identifier}")
        raise typer.Exit(code=1)

    # Ensure standard fields exist
    rec.setdefault("questions", [])

    # If --label is provided, we're in update mode (setting a label on a command)
    if label is not None:
        # Check if identifier matches the command (to ensure we're setting label on a command)
        if rec["command"] == identifier:
            rec["label"] = label
            if "labels" in rec:
                del rec["labels"]
        else:
            typer.echo(
                "Cannot set label: identifier must be a command name when using --label"
            )
            raise typer.Exit(code=1)

    if question:
        if question not in rec["questions"]:
            rec["questions"].append(question)

    if status:
        rec["status"] = status

    save_entries(entries)
    typer.echo("Updated.")


# ------------------------------------------------------------
# get
# ------------------------------------------------------------


@app.command()
def get(
    identifier: str = typer.Argument(..., help="Command name or label to find"),
    format: str = typer.Option("json", "--format", help="json|yaml or json:Projection"),
):
    """
    Return a single record by command name or label.
    """
    entries = load_entries()
    rec = find_record_by_command_or_label(entries, identifier)

    if not rec:
        typer.echo(f"No record found for: {identifier}")
        raise typer.Exit(code=1)

    fmt, projected = apply_projection(rec, format)
    emit_output(projected, fmt)


# ------------------------------------------------------------
# list
# ------------------------------------------------------------


@app.command()
def list(
    status: Optional[str] = typer.Option(None, "--status"),
    question: Optional[str] = typer.Option(None, "--question"),
    command: Optional[str] = typer.Option(
        None, "--command", help="Filter by command name"
    ),
    category: Optional[str] = typer.Option(
        None, "--category", help="Filter by category"
    ),
    description: Optional[str] = typer.Option(
        None, "--description", help="Filter by description (supports regex)"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", help="Limit the number of results returned"
    ),
    format: str = typer.Option("json", "--format", help="json|yaml or json:Projection"),
):
    """
    List all records, or filtered by:
    - status
    - question label
    - command name
    - category
    - description (supports regex patterns)
    - limit number of results
    """
    entries = load_entries()

    if status:
        entries = [e for e in entries if e.get("status") == status]

    if question:
        entries = [e for e in entries if question in e.get("questions", [])]

    if command:
        entries = [
            e for e in entries if e.get("command", "").lower() == command.lower()
        ]

    if category:
        entries = [
            e for e in entries if e.get("category", "").lower() == category.lower()
        ]

    if description:
        try:
            # Try to compile as regex first
            pattern = re.compile(description, re.IGNORECASE)
            entries = [e for e in entries if pattern.search(e.get("description", ""))]
        except re.error:
            # If regex compilation fails, fall back to simple string matching
            entries = [
                e
                for e in entries
                if description.lower() in e.get("description", "").lower()
            ]

    fmt, _ = (format.split(":")[0], None)

    # Apply limit if specified
    if limit is not None:
        entries = entries[:limit]

    results = []
    for e in entries:
        _, projected = apply_projection(e, format)
        results.append(projected)

    emit_output(results, fmt)


# ------------------------------------------------------------
# create
# ------------------------------------------------------------


@app.command()
def create(
    command: str = typer.Argument(..., help="Command name for the new entry"),
    category: str = typer.Argument(..., help="Category for the new entry"),
    description: str = typer.Argument(..., help="Description for the new entry"),
    status: Optional[str] = typer.Option(
        "todo", "--status", help="todo, in_process, addressed, blocked, unknown"
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Set a label for this entry"
    ),
    question: Optional[List[str]] = typer.Option(
        None, "--question", help="Add question labels (can be used multiple times)"
    ),
    format: str = typer.Option("json", "--format", help="json|yaml or json:Projection"),
):
    """
    Create a new DatabricksEntry record.
    """
    entries = load_entries()

    # Check if command already exists
    existing = find_record_by_command_or_label(entries, command)
    if existing:
        typer.echo(f"Entry with command '{command}' already exists.")
        raise typer.Exit(code=1)

    # Create new entry using DatabricksEntry model
    new_entry_data = {
        "command": command,
        "category": category,
        "description": description,
        "status": status,
        "questions": question if question else [],
    }

    # Add label if provided
    if label:
        new_entry_data["label"] = label

    # Create DatabricksEntry to validate the data
    new_entry = DatabricksEntry(**new_entry_data)

    # Convert to dict and add to entries
    entry_dict = new_entry.model_dump()
    entries.append(entry_dict)

    # Save to file
    save_entries(entries)

    # Return the created entry using the specified format
    fmt, projected = apply_projection(entry_dict, format)
    emit_output(projected, fmt)


# ------------------------------------------------------------
# main entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    app()
