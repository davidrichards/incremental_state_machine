from datetime import datetime
import os
import re
import yaml
import json
import importlib
from pathlib import Path
import typer
from typing import Optional, Tuple, List, Annotated

from incremental_state_machine.domain.models.context import Context


app = typer.Typer()


def resolve_contexts_directory() -> Path:
    """Resolve the contexts directory path."""
    # Get the workspace root (project root directory)
    current_file = Path(__file__)
    # Go up: cli -> entry_points -> src -> incremental_state_machine -> project_root
    project_root = current_file.parent.parent.parent.parent.parent
    contexts_dir = project_root / "var" / "spec" / "design" / "contexts"
    contexts_dir.mkdir(parents=True, exist_ok=True)
    return contexts_dir


def parse_semver(version_str: str) -> Tuple[int, int, int]:
    """Parse semantic version string into (major, minor, patch) tuple."""
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_str)
    if not match:
        raise ValueError(f"Invalid semantic version: {version_str}")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def format_semver(major: int, minor: int, patch: int) -> str:
    """Format semantic version tuple into string."""
    return f"{major}.{minor}.{patch}"


def bump_patch_version(version_str: str) -> str:
    """Bump the patch version of a semantic version string."""
    major, minor, patch = parse_semver(version_str)
    return format_semver(major, minor, patch + 1)


def get_version_from_filename(file_path: Path) -> str:
    """Extract version from context filename."""
    filename = file_path.stem
    parts = filename.split(".")
    if len(parts) < 4:
        raise ValueError(f"Invalid context filename: {filename}")
    return ".".join(parts[-3:])


def get_label_from_filename(file_path: Path) -> str:
    """Extract label from context filename."""
    filename = file_path.stem
    parts = filename.split(".")
    if len(parts) < 4:
        raise ValueError(f"Invalid context filename: {filename}")
    return ".".join(parts[:-3])


def find_context_files(
    directory: Path, label: Optional[str] = None, version: Optional[str] = None
) -> List[Path]:
    """Find context files matching the pattern <label>.<semver>.yaml"""
    matching_files = []

    for file_path in directory.glob("*.yaml"):
        filename = file_path.stem  # filename without extension
        parts = filename.split(".")

        # We need at least 4 parts for a valid file: <label>.<major>.<minor>.<patch>
        if len(parts) < 4:
            continue

        # The last 3 parts must form a valid semver
        # This means the format is: <label>.<major>.<minor>.<patch>
        # Where <label> can contain dots, but the file must end with exactly 3 numeric semver parts
        try:
            last_three = parts[-3:]
            semver_str = ".".join(last_three)
            parse_semver(semver_str)

            # Calculate what the label would be (everything except the last 3 parts)
            file_label = ".".join(parts[:-3])

            # Check label filter
            if label is not None and file_label != label:
                continue

            # Check version filter
            if version is not None and semver_str != version:
                continue

            matching_files.append(file_path)

        except ValueError:
            continue  # Skip files where the last 3 parts don't form valid semver

    return matching_files


def get_most_recent_file(files: List[Path]) -> Optional[Path]:
    """Get the most recently modified file from a list of files."""
    if not files:
        return None

    return max(files, key=lambda f: f.stat().st_mtime)


def load_context_from_file(file_path: Path) -> Context:
    """Load a Context from a YAML file."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Remove the filepath comment if it exists
    if "filepath" in data:
        del data["filepath"]

    return Context(**data)


def create_empty_context(directory: Path, label: Optional[str] = None) -> Context:
    """Create an empty context and save it to a file."""
    if label:
        filename = f"{label}.0.1.0.yaml"
    else:
        filename = "default.0.1.0.yaml"

    file_path = directory / filename

    # Create an empty context with minimal required data
    context = Context(body="", label=label)  # body is required

    # Save to file
    with open(file_path, "w") as f:
        # Add filepath comment
        f.write(f"# filepath: {file_path}\n")
        yaml.safe_dump(
            context.model_dump(), f, default_flow_style=False, sort_keys=False
        )

    return context


def apply_projection(context_data: dict, format_arg: str):
    """
    Apply projection to context data.
    format_arg may be:
      - "json" or "yaml" (uses default projection)
      - "json:ProjectionName" or "yaml:ProjectionName" (uses specific projection)
    """
    module = importlib.import_module(
        "incremental_state_machine.domain.models.context_projections"
    )

    if ":" in format_arg:
        fmt, proj_name = format_arg.split(":", 1)
        try:
            Projection = getattr(module, proj_name)
            # Check if projection has from_context method, otherwise use constructor
            if hasattr(Projection, "from_context"):
                projected = Projection.from_context(context_data).model_dump()
            else:
                projected = Projection(**context_data).model_dump()
            return fmt, projected
        except AttributeError:
            raise typer.BadParameter(f"Unknown projection: {proj_name}")
    else:
        # Use default projection
        Default = getattr(module, "Default")
        projected = Default.from_context(context_data).model_dump()
        return format_arg, projected


def emit_output(data, fmt: str):
    """Emit data in the specified format"""
    if fmt.lower() == "json":
        typer.echo(json.dumps(data, indent=2, default=str))
    elif fmt.lower() == "yaml":
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
    else:
        raise typer.BadParameter(f"Unknown format: {fmt}")


def current_context(label: Optional[str] = None) -> Context:
    """
    Retrieve the current context of the system state.

    Args:
        label: Optional label to filter by. If provided, finds the most-recently
               modified file matching that label. If None, finds the most-recently
               modified file overall.

    Returns:
        Context: The current context, either loaded from file or newly created.
    """
    directory = resolve_contexts_directory()

    # Find matching context files
    matching_files = find_context_files(directory, label)

    # Get the most recently modified file
    most_recent_file = get_most_recent_file(matching_files)

    if most_recent_file:
        # Load and return the context from the most recent file
        return load_context_from_file(most_recent_file)
    else:
        # No files found, create an empty context
        return create_empty_context(directory, label)


@app.command()
def get(
    identifier: Optional[str] = typer.Argument(
        None, help="Context identifier (label or ID)"
    ),
    version: Optional[str] = typer.Option(
        None, "--version", help="Filter by specific version (e.g., '1.0.0')"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        help="Output format: yaml, json, yaml:ProjectionName, json:ProjectionName",
    ),
):
    """Get the most recent context matching the filters."""
    directory = resolve_contexts_directory()

    # Find matching context files
    matching_files = find_context_files(directory, identifier, version)

    # Get the most recently modified file
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        typer.echo("No contexts found matching the criteria.")
        raise typer.Exit(code=1)

    # Load the context
    context = load_context_from_file(most_recent_file)

    # Apply projection and emit output
    fmt, projected_data = apply_projection(context.model_dump(), format)
    emit_output(projected_data, fmt)


@app.command("list")
def list_contexts(
    identifier: Optional[str] = typer.Argument(
        None, help="Context identifier (label or ID)"
    ),
    version: Optional[str] = typer.Option(
        None, "--version", help="Filter by specific version (e.g., '1.0.0')"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        help="Output format: yaml, json, yaml:ProjectionName, json:ProjectionName",
    ),
):
    """List all contexts matching the filters."""
    directory = resolve_contexts_directory()

    # Find matching context files
    matching_files = find_context_files(directory, identifier, version)

    if not matching_files:
        typer.echo("No contexts found matching the criteria.")
        return

    # Sort by modification time (most recent first)
    sorted_files = sorted(matching_files, key=lambda f: f.stat().st_mtime, reverse=True)

    # Load and collect all contexts
    contexts = []
    fmt = format  # Initialize fmt with the format parameter
    for file_path in sorted_files:
        context = load_context_from_file(file_path)
        # Add filename info for context
        context_data = context.model_dump()
        context_data["_filename"] = file_path.name
        context_data["_modified"] = datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).isoformat()

        # Apply projection to each context
        fmt, projected_data = apply_projection(context_data, format)
        contexts.append(projected_data)

    # Output the contexts
    emit_output(contexts, fmt)


def save_context_to_file(context: Context, file_path: Path):
    """Save a context to a YAML file."""
    with open(file_path, "w") as f:
        # Add filepath comment
        f.write(f"# filepath: {file_path}\n")
        yaml.safe_dump(
            context.model_dump(), f, default_flow_style=False, sort_keys=False
        )


@app.command()
def create(
    label: str = typer.Option("default", "--label", help="Label for the new context"),
    version: str = typer.Option(
        "0.1.0", "--version", help="Version for the new context (e.g., '1.0.0')"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        help="Output format: yaml, json, yaml:ProjectionName, json:ProjectionName",
    ),
):
    """Create a new context. Idempotent - won't overwrite existing contexts."""
    directory = resolve_contexts_directory()

    # Validate version format
    try:
        parse_semver(version)
    except ValueError:
        typer.echo(
            f"Invalid version format: {version}. Expected format: major.minor.patch"
        )
        raise typer.Exit(code=1)

    # Create filename
    filename = f"{label}.{version}.yaml"
    file_path = directory / filename

    # Check if file already exists (idempotent behavior)
    if file_path.exists():
        typer.echo(f"Context already exists: {file_path}")
        # Load and return existing context
        context = load_context_from_file(file_path)
    else:
        # Create new empty context
        context = Context(body="", label=label)

        # Save to file
        save_context_to_file(context, file_path)
        typer.echo(f"Created new context: {file_path}")

    # Apply projection and emit output (same as get command)
    fmt, projected_data = apply_projection(context.model_dump(), format)
    emit_output(projected_data, fmt)


@app.command()
def replace(
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Label of context to replace (if not provided, uses most recent)",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Specific version to create (e.g., '1.2.0'). If not provided, bumps patch version",
    ),
):
    """Create a new version of a context by copying an existing one."""
    directory = resolve_contexts_directory()

    # Find the source context
    matching_files = find_context_files(directory, label)
    source_file = get_most_recent_file(matching_files)

    if not source_file:
        if label:
            typer.echo(f"No context found with label '{label}'.")
        else:
            typer.echo("No contexts found.")
        raise typer.Exit(code=1)

    # Load the source context
    source_context = load_context_from_file(source_file)

    # Get the current version and label from the source file
    current_version = get_version_from_filename(source_file)
    source_label = get_label_from_filename(source_file)

    # Determine the new version
    if version:
        # Validate the provided version
        try:
            parse_semver(version)
            new_version = version
        except ValueError:
            typer.echo(
                f"Invalid version format: {version}. Expected format: major.minor.patch"
            )
            raise typer.Exit(code=1)
    else:
        # Bump the patch version
        new_version = bump_patch_version(current_version)

    # Create the new filename
    new_filename = f"{source_label}.{new_version}.yaml"
    new_file_path = directory / new_filename

    # Check if the target version already exists
    if new_file_path.exists():
        typer.echo(
            f"Context with version {new_version} already exists: {new_file_path}"
        )
        raise typer.Exit(code=1)

    # Create a copy of the context with updated timestamps
    new_context = Context(
        body=source_context.body,
        label=source_context.label,
        session=source_context.session,
        claim=source_context.claim,
        interest=source_context.interest,
        questions=source_context.questions.copy(),
        doc_chat=source_context.doc_chat,
        design=source_context.design,
        offer=source_context.offer,
        engagement=source_context.engagement,
        tasks=source_context.tasks.copy(),
        build=source_context.build,
        state_machine=source_context.state_machine,
        agents=source_context.agents.copy(),
        peek=source_context.peek,
    )

    # Save the new context
    save_context_to_file(new_context, new_file_path)

    typer.echo(f"Created new context version {new_version}")
    typer.echo(f"Source: {source_file}")
    typer.echo(f"Target: {new_file_path}")


@app.command()
def attach(
    context_label: Optional[str] = typer.Option(
        None,
        "--context-label",
        help="Label of context to modify (if not provided, uses most recent)",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Set label for this context"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="Set session for this context"
    ),
    claim: Optional[str] = typer.Option(
        None, "--claim", help="Set claim for this context"
    ),
    interest: Optional[str] = typer.Option(
        None, "--interest", help="Set interest for this context"
    ),
    doc_chat: Optional[str] = typer.Option(
        None, "--doc-chat", help="Set doc_chat for this context"
    ),
    design: Optional[str] = typer.Option(
        None, "--design", help="Set design for this context"
    ),
    offer: Optional[str] = typer.Option(
        None, "--offer", help="Set offer for this context"
    ),
    engagement: Optional[str] = typer.Option(
        None, "--engagement", help="Set engagement for this context"
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Set build for this context"
    ),
    state_machine: Optional[str] = typer.Option(
        None, "--state-machine", help="Set state machine for this context"
    ),
    peek: Optional[str] = typer.Option(
        None, "--peek", help="Set peek for this context"
    ),
    question: Optional[str] = typer.Option(
        None, "--question", help="Add question to context"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Add task to context"),
    agent: Optional[str] = typer.Option(None, "--agent", help="Add agent to context"),
):
    """Attach values to a context. Overwrites singular fields, appends to list fields if not already present."""
    directory = resolve_contexts_directory()

    # Find the target context
    matching_files = find_context_files(directory, context_label)
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        if context_label:
            typer.echo(
                f"No context found with label '{context_label}'. Creating new context."
            )
            context = create_empty_context(directory, context_label)
            # Update the file path to match what we just created
            filename = f"{context_label}.0.1.0.yaml"
            most_recent_file = directory / filename
        else:
            typer.echo("No contexts found. Creating new default context.")
            context = create_empty_context(directory, None)
            filename = "default.0.1.0.yaml"
            most_recent_file = directory / filename
    else:
        # Load existing context
        context = load_context_from_file(most_recent_file)

    # Track if anything was modified
    modified = False

    # Handle single-value fields (overwrite)
    singular_operations = [
        (label, "label"),
        (session, "session"),
        (claim, "claim"),
        (interest, "interest"),
        (doc_chat, "doc_chat"),
        (design, "design"),
        (offer, "offer"),
        (engagement, "engagement"),
        (build, "build"),
        (state_machine, "state_machine"),
        (peek, "peek"),
    ]

    for value, field_name in singular_operations:
        if value is not None:
            setattr(context, field_name, value)
            modified = True

    # Handle list fields (append if not present)
    list_operations = [
        (question, context.questions, "question"),
        (task, context.tasks, "task"),
        (agent, context.agents, "agent"),
    ]

    for value, target_list, item_type in list_operations:
        if value is not None:
            if value not in target_list:
                target_list.append(value)
                typer.echo(f"Added {item_type}: {value}")
                modified = True
            else:
                typer.echo(f"{item_type.capitalize()} '{value}' already exists")

    if modified:
        # Save the updated context
        save_context_to_file(context, most_recent_file)
        typer.echo(f"Context saved to: {most_recent_file}")
    else:
        typer.echo("No changes made to context.")


@app.command()
def detach(
    context_label: Optional[str] = typer.Option(
        None,
        "--context-label",
        help="Label of context to modify (if not provided, uses most recent)",
    ),
    label: bool = typer.Option(False, "--label", help="Clear label from context"),
    session: bool = typer.Option(False, "--session", help="Clear session from context"),
    claim: bool = typer.Option(False, "--claim", help="Clear claim from context"),
    interest: bool = typer.Option(
        False, "--interest", help="Clear interest from context"
    ),
    doc_chat: bool = typer.Option(
        False, "--doc-chat", help="Clear doc_chat from context"
    ),
    design: bool = typer.Option(False, "--design", help="Clear design from context"),
    offer: bool = typer.Option(False, "--offer", help="Clear offer from context"),
    engagement: bool = typer.Option(
        False, "--engagement", help="Clear engagement from context"
    ),
    build: bool = typer.Option(False, "--build", help="Clear build from context"),
    state_machine: bool = typer.Option(
        False, "--state-machine", help="Clear state machine from context"
    ),
    peek: bool = typer.Option(False, "--peek", help="Clear peek from context"),
    question: Optional[str] = typer.Option(
        None, "--question", help="Remove question from context"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Remove task from context"),
    agent: Optional[str] = typer.Option(
        None, "--agent", help="Remove agent from context"
    ),
):
    """Detach values from a context. Sets singular fields to None, removes from list fields if present."""
    directory = resolve_contexts_directory()

    # Find the target context
    matching_files = find_context_files(directory, context_label)
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        if context_label:
            typer.echo(f"No context found with label '{context_label}'.")
        else:
            typer.echo("No contexts found.")
        raise typer.Exit(code=1)

    # Load existing context
    context = load_context_from_file(most_recent_file)

    # Track if anything was modified
    modified = False

    # Handle single-value fields (set to None)
    singular_operations = [
        (label, "label"),
        (session, "session"),
        (claim, "claim"),
        (interest, "interest"),
        (doc_chat, "doc_chat"),
        (design, "design"),
        (offer, "offer"),
        (engagement, "engagement"),
        (build, "build"),
        (state_machine, "state_machine"),
        (peek, "peek"),
    ]

    for clear_field, field_name in singular_operations:
        if clear_field:
            current_value = getattr(context, field_name)
            if current_value is not None:
                setattr(context, field_name, None)
                typer.echo(f"Cleared {field_name}")
                modified = True
            else:
                typer.echo(f"{field_name.capitalize()} was already empty")

    # Handle list fields (remove if present)
    list_operations = [
        (question, context.questions, "question"),
        (task, context.tasks, "task"),
        (agent, context.agents, "agent"),
    ]

    for value, target_list, item_type in list_operations:
        if value is not None:
            if value in target_list:
                target_list.remove(value)
                typer.echo(f"Removed {item_type}: {value}")
                modified = True
            else:
                typer.echo(f"{item_type.capitalize()} '{value}' not found")

    if modified:
        # Save the updated context
        save_context_to_file(context, most_recent_file)
        typer.echo(f"Context saved to: {most_recent_file}")
    else:
        typer.echo("No changes made to context.")


if __name__ == "__main__":
    app()
