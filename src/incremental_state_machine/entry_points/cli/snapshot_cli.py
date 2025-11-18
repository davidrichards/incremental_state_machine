from datetime import datetime
import os
import re
import yaml
import json
import importlib
from pathlib import Path
import typer
from typing import Optional, Tuple, List

from incremental_state_machine.domain.models.snapshot import Snapshot


app = typer.Typer()


def resolve_snapshots_directory() -> Path:
    """Resolve the snapshots directory path."""
    # Get the workspace root (project root directory)
    current_file = Path(__file__)
    # Go up: cli -> entry_points -> src -> incremental_state_machine -> project_root
    project_root = current_file.parent.parent.parent.parent.parent
    snapshots_dir = project_root / "var" / "spec" / "design" / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    return snapshots_dir


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
    """Extract version from snapshot filename."""
    filename = file_path.stem
    parts = filename.split(".")
    if len(parts) < 4:
        raise ValueError(f"Invalid snapshot filename: {filename}")
    return ".".join(parts[-3:])


def get_label_from_filename(file_path: Path) -> str:
    """Extract label from snapshot filename."""
    filename = file_path.stem
    parts = filename.split(".")
    if len(parts) < 4:
        raise ValueError(f"Invalid snapshot filename: {filename}")
    return ".".join(parts[:-3])


def find_snapshot_files(
    directory: Path, label: Optional[str] = None, version: Optional[str] = None
) -> List[Path]:
    """Find snapshot files matching the pattern <label>.<semver>.yaml"""
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


def load_snapshot_from_file(file_path: Path) -> Snapshot:
    """Load a Snapshot from a YAML file."""
    with open(file_path, "r") as f:
        data = yaml.safe_load(f)

    # Remove the filepath comment if it exists
    if "filepath" in data:
        del data["filepath"]

    return Snapshot(**data)


def create_empty_snapshot(directory: Path, label: Optional[str] = None) -> Snapshot:
    """Create an empty snapshot and save it to a file."""
    if label:
        filename = f"{label}.0.1.0.yaml"
    else:
        filename = "default.0.1.0.yaml"

    file_path = directory / filename

    # Create an empty snapshot with minimal required data
    snapshot = Snapshot(body="", label=label)  # body is required

    # Save to file
    with open(file_path, "w") as f:
        # Add filepath comment
        f.write(f"# filepath: {file_path}\n")
        yaml.safe_dump(
            snapshot.model_dump(), f, default_flow_style=False, sort_keys=False
        )

    return snapshot


def apply_projection(snapshot_data: dict, format_arg: str):
    """
    Apply projection to snapshot data.
    format_arg may be:
      - "json" or "yaml" (uses default projection)
      - "json:ProjectionName" or "yaml:ProjectionName" (uses specific projection)
    """
    module = importlib.import_module(
        "incremental_state_machine.domain.models.snapshot_projections"
    )

    if ":" in format_arg:
        fmt, proj_name = format_arg.split(":", 1)
        try:
            Projection = getattr(module, proj_name)
            projected = Projection(**snapshot_data).model_dump()
            return fmt, projected
        except AttributeError:
            raise typer.BadParameter(f"Unknown projection: {proj_name}")
    else:
        # Use default projection
        Default = getattr(module, "Default")
        projected = Default.from_snapshot(snapshot_data).model_dump()
        return format_arg, projected


def emit_output(data, fmt: str):
    """Emit data in the specified format"""
    if fmt.lower() == "json":
        typer.echo(json.dumps(data, indent=2, default=str))
    elif fmt.lower() == "yaml":
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False))
    else:
        raise typer.BadParameter(f"Unknown format: {fmt}")


def current_snapshot(label: Optional[str] = None) -> Snapshot:
    """
    Retrieve the current snapshot of the system state.

    Args:
        label: Optional label to filter by. If provided, finds the most-recently
               modified file matching that label. If None, finds the most-recently
               modified file overall.

    Returns:
        Snapshot: The current snapshot, either loaded from file or newly created.
    """
    directory = resolve_snapshots_directory()

    # Find matching snapshot files
    matching_files = find_snapshot_files(directory, label)

    # Get the most recently modified file
    most_recent_file = get_most_recent_file(matching_files)

    if most_recent_file:
        # Load and return the snapshot from the most recent file
        return load_snapshot_from_file(most_recent_file)
    else:
        # No files found, create an empty snapshot
        return create_empty_snapshot(directory, label)


@app.command()
def get(
    label: Optional[str] = typer.Option(
        None, "--label", help="Filter by snapshot label"
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
    """Get the most recent snapshot matching the filters."""
    directory = resolve_snapshots_directory()

    # Find matching snapshot files
    matching_files = find_snapshot_files(directory, label, version)

    # Get the most recently modified file
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        typer.echo("No snapshots found matching the criteria.")
        raise typer.Exit(code=1)

    # Load the snapshot
    snapshot = load_snapshot_from_file(most_recent_file)

    # Apply projection and emit output
    fmt, projected_data = apply_projection(snapshot.model_dump(), format)
    emit_output(projected_data, fmt)


@app.command("list")
def list_snapshots(
    label: Optional[str] = typer.Option(
        None, "--label", help="Filter by snapshot label"
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
    """List all snapshots matching the filters."""
    directory = resolve_snapshots_directory()

    # Find matching snapshot files
    matching_files = find_snapshot_files(directory, label, version)

    if not matching_files:
        typer.echo("No snapshots found matching the criteria.")
        return

    # Sort by modification time (most recent first)
    sorted_files = sorted(matching_files, key=lambda f: f.stat().st_mtime, reverse=True)

    # Load and collect all snapshots
    snapshots = []
    fmt = format  # Initialize fmt with the format parameter
    for file_path in sorted_files:
        snapshot = load_snapshot_from_file(file_path)
        # Add filename info for context
        snapshot_data = snapshot.model_dump()
        snapshot_data["_filename"] = file_path.name
        snapshot_data["_modified"] = datetime.fromtimestamp(
            file_path.stat().st_mtime
        ).isoformat()

        # Apply projection to each snapshot
        fmt, projected_data = apply_projection(snapshot_data, format)
        snapshots.append(projected_data)

    # Output the snapshots
    emit_output(snapshots, fmt)


def save_snapshot_to_file(snapshot: Snapshot, file_path: Path):
    """Save a snapshot to a YAML file."""
    with open(file_path, "w") as f:
        # Add filepath comment
        f.write(f"# filepath: {file_path}\n")
        yaml.safe_dump(
            snapshot.model_dump(), f, default_flow_style=False, sort_keys=False
        )


@app.command()
def create(
    label: str = typer.Option("default", "--label", help="Label for the new snapshot"),
    version: str = typer.Option(
        "0.1.0", "--version", help="Version for the new snapshot (e.g., '1.0.0')"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        help="Output format: yaml, json, yaml:ProjectionName, json:ProjectionName",
    ),
):
    """Create a new snapshot. Idempotent - won't overwrite existing snapshots."""
    directory = resolve_snapshots_directory()

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
        typer.echo(f"Snapshot already exists: {file_path}")
        # Load and return existing snapshot
        snapshot = load_snapshot_from_file(file_path)
    else:
        # Create new empty snapshot
        snapshot = Snapshot(body="", label=label)

        # Save to file
        save_snapshot_to_file(snapshot, file_path)
        typer.echo(f"Created new snapshot: {file_path}")

    # Apply projection and emit output (same as get command)
    fmt, projected_data = apply_projection(snapshot.model_dump(), format)
    emit_output(projected_data, fmt)


@app.command()
def replace(
    label: Optional[str] = typer.Option(
        None,
        "--label",
        help="Label of snapshot to replace (if not provided, uses most recent)",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Specific version to create (e.g., '1.2.0'). If not provided, bumps patch version",
    ),
):
    """Create a new version of a snapshot by copying an existing one."""
    directory = resolve_snapshots_directory()

    # Find the source snapshot
    matching_files = find_snapshot_files(directory, label)
    source_file = get_most_recent_file(matching_files)

    if not source_file:
        if label:
            typer.echo(f"No snapshot found with label '{label}'.")
        else:
            typer.echo("No snapshots found.")
        raise typer.Exit(code=1)

    # Load the source snapshot
    source_snapshot = load_snapshot_from_file(source_file)

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
            f"Snapshot with version {new_version} already exists: {new_file_path}"
        )
        raise typer.Exit(code=1)

    # Create a copy of the snapshot with updated timestamps
    new_snapshot = Snapshot(
        body=source_snapshot.body,
        label=source_snapshot.label,
        session=source_snapshot.session,
        questions=source_snapshot.questions.copy(),
        designs=source_snapshot.designs.copy(),
        offers=source_snapshot.offers.copy(),
        engagements=source_snapshot.engagements.copy(),
        tasks=source_snapshot.tasks.copy(),
        builds=source_snapshot.builds.copy(),
        state_machines=source_snapshot.state_machines.copy(),
        agents=source_snapshot.agents.copy(),
        pipelines=source_snapshot.pipelines.copy(),
    )

    # Save the new snapshot
    save_snapshot_to_file(new_snapshot, new_file_path)

    typer.echo(f"Created new snapshot version {new_version}")
    typer.echo(f"Source: {source_file}")
    typer.echo(f"Target: {new_file_path}")


@app.command()
def attach(
    snapshot_label: Optional[str] = typer.Option(
        None,
        "--snapshot-label",
        help="Label of snapshot to modify (if not provided, uses most recent)",
    ),
    label: Optional[str] = typer.Option(
        None, "--label", help="Set label for this snapshot"
    ),
    session: Optional[str] = typer.Option(
        None, "--session", help="Set session for this snapshot"
    ),
    question: Optional[str] = typer.Option(
        None, "--question", help="Add question to snapshot"
    ),
    design: Optional[str] = typer.Option(
        None, "--design", help="Add design to snapshot"
    ),
    offer: Optional[str] = typer.Option(None, "--offer", help="Add offer to snapshot"),
    engagement: Optional[str] = typer.Option(
        None, "--engagement", help="Add engagement to snapshot"
    ),
    task: Optional[str] = typer.Option(None, "--task", help="Add task to snapshot"),
    build: Optional[str] = typer.Option(None, "--build", help="Add build to snapshot"),
    state_machine: Optional[str] = typer.Option(
        None, "--state-machine", help="Add state machine to snapshot"
    ),
    agent: Optional[str] = typer.Option(None, "--agent", help="Add agent to snapshot"),
    pipeline: Optional[str] = typer.Option(
        None, "--pipeline", help="Add pipeline to snapshot"
    ),
):
    """Attach values to a snapshot. Overwrites label/session, appends to lists if not already present."""
    directory = resolve_snapshots_directory()

    # Find the target snapshot
    matching_files = find_snapshot_files(directory, snapshot_label)
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        if snapshot_label:
            typer.echo(
                f"No snapshot found with label '{snapshot_label}'. Creating new snapshot."
            )
            snapshot = create_empty_snapshot(directory, snapshot_label)
            # Update the file path to match what we just created
            filename = f"{snapshot_label}.0.1.0.yaml"
            most_recent_file = directory / filename
        else:
            typer.echo("No snapshots found. Creating new default snapshot.")
            snapshot = create_empty_snapshot(directory, None)
            filename = "default.0.1.0.yaml"
            most_recent_file = directory / filename
    else:
        # Load existing snapshot
        snapshot = load_snapshot_from_file(most_recent_file)

    # Track if anything was modified
    modified = False

    # Handle single-value fields (overwrite)
    if label is not None:
        snapshot.label = label
        modified = True

    if session is not None:
        snapshot.session = session
        modified = True

    # Handle list fields (append if not present)
    list_operations = [
        (question, snapshot.questions, "question"),
        (design, snapshot.designs, "design"),
        (offer, snapshot.offers, "offer"),
        (engagement, snapshot.engagements, "engagement"),
        (task, snapshot.tasks, "task"),
        (build, snapshot.builds, "build"),
        (state_machine, snapshot.state_machines, "state machine"),
        (agent, snapshot.agents, "agent"),
        (pipeline, snapshot.pipelines, "pipeline"),
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
        # Save the updated snapshot
        save_snapshot_to_file(snapshot, most_recent_file)
        typer.echo(f"Snapshot saved to: {most_recent_file}")
    else:
        typer.echo("No changes made to snapshot.")


@app.command()
def detach(
    snapshot_label: Optional[str] = typer.Option(
        None,
        "--snapshot-label",
        help="Label of snapshot to modify (if not provided, uses most recent)",
    ),
    label: bool = typer.Option(False, "--label", help="Clear label from snapshot"),
    session: bool = typer.Option(
        False, "--session", help="Clear session from snapshot"
    ),
    question: Optional[str] = typer.Option(
        None, "--question", help="Remove question from snapshot"
    ),
    design: Optional[str] = typer.Option(
        None, "--design", help="Remove design from snapshot"
    ),
    offer: Optional[str] = typer.Option(
        None, "--offer", help="Remove offer from snapshot"
    ),
    engagement: Optional[str] = typer.Option(
        None, "--engagement", help="Remove engagement from snapshot"
    ),
    task: Optional[str] = typer.Option(
        None, "--task", help="Remove task from snapshot"
    ),
    build: Optional[str] = typer.Option(
        None, "--build", help="Remove build from snapshot"
    ),
    state_machine: Optional[str] = typer.Option(
        None, "--state-machine", help="Remove state machine from snapshot"
    ),
    agent: Optional[str] = typer.Option(
        None, "--agent", help="Remove agent from snapshot"
    ),
    pipeline: Optional[str] = typer.Option(
        None, "--pipeline", help="Remove pipeline from snapshot"
    ),
):
    """Detach values from a snapshot. Sets label/session to None, removes from lists if present."""
    directory = resolve_snapshots_directory()

    # Find the target snapshot
    matching_files = find_snapshot_files(directory, snapshot_label)
    most_recent_file = get_most_recent_file(matching_files)

    if not most_recent_file:
        if snapshot_label:
            typer.echo(f"No snapshot found with label '{snapshot_label}'.")
        else:
            typer.echo("No snapshots found.")
        raise typer.Exit(code=1)

    # Load existing snapshot
    snapshot = load_snapshot_from_file(most_recent_file)

    # Track if anything was modified
    modified = False

    # Handle single-value fields (set to None)
    if label:
        if snapshot.label is not None:
            snapshot.label = None
            typer.echo("Cleared label")
            modified = True
        else:
            typer.echo("Label was already empty")

    if session:
        if snapshot.session is not None:
            snapshot.session = None
            typer.echo("Cleared session")
            modified = True
        else:
            typer.echo("Session was already empty")

    # Handle list fields (remove if present)
    list_operations = [
        (question, snapshot.questions, "question"),
        (design, snapshot.designs, "design"),
        (offer, snapshot.offers, "offer"),
        (engagement, snapshot.engagements, "engagement"),
        (task, snapshot.tasks, "task"),
        (build, snapshot.builds, "build"),
        (state_machine, snapshot.state_machines, "state machine"),
        (agent, snapshot.agents, "agent"),
        (pipeline, snapshot.pipelines, "pipeline"),
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
        # Save the updated snapshot
        save_snapshot_to_file(snapshot, most_recent_file)
        typer.echo(f"Snapshot saved to: {most_recent_file}")
    else:
        typer.echo("No changes made to snapshot.")


if __name__ == "__main__":
    app()
