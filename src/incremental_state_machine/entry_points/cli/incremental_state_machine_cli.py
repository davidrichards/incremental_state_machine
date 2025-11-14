import typer

app = typer.Typer(help="incremental_state_machine")

@app.command()
def hello():
    typer.echo(f"Hello from IncrementalStateMachine")

