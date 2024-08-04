import click
import os
from typing import Optional
from appdirs import user_data_dir
from .mindex import Mindex

MINDEX_DIR = user_data_dir("Mindex", "dev")
os.makedirs(MINDEX_DIR, exist_ok=True)


def get_mindex_path(name: str) -> str:
    return os.path.join(MINDEX_DIR, f"{name}.pkl")


def get_mindex(name: str) -> Mindex:
    filename = get_mindex_path(name)
    if os.path.exists(filename):
        return Mindex.load(filename)
    else:
        mindex = Mindex(name)
        mindex.save(filename)
        click.echo(f"Created new Mindex instance '{name}'.")
        return mindex


def perform_search(mindex: Mindex, query: str, top_k: int):
    results = mindex.search(query, top_k)
    for doc_id, score in zip(results[0], results[1]):
        document = mindex.documents[doc_id]
        click.echo(f"Score: {score:.4f}, Title: {document[0]}, URL: {document[1]}")


@click.group(invoke_without_command=True)
@click.pass_context
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
@click.option('--top-k', default=5, help='Number of top results to return.')
@click.argument('query', required=False)
def cli(ctx, name: str, top_k: int, query: Optional[str]):
    """Manage and search your Mindex."""
    if ctx.invoked_subcommand is None:
        if query:
            mindex = get_mindex(name)
            perform_search(mindex, query, top_k)
        else:
            click.echo(ctx.get_help())


@cli.command()
@click.argument('query')
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
@click.option('--top-k', default=5, help='Number of top results to return.')
def search(query: str, name: str, top_k: int):
    """Perform a search query."""
    mindex = get_mindex(name)
    perform_search(mindex, query, top_k)


@cli.command()
@click.argument('url')
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
def add(url: str, name: str):
    """Add a new document to the index."""
    mindex = get_mindex(name)
    mindex.add(urls=[url])
    mindex.save(get_mindex_path(name))
    click.echo(f"Document from {url} added successfully to {name} index.")


@cli.command()
@click.argument('url')
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
def remove(url: str, name: str):
    """Remove a document from the index."""
    mindex = get_mindex(name)
    mindex.remove(url)
    mindex.save(get_mindex_path(name))
    click.echo(f"Document from {url} removed successfully from {name} index.")


@cli.command()
@click.argument('name')
@click.option('--model-id', help='Model ID to use for the new Mindex instance.')
def create(name: str, model_id: Optional[str]):
    """Create a new Mindex instance."""
    filename = get_mindex_path(name)
    if os.path.exists(filename):
        click.echo(f"An index named '{name}' already exists.")
        return

    kwargs = {"name": name}
    if model_id:
        kwargs["model_id"] = model_id

    mindex = Mindex(**kwargs)
    mindex.save(filename)
    click.echo(f"Created new Mindex instance '{name}'.")


@cli.command()
def list_indices():
    """List all available Mindex instances."""
    indices = [f.replace('.pkl', '') for f in os.listdir(MINDEX_DIR) if f.endswith('.pkl')]
    if indices:
        click.echo("Available Mindex instances:")
        for index in indices:
            click.echo(f"  - {index}")
    else:
        click.echo("No Mindex instances found.")


if __name__ == "__main__":
    cli()