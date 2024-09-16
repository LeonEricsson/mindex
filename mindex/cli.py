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
        mindex = Mindex.load(filename)
    else:
        click.echo(f"Creating new Mindex instance '{name}'...")
        mindex = Mindex(name)
    return mindex


@click.group()
def cli():
    """Mindex CLI for managing and searching your mind index."""
    pass


@cli.command()
@click.argument('query')
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
@click.option('--top-k', default=5, help='Number of top results to return.')
def search(query: str, name: str, top_k: int):
    """Perform an interactive search query."""
    mindex = get_mindex(name)
    top_m_documents, _, top_k_chunks, _ = mindex.search(query, top_k, method='hybrid', top_l=20)
    
    document_chunks = {}
    for doc_idx, chunk_idx in zip(top_m_documents, top_k_chunks):
        if doc_idx not in document_chunks:
            document_chunks[doc_idx] = []
        document_chunks[doc_idx].append(chunk_idx)

    while True:
        click.echo("\nTop documents:")
        for i, doc_idx in enumerate(top_m_documents, 1):
            doc = mindex.documents[doc_idx]
            click.echo(f"{i}. {doc[0]} - {doc[1]}")
        
        click.echo("\nEnter the number of the document to view chunks, or 'q' to quit:")
        choice = click.prompt("Your choice", type=str)
        
        if choice.lower() == 'q':
            break
        
        try:
            doc_index = int(choice) - 1
            if 0 <= doc_index < len(top_m_documents):
                doc_idx = top_m_documents[doc_index]
                doc = mindex.documents[doc_idx]
                chunks = [mindex.chunks[i] for i in document_chunks[doc_idx]]
                
                click.echo(f"\nTitle: {doc[0]}")
                click.echo(f"URL: {doc[1]}")
                click.echo("Top chunks:")
                for c in chunks:
                    click.echo(f"- {c}")
                    click.echo("")
                
                click.echo("Press Enter to return to the document list...")
                click.prompt("", prompt_suffix="", default="", show_default=False)
            else:
                click.echo("Invalid selection. Please try again.")
        except ValueError:
            click.echo("Invalid input. Please enter a number or 'q'.")
    

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
@click.argument('name', required=False)
@click.option('--force', is_flag=True, help='Force deletion without confirmation')
def delete(name: Optional[str], force: bool):
    """Delete a Mindex instance."""
    if name is None:
        name = 'default'
    
    filename = get_mindex_path(name)
    
    if not os.path.exists(filename):
        click.echo(f"No Mindex instance named '{name}' found.")
        return

    if name == 'default' and not force:
        click.echo("Warning: You are attempting to delete the default Mindex instance.")
        click.echo("This may affect the normal operation of the tool.")
        click.echo("Use --force if you really want to delete the default instance.")
        return

    if not force:
        confirm = click.confirm(f"Are you sure you want to delete the Mindex instance '{name}'?")
        if not confirm:
            click.echo("Deletion cancelled.")
            return

    try:
        os.remove(filename)
        click.echo(f"Mindex instance '{name}' has been deleted.")
    except Exception as e:
        click.echo(f"An error occurred while trying to delete the instance: {str(e)}")


@cli.command()
@click.option('--name', default='default', help='Name of the Mindex instance to use.')
def info(name: str):
    """Display information about the Mindex instance."""
    mindex = get_mindex(name)
    
    click.echo(f"Information for Mindex instance '{name}':")
    click.echo(f"  NAME: {mindex.NAME}")
    click.echo(f"  CHUNK_SIZE: {mindex.CHUNK_SIZE}")
    click.echo(f"  EMBEDDING_DIM: {mindex.EMBEDDING_DIM}")
    click.echo(f"  Model ID: {mindex.model_id}")
    click.echo(f"  Number of documents: {len(mindex.documents)}")
    click.echo(f"  Number of chunks: {len(mindex.chunks)}")
    
    if mindex.chunk_index is not None and len(mindex.chunk_index) > 1:
        click.echo(f"  Chunk indexes:")
        for i, count in enumerate(mindex.chunk_index[1:], 1):
            click.echo(f"    Document {i}: {count - mindex.chunk_index[i-1]} chunks")
    
    if mindex.documents:
        click.echo("\nDocument List:")
        for i, (title, url) in enumerate(mindex.documents, 1):
            click.echo(f"  {i}. Title: {title}")
            click.echo(f"     URL: {url}")
    else:
        click.echo("\nNo documents indexed.")


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