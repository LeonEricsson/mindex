import os
import pickle
import requests
import tempfile
from requests.exceptions import HTTPError
from typing import Union, Tuple, List
from .bm25 import BM25
from .vector_store import VectorStorage, SimilarityMetric

import numpy as np
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from rerankers import Reranker

Array = np.ndarray

def cosine_similarity(x, y):
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return x_norm @ y_norm.T

class Mindex:
    """A class for indexing and searching document content from various sources.

    Mindex (short for Mind Index) provides functionality to download, parse,
    and index content from URLs or local files. It supports both HTML and PDF
    documents, breaking them into overlapping chunks for efficient searching.

    The class uses sentence transformers for embedding generation and a vector
    storage system for similarity search. It can handle multiple documents,
    maintaining information about their sources and dividing them into
    searchable chunks.

    Key features include:
    - Adding documents from URLs or local files
    - Automatic content type detection and parsing (HTML/PDF)
    - Text chunking with configurable overlap
    - Embedding generation using sentence transformers
    - Efficient similarity search on indexed content
    - Persistence through saving and loading of index state

    Attributes:
        NAME (str): Name of the index.
        CHUNK_SIZE (int): Size of text chunks for indexing.
        EMBEDDING_DIM (int): Dimensionality of the embedding vectors.
        model_id (str): Identifier for the sentence transformer model.
        storage (VectorStorage): Vector storage for embeddings and search.
        documents (List[Tuple[str, str]]): List of (title, url) pairs for indexed documents.
        chunks (List[str]): List of all text chunks across all documents.
        chunk_index (List[int]): Cumulative count of chunks per document.
    """

    def __init__(
        self,
        name: str,
        model_id: str = "mixedbread-ai/mxbai-embed-large-v1",
        reranker_id: str = "answerdotai/answerai-colbert-small-v1",
        EMBEDDING_DIM: int = 512,
        CHUNK_SIZE: int = 600,
        CHUNK_OVERLAP: int = 360,
        QUERY_PREFIX = "",
    ) -> None:
        self.NAME = name
        self.CHUNK_SIZE = CHUNK_SIZE
        self.CHUNK_OVERLAP = CHUNK_OVERLAP
        self.model_id = model_id

        self.storage = VectorStorage(
            embedder=SentenceTransformer(model_id, trust_remote_code=True, truncate_dim=EMBEDDING_DIM),
            similarity=SimilarityMetric.COSINE,
            query_prefix=QUERY_PREFIX,
            save_embedder=True,
            embedding_dim=EMBEDDING_DIM
        )
        self.reranker = Reranker(reranker_id)
        self.bm25 = BM25()

        self.documents: List[Tuple[str, str]] = []
        self.chunks: List[str] = []
        self.chunk_index: List[int] = [0]
        self.chunk_index: Array = np.zeros(1, dtype=np.int16)


    def add(
        self,
        urls: Union[Tuple[str, ...], List[str]] = [],
        filename: str = None,
        debug: bool = False,
    ):
        """Add document(s) to Mindex.

        Args:
            urls (Union[Tuple[str, ...], List[str]]): List of URLs to add.
            filename (str, optional): Path to a file containing URLs to add.
            debug (bool, optional): If True, prints debug information related to
                the download and parsing process. Defaults to False.

        """
        assert isinstance(urls, (tuple, list))
        assert urls != [] or filename is not None

        if filename:
            with open(filename, "r") as f:
                urls.extend([line.strip() for line in f.readlines()])

        new_chunks = []
        new_chunk_idxs = []
        prev_chunk_index = self.chunk_index[-1]
        for url in urls:
            if url in [doc[1] for doc in self.documents]:
                print(f"Skipped {url} as it already exists in the index.")
                continue

            try:
                title, text = self._download(url)
            except HTTPError as e:
                print(f"Error downloading {url}: {e}")
                continue

            self.documents.append((title, url))
            chunks, n_chunks = self._chunk(text)
            new_chunks.extend(chunks)
            new_chunk_idxs.append(prev_chunk_index + n_chunks)
            prev_chunk_index = new_chunk_idxs[-1]
            if debug:
                print(f"Added {title} from {url} with {n_chunks} chunks.")

        if not new_chunks:
            return

        self.chunk_index = np.concatenate([self.chunk_index, new_chunk_idxs])
        self.storage.index(new_chunks)
        self.chunks.extend(new_chunks)
        self.bm25.fit(self.chunks)

    def remove(self, url: str):
        """Remove a document from the index by URL."""
        index = next((i for i, doc in enumerate(self.documents) if doc[1] == url), None)
        if index is not None:
            self.documents.pop(index)

            c_s = self.chunk_index[index]
            c_e = self.chunk_index[index + 1]
            self.chunks = self.chunks[:c_s] + self.chunks[c_e:]

            self.chunk_index = np.delete(self.chunk_index, index + 1)
            self.chunk_index[index + 1 :] -= c_e - c_s

            self.storage.remove(c_s, c_e)
        else:
            print(f"Document with URL {url} not found in the index.")

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            mindex = pickle.load(f)
        return mindex

    def search(self, query: str, top_k: int, top_l: int = 1, method: str = 'hybrid', rerank: bool = True) -> Tuple[Array, Array, Array, Array, Array]:
        """
        Search the knowledge base for relevant chunks and their corresponding source documents.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.
            method (str): The search method to use. Defaults to 'hybrid'. 
                bm25: Okapi BM25
                embedding: Embeddings with cosine similarity
                hybrid': -

        Returns:
            Tuple[Array, Array, Array, Array, Array]: Top documents, doc scores, top document indices, top chunks, chunk scores.
        """
        assert top_k > 0 and top_k <= len(self.chunks)
        assert method in ['bm25', 'embedding', 'hybrid'], "Invalid search method"

        if method == 'bm25':
            top_k_chunks, chunk_scores = self._bm25_search(query, top_l)
        elif method == 'embedding':
            top_k_chunks, chunk_scores = self._embedding_search(query, top_l)
        elif method == 'hybrid':
            top_k_chunks, chunk_scores = self._hybrid_search(query, top_l)

        if rerank:
            chunks = [self.chunks[i] for i in top_k_chunks]
            ranked_result = self.reranker.rank(query=query, docs=chunks)
            top_k_chunks = [top_k_chunks[r.doc_id] for r in ranked_result.top_k(top_k)]
            chunk_scores = [chunk_scores[r.doc_id] for r in ranked_result.top_k(top_k)]

        # Match chunk to document, and score them.
        top_k_documents = (
            np.searchsorted(self.chunk_index, top_k_chunks, side="right") - 1
        )
        top_m_documents, _ = self._aggregate_and_sort(
            top_k_documents, chunk_scores
        )

        return top_m_documents, top_k_documents, top_k_chunks, chunk_scores


    def _embedding_search(self, query: str, top_k: int) -> Tuple[Array, Array]:
        top_k_chunks, chunk_scores = self.storage.search_top_k([query], top_k)
        return top_k_chunks.squeeze(), chunk_scores.squeeze()


    def _bm25_search(self, query: str, top_k: int) -> Tuple[Array, Array]:
        scores = self.bm25.get_scores(query)
        top_k_chunks = np.argpartition(-scores, top_k)[:top_k]
        return top_k_chunks, scores[top_k_chunks]


    def _hybrid_search_seq(self, query: str, top_k: int) -> Tuple[Array, Array]:
        """
        Hybrid search using BM25 and Embeddings sequentially.
        """
        bm25_scores = self.bm25.get_scores(query)
        
        top_l = top_k * 10
        top_l_chunks = np.argpartition(-bm25_scores, top_l)[:top_l]

        query_embeddings = self.storage._embedder.encode([query])
        chunk_embeddings = self.storage._index[top_l_chunks]

        chunk_scores = cosine_similarity(query_embeddings, chunk_embeddings).squeeze()

        top_k_indices = np.argsort(-chunk_scores)[:top_k]
        top_k_chunks = top_l_chunks[top_k_indices]
        chunk_scores = chunk_scores[top_k_indices]

        return top_k_chunks, chunk_scores
    
    def _hybrid_search_rrf(self, query: str, top_k: int) -> Tuple[Array, Array]:
        """
        Hybrid search using reciprocal rank fusion.
        """
        top_l = top_k

        bm25_scores = self.bm25.get_scores(query)
        bm25_top_l = np.argpartition(-bm25_scores, top_l)[:top_l]
        
        embedding_top_l, _ = self.storage.search_top_k([query], top_l)
        embedding_top_l = embedding_top_l.squeeze()
        
        # Combine results using Reciprocal Rank Fusion
        k = 60  # constant for RRF, can be tuned
        all_indices = np.unique(np.concatenate([bm25_top_l, embedding_top_l]))
        
        rrf_scores = np.zeros(len(all_indices))
        for i, idx in enumerate(all_indices):
            bm25_rank = np.where(bm25_top_l == idx)[0]
            emb_rank = np.where(embedding_top_l == idx)[0]
            
            bm25_score = 1 / (k + bm25_rank[0] + 1) if len(bm25_rank) > 0 else 0
            emb_score = 1 / (k + emb_rank[0] + 1) if len(emb_rank) > 0 else 0
            
            rrf_scores[i] = bm25_score + emb_score
        
        # Get top-k results based on RRF scores
        top_k_indices = np.argsort(-rrf_scores)[:top_k]
        top_k_chunks = all_indices[top_k_indices]
        final_scores = rrf_scores[top_k_indices]
        
        return top_k_chunks, final_scores
    
    def _hybrid_search(self, query: str, top_k: int, alpha: float = 0.5) -> Tuple[Array, Array]:
        """
        Hybrid search combining BM25 and embedding scores.
        """        
        
        top_l = top_k

        bm25_scores = self.bm25.get_scores(query)
        bm25_top_l = np.argpartition(-bm25_scores, top_l)[:top_l]
        bm25_top_scores = bm25_scores[bm25_top_l]
        
        embedding_top_l, embedding_scores = self.storage.search_top_k([query], top_l)
        embedding_top_l = embedding_top_l.squeeze()
        embedding_scores = embedding_scores.squeeze()
        
        all_indices = np.unique(np.concatenate([bm25_top_l, embedding_top_l]))
        
        bm25_scores_norm = np.zeros(len(all_indices))
        emb_scores_norm = np.zeros(len(all_indices))
        
        bm25_min, bm25_max = np.min(bm25_top_scores), np.max(bm25_top_scores)
        bm25_range = bm25_max - bm25_min
        
        emb_min, emb_max = np.min(embedding_scores), np.max(embedding_scores)
        emb_range = emb_max - emb_min
        
        for i, idx in enumerate(all_indices):
            if bm25_range != 0:
                bm25_scores_norm[i] = (bm25_scores[idx] - bm25_min) / bm25_range
            if idx in embedding_top_l and emb_range != 0:
                emb_index = np.where(embedding_top_l == idx)[0][0]
                emb_scores_norm[i] = (embedding_scores[emb_index] - emb_min) / emb_range
        
        combined_scores = alpha * bm25_scores_norm + (1 - alpha) * emb_scores_norm
        
        final_top_k = np.argsort(-combined_scores)[:top_k]
        final_scores = combined_scores[final_top_k]
        final_indices = all_indices[final_top_k]
        
        return final_indices, final_scores


    def _aggregate_and_sort(
        self, documents: Array, scores: Array
    ) -> Tuple[Array, Array]:
        """Aggregate chunk similarity scores by source document and sort the documents by new scores."""

        unique_docs, inverse_indices = np.unique(documents, return_inverse=True)
        aggregated_scores = np.bincount(inverse_indices, weights=scores)
        sorted_indices = np.argsort(-aggregated_scores)

        sorted_documents = unique_docs[sorted_indices]
        sorted_scores = aggregated_scores[sorted_indices]
        return sorted_documents, sorted_scores

    def _chunk(self, text: str) -> Tuple[List[str], int]:
        """Split documents into 50% overlapping segments."""
        words = text.split()
        chunks = [
            " ".join(words[i : i + self.CHUNK_SIZE])
            for i in range(0, len(words), self.CHUNK_OVERLAP)
        ]
        return chunks, len(chunks)

    def _chunk2(self, text: str) -> Tuple[List[str], int]:
        """Split documents into chunks based on '.\n' sequence or CHUNK_SIZE words."""
        chunks = []
        current_chunk = []
        word_count = 0
        current_word = ""

        for char in text:
            if char.isspace():
                if current_word:
                    current_chunk.append(current_word)
                    word_count += 1
                    current_word = ""

                if word_count >= self.CHUNK_SIZE or (
                    current_chunk and current_chunk[-1].endswith(".") and char == "\n"
                ):
                    chunks.append(" ".join(current_chunk).replace("\n", " ").strip())
                    current_chunk = []
                    word_count = 0
            else:
                current_word += char

        if current_word:
            current_chunk.append(current_word)
        if current_chunk:
            chunks.append(" ".join(current_chunk).replace("\n", " ").strip())

        return chunks, len(chunks)

    def _download(self, url: str) -> Tuple[str, str]:
        """Download content from the given URL, determine its type, and extract the title and text."""
        response = requests.get(url)
        response.raise_for_status()

        content_type = response.headers["Content-Type"]

        if "text/html" in content_type:
            return self._parse_html(response.content)
        elif "application/pdf" in content_type:
            return self._parse_pdf(response.content)
        else:
            raise Exception("Unsupported content type")

    def _parse_html(self, content: bytes) -> Tuple[str, str]:
        """Extract the title and text content from HTML data."""
        soup = BeautifulSoup(content, "html.parser")
        title = soup.find("title").text if soup.find("title") else ""
        text = self._clean_text(soup.get_text())
        return title, text

    def _parse_pdf(self, content: bytes) -> Tuple[str, str]:
        """Extract the title and text content from a PDF file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        doc = fitz.open(temp_file_path)
        text = " ".join([page.get_text() for page in doc])

        text = self._clean_text(text)

        # Extract title from metadata or the first block of text
        title = doc.metadata.get("title", self._extract_first_block_text(doc[0]))

        os.remove(temp_file_path)
        return title, text

    def _clean_text(self, text: str) -> str:
        text = text.replace("-\n", "")
        return text

    def _extract_first_block_text(self, page) -> str:
        blocks = page.get_text("blocks")
        return blocks[0][4].strip() if blocks else "unknown"
