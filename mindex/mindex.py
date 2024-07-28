import os
import pickle
import requests
import tempfile
from typing import Union, Tuple, List

import numpy as np
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from vector_store import VectorStorage, SimilarityMetric

Array = np.ndarray

class Mindex:
    def __init__(self, name: str, model_id: str = "mixedbread-ai/mxbai-embed-large-v1", EMBEDDING_DIM: int = 512, CHUNK_SIZE: int = 200, QUERY_PREFIX = '') -> None:
        self.NAME = name
        self.CHUNK_SIZE = CHUNK_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.model_id = model_id

        self.storage = VectorStorage(
            embedder=SentenceTransformer(model_id, truncate_dim=EMBEDDING_DIM),
            similarity=SimilarityMetric.COSINE,
            query_prefix=QUERY_PREFIX,
            save_embedder=False
        )


        self.documents: List[Tuple[str, str]] = []
        self.chunks: List[str] = []
        self.chunk_index: List[int] = [0]

    def add(self, urls: Union[Tuple[str, ...], List[str]] = [], filename: str = None):
        """Add document(s) to Mindex.

        Args:
            urls (Union[Tuple[str, ...], List[str]]): List of URLs to add.
            filename (str, optional): Path to a file containing URLs to add.

        
        """
        assert isinstance(urls, (tuple, list))
        assert urls != [] or filename is not None
        
        if filename:
            with open(filename, 'r') as f:
                urls.extend([line.strip() for line in f.readlines()])

        new_chunks = []
        for url in urls:
            if url in [doc[1] for doc in self.documents]:
                print(f"Skipped {url} as it already exists in the index.")
                continue

            title, text = self._download(url)
            self.documents.append((title, url))
            chunks, n_chunks = self._chunk(text)
            new_chunks.extend(chunks)
            self.chunk_index.append(self.chunk_index[-1] + n_chunks)

        assert new_chunks != []
        
        self.storage.index(new_chunks)
        self.chunks.extend(new_chunks)

        self.save(f"{self.NAME}.pkl")

    
    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, filename: str):
        with open(filename, 'rb') as f:
            obj = pickle.load(f)
            
            # reload embedding model
            obj.storage.set_embedder(
                SentenceTransformer(
                    obj.model_id, 
                    truncate_dim=obj.EMBEDDING_DIM
                    )
                )
            return obj

    def search(self, query: str, top_k: int) -> Tuple[Array, Array, Array]:
        """
        Search the embedding database for the most relevant documents to the query.

        Args:
            query (str): The search query.
            top_k (int): The number of top results to return.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Top documents, doc scores, top chunks, chunk scores.
        """
        assert top_k > 0 and top_k <= len(self.chunks)
        
        top_k_chunks, chunk_scores = self.storage.search_top_k([query], top_k)
        top_k_chunks = top_k_chunks.squeeze()
        chunk_scores = chunk_scores.squeeze()

        # connect chunks to documents
        top_k_documents = np.searchsorted(self.chunk_index, top_k_chunks, side='right') 
        top_m_documents, document_scores = self._aggregate_and_sort(top_k_documents, chunk_scores) # m <= k

        return top_m_documents.squeeze(), document_scores.squeeze(), top_k_chunks, chunk_scores


    def _aggregate_and_sort(self, documents: Array, scores: Array) -> Tuple[Array, Array]:
        """ Aggregate chunk similarity scores by source document and sort the documents by new scores."""

        unique_docs, inverse_indices = np.unique(documents, return_inverse=True)    
        aggregated_scores = np.bincount(inverse_indices, weights=scores)
        sorted_indices = np.argsort(-aggregated_scores)
        
        sorted_documents = unique_docs[sorted_indices]
        sorted_scores = aggregated_scores[sorted_indices]
        return sorted_documents, sorted_scores
    

    def _chunk(self, text: str) -> Tuple[List[str], int]:
        """Split documents into 50% overlapping segments."""
        words = text.split()
        chunks = [' '.join(words[i:i+self.CHUNK_SIZE]) for i in range(0, len(words), self.CHUNK_SIZE // 2)]
        return chunks, len(chunks)

    
    def _download(self, url: str) -> Tuple[str, str]:
        """Download content from the given URL, determine its type, and extract the title and text."""
        response = requests.get(url)
        response.raise_for_status() 

        content_type = response.headers['Content-Type']

        if 'text/html' in content_type:
            return self._parse_html(response.content)
        elif 'application/pdf' in content_type:
            return self._parse_pdf(response.content)
        else:
            raise Exception('Unsupported content type')

    def _parse_html(self, content: bytes) -> Tuple[str, str]:
        """Extract the title and text content from HTML data."""
        soup = BeautifulSoup(content, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No Title'
        text = self._clean_text(soup.get_text())
        return title, text

    def _parse_pdf(self, content: bytes) -> Tuple[str, str]:
        """Extract the title and text content from a PDF file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        doc = fitz.open(temp_file_path)
        text = ' '.join([page.get_text() for page in doc])
        text = self._clean_text(text)

        # Extract title from metadata or the first block of text
        title = doc.metadata.get('title', self._extract_first_block_text(doc[0]))

        os.remove(temp_file_path)
        return title, text

    def _clean_text(self, text: str) -> str:
        text = text.replace('\n', ' ')
        text = text.replace('- ', '')
        return ' '.join(text.split())


    def _extract_first_block_text(self, page) -> str:
        blocks = page.get_text("blocks")
        return blocks[0][4].strip() if blocks else 'unknown'    
    