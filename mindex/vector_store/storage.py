import numpy as np
from enum import Enum
from typing import List

class SimilarityMetric(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2

def cosine_similarity(x, y):
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return x_norm @ y_norm.T

def euclidean_similarity(x, y):
    return -np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=-1))

def manhattan_similarity(x, y):
    return -np.sum(np.abs(x[:, np.newaxis] - y), axis=-1)

sims = [cosine_similarity, euclidean_similarity, manhattan_similarity]

class VectorStorage:
    """
    A class for efficient storage and retrieval of vector embeddings.

    VectorStorage indexes documents using their vector representations and 
    provides fast similarity search functionality.
    """
    def __init__(self, embedder, embedding_dim, similarity = SimilarityMetric.COSINE, query_prefix = "", save_embedder = False):
        """
        Initialize the VectorStorage.

        Args:
            embedder: Object with encode method and truncate_dim attribute.
            similarity: Similarity metric for comparing embeddings.
            query_prefix: Prefix added to queries before encoding.
            save_embedder: If True, save embedder when pickling.
        """
        self._embedder = embedder
        self._similarity_fn = sims[similarity.value]
        self._query_prefix = query_prefix
        self._index = np.zeros((0, embedding_dim))
        self._save_embedder = save_embedder


    def index(self, docs: List[str]):
        """
        Add documents to the index by encoding them with the embedder.
        """
        new_embeddings = self._embedder.encode(docs, prompt_name="s2p_query")
        self._index = np.concatenate([self._index, new_embeddings])

    def remove(self, a: int, b: int = None):
        """
        Remove entries between indices [a, b). If b is None, remove element at index a.
        """
        assert 0 <= a < self._index.shape[0]
        i = slice(a, b) if b else a
        self._index = np.delete(self._index, i, axis=0)

    def search_top_k(self, queries: List[str], k: int = 10):
        """Batched search for top k similar entries in index."""
        queries = [self._query_prefix + query for query in queries]
        query_embeddings = self._embedder.encode(queries)
        similarities = self._similarity_fn(query_embeddings, self._index)

        def top_k_partitioned(x):
            partitioned = np.argpartition(x, -k)[-k:]
            return partitioned[np.argsort(x[partitioned])][::-1]

        top_k_indices = np.apply_along_axis(top_k_partitioned, 1, similarities)

        row_indices = np.arange(similarities.shape[0])[:, np.newaxis]
        return top_k_indices, similarities[row_indices, top_k_indices]


    def get_embedder(self):
        return self._embedder

    def set_embedder(self, embedder):
        """Method to set the embedder after unpickling if it wasn't saved."""
        self._embedder = embedder
    

    def __getstate__(self):
        """Custom method to define what gets pickled."""
        state = self.__dict__.copy()
        if not self._save_embedder:
            del state['_embedder']
        return state


    def __setstate__(self, state):
        """Custom method to define how the object is reconstructed when unpickled."""
        self.__dict__.update(state)
        if '_save_embedder' not in state or not state['_save_embedder']:
            self._embedder = None

