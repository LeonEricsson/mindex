import numpy as np
from enum import Enum

from jaxtyping import Float
from typing import List

class SimilarityMetric(Enum):
    COSINE = 0
    EUCLIDEAN = 1
    MANHATTAN = 2

def cosine_similarity(x, y) -> Float:
    x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
    y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)
    return x_norm @ y_norm.T

def euclidean_similarity(x, y):
    return -np.sqrt(np.sum((x[:, np.newaxis] - y) ** 2, axis=-1))

def manhattan_similarity(x, y):
    return -np.sum(np.abs(x[:, np.newaxis] - y), axis=-1)

sims = [cosine_similarity, euclidean_similarity, manhattan_similarity]

class VectorStorage:
    def __init__(self, embedder, similarity = SimilarityMetric.COSINE, query_prefix = ""):
        self.embedder = embedder
        self.similarity = sims[similarity.value]
        self.query_prefix = query_prefix
    
    def index(self, docs: List[str]):
        self.index = self.embedder.encode(docs)

    def search_top_k(self, queries: List[str], k: int = 10):
        """Batched search for top k similar documents to each query."""
        queries = [self.query_prefix + query for query in queries]
        query_embeddings = self.embedder.encode(queries)
        similarities = self.similarity(query_embeddings, self.index)

        top_k_indices = np.apply_along_axis(lambda x: np.argsort(x)[-k:][::-1], 1, similarities)

        row_indices = np.arange(similarities.shape[0])[:, np.newaxis]
        return top_k_indices, similarities[row_indices, top_k_indices]
    