import numpy as np
from typing import List
from collections import Counter

class BM25:
    """A class implementing the Okapi BM25 ranking function.

    BM25 is a bag-of-words retrieval function that ranks a set of documents based on
    the query terms appearing in each document, regardless of their proximity within
    the document. It is a probabilistic model that extends the binary independence model.

    Attributes:
        k1 (float): A positive tuning parameter that calibrates document term frequency scaling.
        b (float): A tuning parameter (0 ≤ b ≤ 1) which determines the scaling by document length.
        idf (np.ndarray): Inverse document frequency scores for each term in the vocabulary.
        doc_len (np.ndarray): Length of each document in the corpus.
        avgdl (float): Average document length in the corpus.
        doc_freqs (np.ndarray): Term frequency matrix for the corpus.
        corpus_size (int): Number of documents in the corpus.
        vocabulary (dict): Mapping of terms to their indices in the vocabulary.
    """
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.idf = None
        self.doc_len = None
        self.avgdl = None
        self.doc_freqs = None
        self.corpus_size = None
        self.vocabulary = None


    def fit(self, corpus: List[str]):
        """Fit the BM25 model to a given corpus.

        This method preprocesses the corpus, builds the vocabulary, computes document
        frequencies, and calculates the inverse document frequency (IDF) scores.

        Args:
            corpus (List[str]): A list of documents, where each document is a string.

        """
        self.corpus_size = len(corpus)
        
        # Tokenize all documents once
        tokenized_corpus = [doc.split() for doc in corpus]
        
        self.doc_len = np.array([len(tokens) for tokens in tokenized_corpus], dtype=np.int16)
        self.avgdl = np.mean(self.doc_len)
        
        # Create vocabulary and document-term matrix
        vocab = set()
        for tokens in tokenized_corpus:
            vocab.update(tokens)
        self.vocabulary = {term: i for i, term in enumerate(vocab)}
        
        self.doc_freqs = np.zeros((self.corpus_size, len(self.vocabulary)), dtype=np.int16)
        for i, tokens in enumerate(tokenized_corpus):
            for term, freq in Counter(tokens).items():
                self.doc_freqs[i, self.vocabulary[term]] = freq
        
        # Calculate IDF
        df = np.sum(self.doc_freqs > 0, axis=0)
        self.idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)


    def get_scores(self, query: str) -> np.ndarray:
        """Calculate BM25 scores for a given query against all documents in the corpus.

        Args:
            query (str): The search query string.

        Returns:
            np.ndarray: An array of BM25 scores for each document in the corpus.

        """
        query_terms = query.split()
        query_freqs = np.zeros(len(self.vocabulary), dtype=np.float32)

        # q is so small that removing this is not worth it
        for term in query_terms:
            if term in self.vocabulary:
                query_freqs[self.vocabulary[term]] += 1
        
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        non_zero_query_terms = query_freqs > 0
        
        term_scores = (
            self.idf[non_zero_query_terms] *
            (self.doc_freqs[:, non_zero_query_terms] * (self.k1 + 1)) /
            (self.doc_freqs[:, non_zero_query_terms] + self.k1 * (1 - self.b + self.b * self.doc_len[:, np.newaxis] / self.avgdl))
        )
        
        scores = np.sum(term_scores, axis=1)


        return scores