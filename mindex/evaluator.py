import json
from typing import List, Dict, Any
from nltk import ngrams

class Evaluator:
    """A helper class to evaluate Mindex search quality.

    This class provides methods to load, access, and evaluate data from JSON files
    containing validation and test sets for retrieval tasks. It includes
    functionality for retrieving queries, answers, and document URLs, as well as
    evaluating the quality of retrieved chunks against true answers.

    Attributes:
        dataset_name (str): The name of the dataset.
        validation_set (List[Dict]): The validation dataset.
        test_set (List[Dict]): The test dataset.
    """
    def __init__(self, file_path: str):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.dataset_name = data.get('dataset', '')
        self.validation_set = data.get('validation', [])
        self.test_set = data.get('test', [])


    def get_validation_set(self) -> List[str]:
        return [(item['query'], item['answer']) for item in self.validation_set]


    def get_validation_queries(self) -> List[str]:
        return [item['query'] for item in self.validation_set]


    def get_test_queries(self) -> List[str]:
        return [item['query'] for item in self.test_set]


    def get_validation_item_by_id(self, item_id: str) -> Dict[str, Any]:
        return next((item for item in self.validation_set if item['id'] == item_id), None)


    def get_test_item_by_id(self, item_id: str) -> Dict[str, Any]:
        return next((item for item in self.test_set if item['id'] == item_id), None)


    def get_answer_for_query(self, query: str, dataset: str = 'validation') -> Dict[str, Any]:
        target_set = self.validation_set if dataset == 'validation' else self.test_set
        item = next((item for item in target_set if item['query'] == query), None)
        return item['answer'] if item else None

    def get_document_urls(self) -> List[str]:
        validation_urls =  set([item['document_url'] for item in self.validation_set])
        test_urls = set([item['document_url'] for item in self.test_set])
        return list(validation_urls.union(test_urls))

    @staticmethod
    def calculate_ngram_overlap_score(retrieved_chunk: str, true_answer: str, n: int = 5) -> float:
        def get_ngrams(t: str, n: int):
            return set(ngrams(t.lower().split(), n))

        retrieved_ngrams = get_ngrams(retrieved_chunk, n)
        answer_ngrams = get_ngrams(true_answer, n)
        
        overlap = len(retrieved_ngrams.intersection(answer_ngrams))
        return overlap / len(answer_ngrams)

    def evaluate_retrieval(self, answer: str, retrieved_chunks: List[str], threshold: float = 0.7) -> Dict[str, Any]:
        scores = [self.calculate_ngram_overlap_score(chunk, answer) for chunk in retrieved_chunks]
        
        max_score = max(scores)
        best_chunk_index = scores.index(max_score)
        is_answer_found = max_score >= threshold
        
        return {
            'scores': scores,
            'max_score': max_score,
            'best_chunk_index': best_chunk_index,
            'is_answer_found': is_answer_found
        }