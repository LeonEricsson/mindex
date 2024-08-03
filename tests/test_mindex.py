import unittest
import numpy as np
from unittest.mock import patch
from mindex import Mindex

class TestMindex(unittest.TestCase):

    def setUp(self):
        self.mindex = Mindex("test_index", EMBEDDING_DIM=512, CHUNK_SIZE=200)
        self.urls = [
            "https://www.paulgraham.com/persistence.html",
            "https://www.paulgraham.com/reddits.html",
            "https://www.paulgraham.com/google.html",
            "https://www.paulgraham.com/best.html"
        ]

    @patch('mindex.Mindex._download')
    @patch('mindex.Mindex._chunk')
    def test_add(self, mock_chunk, mock_download):
        # Mock the _download and _chunk methods
        mock_download.side_effect = [
            ("Title 1", "Content 1"),
            ("Title 2", "Content 2"),
            ("Title 3", "Content 3"),
            ("Title 4", "Content 4")
        ]
        mock_chunk.side_effect = [
            (["Chunk 1", "Chunk 2"], 2),
            (["Chunk 3", "Chunk 4"], 2),
            (["Chunk 5", "Chunk 6"], 2),
            (["Chunk 7", "Chunk 8"], 2)
        ]

        self.mindex.add(self.urls)

        self.assertEqual(len(self.mindex.documents), 4)
        self.assertEqual(self.mindex.documents[0], ("Title 1", self.urls[0]))

        self.assertEqual(len(self.mindex.chunks), 8)
        self.assertEqual(self.mindex.chunks[0], "Chunk 1")

        np.testing.assert_array_equal(self.mindex.chunk_index, [0, 2, 4, 6, 8])

        self.assertEqual(self.mindex.storage._index.shape, (8, 512))

    def test_remove(self):
        with patch('mindex.Mindex._download') as mock_download, \
             patch('mindex.Mindex._chunk') as mock_chunk:
            mock_download.side_effect = [
                ("Title 1", "Content 1"),
                ("Title 2", "Content 2")
            ]
            mock_chunk.side_effect = [
                (["Chunk 1", "Chunk 2"], 2),
                (["Chunk 3", "Chunk 4"], 2)
            ]
            self.mindex.add(self.urls[:2])

        self.mindex.remove(self.urls[0])

        self.assertEqual(len(self.mindex.documents), 1)
        self.assertEqual(self.mindex.documents[0], ("Title 2", self.urls[1]))

        self.assertEqual(len(self.mindex.chunks), 2)
        self.assertEqual(self.mindex.chunks[0], "Chunk 3")

        np.testing.assert_array_equal(self.mindex.chunk_index, [0, 2])

        self.assertEqual(self.mindex.storage._index.shape, (2, 512))

    @patch('mindex.Mindex.search')
    def test_search(self, mock_search):
        # Create a mock for the search method
        mock_search.return_value = (
            np.array([0, 1, 2]),  # top_docs
            np.array([0.9, 0.8, 0.7]),  # doc_scores
            np.array([0, 2, 4]),  # top_chunks
            np.array([0.9, 0.8, 0.7])  # chunk_scores
        )

        mindex = Mindex("test_index", EMBEDDING_DIM=512, CHUNK_SIZE=200)

        top_docs, doc_scores, top_chunks, chunk_scores = mindex.search("test query", 3)

        mock_search.assert_called_once_with("test query", 3)

        np.testing.assert_array_equal(top_docs, [0, 1, 2])
        np.testing.assert_array_almost_equal(doc_scores, [0.9, 0.8, 0.7])
        np.testing.assert_array_equal(top_chunks, [0, 2, 4])
        np.testing.assert_array_almost_equal(chunk_scores, [0.9, 0.8, 0.7])

if __name__ == '__main__':
    unittest.main()