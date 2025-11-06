import unittest
from unittest.mock import MagicMock, patch

from google.cloud import spanner
from langchain_core.embeddings import FakeEmbeddings

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_google_spanner.graph_vector_store import SpannerGraphVectorStore
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph


class TestSpannerGraphVectorStore(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.mock_snapshot = MagicMock()
        self.mock_database = MagicMock()
        self.mock_database.snapshot.return_value.__enter__.return_value = (
            self.mock_snapshot
        )
        self.mock_graph = MagicMock(spec=SpannerSchemalessGraph)
        self.mock_graph.node_table = "test_node_table"
        self.mock_graph._database = self.mock_database

    def test_get_value_by_path(self):
        """Tests the _get_value_by_path static method."""
        data = {
            "top_level": "value1",
            "nested": {
                "level2": "value2",
                "another": {
                    "level3": "value3"
                }
            }
        }
        self.assertEqual(SpannerGraphVectorStore._get_value_by_path(data, "top_level"), "value1")
        self.assertEqual(SpannerGraphVectorStore._get_value_by_path(data, "nested.level2"), "value2")
        self.assertEqual(SpannerGraphVectorStore._get_value_by_path(data, "nested.another.level3"), "value3")
        self.assertIsNone(SpannerGraphVectorStore._get_value_by_path(data, "non.existent.path"))
        self.assertIsNone(SpannerGraphVectorStore._get_value_by_path(data, "top_level.non_existent"))

    def test_similarity_search_by_vector_sql_generation(self):
        """
        Tests if similarity_search_by_vector generates the correct SQL query and parameters.
        """
        # 1. Setup Mocks
        embedding_service = FakeEmbeddings(size=8)

        # 2. Instantiate the VectorStore
        vector_store = SpannerGraphVectorStore(
            graph=self.mock_graph,
            embedding=embedding_service,
            text_properties=['text'],
        )

        # 3. Call the method to be tested
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        vector_store.similarity_search_by_vector(embedding=test_embedding, k=5, node_label="document")

        # 4. Assertions
        # Check if execute_sql was called
        self.mock_snapshot.execute_sql.assert_called_once()

        # Get the actual arguments passed to execute_sql
        args, kwargs = self.mock_snapshot.execute_sql.call_args
        
        # Extract query, params, and param_types from the call
        actual_query = args[0]
        actual_params = kwargs.get('params', {})
        actual_param_types = kwargs.get('param_types', {})

        # Define the expected query string
        expected_query = f"""
        SELECT properties, embedding
        FROM {self.mock_graph.node_table}
        WHERE embedding IS NOT NULL AND label = @node_label
        ORDER BY COSINE_DISTANCE(embedding, @query_embedding)
        LIMIT @limit
        """
        
        # Define expected parameters
        expected_params = {
            "node_label": "document",
            "query_embedding": test_embedding,
            "limit": 5,
        }
        
        # Define expected parameter types
        expected_param_types = {
            "node_label": spanner.param_types.STRING,
            "query_embedding": spanner.param_types.Array(spanner.param_types.FLOAT64),
            "limit": spanner.param_types.INT64,
        }

        # Assert the query string is correct (ignoring whitespace differences)
        self.assertEqual(
            ' '.join(expected_query.split()), 
            ' '.join(actual_query.split())
        )
        
        # Assert parameters and their types are correct
        self.assertDictEqual(expected_params, actual_params)
        self.assertDictEqual(expected_param_types, actual_param_types)


    @patch("langchain_core.embeddings.FakeEmbeddings.embed_documents")
    def test_from_existing_graph_batching(self, mock_embed_documents):
        """Tests the batching and pagination logic in from_existing_graph."""
        # 1. Setup Mocks
        # Mock the result stream to simulate fetching data in multiple batches
        self.mock_snapshot.execute_sql.side_effect = [
            [
                (1, "doc", {"text": "Content 1"}),
                (2, "doc", {"text": "Content 2"}),
            ],
            [
                (3, "person", {"name": "John Doe"}),
            ],
            [] # Terminate the loop
        ]

        embedding_service = FakeEmbeddings(size=8)
        mock_embed_documents.side_effect = [[[0.1]*8, [0.2]*8], [[0.3]*8]]

        # 2. Call the method with empty text_properties to test label-only embedding
        SpannerGraphVectorStore.from_existing_graph(
            graph=self.mock_graph,
            embedding=embedding_service,
            text_properties=[], # Test label-only embedding
            include_label_in_embedding=True,
            batch_size=2 # Set batch size to 2
        )

        # 3. Assertions
        self.assertEqual(self.mock_snapshot.execute_sql.call_count, 3)
        self.assertEqual(mock_embed_documents.call_count, 2)
        self.assertEqual(self.mock_graph._database.run_in_transaction.call_count, 2)

        # Assert content of the first embedding call (should be labels only)
        first_call_args, _ = mock_embed_documents.call_args_list[0]
        self.assertEqual(["doc", "doc"], first_call_args[0])

        # Assert content of the second embedding call (should be labels only)
        second_call_args, _ = mock_embed_documents.call_args_list[1]
        self.assertEqual(["person"], second_call_args[0])


    def test_similarity_search_by_vector_across_all_nodes(self):
        """Test similarity search by vector across all nodes."""
        # Arrange
        embedding_service = FakeEmbeddings(size=2)
        vector_store = SpannerGraphVectorStore(
            graph=self.mock_graph,
            embedding=embedding_service,
            text_properties=["text"],
        )
        query_embedding = [1.0, 2.0]

        # Act
        vector_store.similarity_search_by_vector(query_embedding, k=5)

        # Assert
        self.mock_snapshot.execute_sql.assert_called_once()
        call_args, call_kwargs = self.mock_snapshot.execute_sql.call_args
        
        # Check the query string
        expected_query = f"""
        SELECT properties, embedding
        FROM {self.mock_graph.node_table}
        WHERE embedding IS NOT NULL
        ORDER BY COSINE_DISTANCE(embedding, @query_embedding)
        LIMIT @limit
        """
        self.assertEqual(
            ' '.join(call_args[0].strip().split()), 
            ' '.join(expected_query.strip().split())
        )

        # Check the parameters
        self.assertEqual(call_kwargs["params"]["query_embedding"], query_embedding)
        self.assertEqual(call_kwargs["params"]["limit"], 5)
        self.assertNotIn("node_label", call_kwargs["params"])


if __name__ == "__main__":
    unittest.main()