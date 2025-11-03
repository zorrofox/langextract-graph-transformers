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
    def test_similarity_search_by_vector_sql_generation(self):
        """
        Tests if similarity_search_by_vector generates the correct SQL query and parameters.
        """
        # 1. Setup Mocks
        mock_graph = MagicMock()
        mock_graph.node_table = "MockNodes"
        
        # Mock the database snapshot and execute_sql call
        mock_snapshot = MagicMock()
        mock_graph._database.snapshot.return_value.__enter__.return_value = mock_snapshot
        
        embedding_service = FakeEmbeddings(size=8)

        # 2. Instantiate the VectorStore
        vector_store = SpannerGraphVectorStore(
            graph=mock_graph,
            embedding=embedding_service,
            node_label="document",
            text_properties=['text'],
            embedding_property="embedding"
        )

        # 3. Call the method to be tested
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        vector_store.similarity_search_by_vector(embedding=test_embedding, k=5)

        # 4. Assertions
        # Check if execute_sql was called
        mock_snapshot.execute_sql.assert_called_once()

        # Get the actual arguments passed to execute_sql
        args, kwargs = mock_snapshot.execute_sql.call_args
        
        # Extract query, params, and param_types from the call
        actual_query = args[0]
        actual_params = kwargs.get('params', {})
        actual_param_types = kwargs.get('param_types', {})

        # Define the expected query string
        expected_query = """
        SELECT properties
        FROM MockNodes
        WHERE label = @node_label
        ORDER BY COSINE_DISTANCE(FLOAT64_ARRAY(JSON_QUERY(properties, '$.embedding')), @query_embedding)
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

if __name__ == "__main__":
    unittest.main()