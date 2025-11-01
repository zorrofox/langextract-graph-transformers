
import unittest
from unittest.mock import patch, MagicMock, ANY, call
import json
import hashlib
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

class TestSpannerSchemalessGraphStore(unittest.TestCase):

    def _get_mock_db(self, MockSpannerClient):
        mock_instance = MagicMock()
        mock_database = MagicMock()
        mock_snapshot = MagicMock()
        mock_client = MockSpannerClient.return_value
        mock_client.instance.return_value = mock_instance
        mock_instance.database.return_value = mock_database
        mock_database.snapshot.return_value.__enter__.return_value = mock_snapshot
        mock_database.run_in_transaction.side_effect = lambda func, *args, **kwargs: func(MagicMock(), *args, **kwargs)
        return mock_client, mock_database, mock_snapshot

    @patch("google.cloud.spanner_v1.Client")
    def test_initialization_creates_schema(self, MockSpannerClient):
        """Tests that DDLs are executed if the schema does not exist."""
        _, mock_database, mock_snapshot = self._get_mock_db(MockSpannerClient)
        mock_ddl_op = MagicMock()
        mock_database.update_ddl.return_value = mock_ddl_op
        mock_snapshot.execute_sql.return_value = []

        SpannerSchemalessGraph(
            instance_id="test-instance", database_id="test-db", client=MockSpannerClient()
        )

        self.assertEqual(mock_database.update_ddl.call_count, 2)

    @patch("google.cloud.spanner_v1.Client")
    def test_add_graph_documents_logic(self, MockSpannerClient):
        """Tests the logic of converting GraphDocuments to Spanner mutations."""
        mock_client, _, _ = self._get_mock_db(MockSpannerClient)
        
        # Bypass schema creation for this test by mocking the method
        with patch.object(SpannerSchemalessGraph, '_create_or_verify_schema') as mock_create_schema:
            graph_store = SpannerSchemalessGraph(
                instance_id="test-instance", database_id="test-db", client=mock_client
            )
            mock_create_schema.assert_called_once()

            # Prepare graph documents
            source_doc = Document(page_content="Source text")
            node1 = Node(id="Google", type="Company", properties={"country": "USA"})
            node2 = Node(id="Sundar Pichai", type="Person")
            relationship = Relationship(source=node1, target=node2, type="IS_CEO_OF", properties={"start_year": 2015})
            graph_doc = GraphDocument(source=source_doc, nodes=[node1, node2], relationships=[relationship])

            # Mock the transaction to capture the mutations
            mock_transaction = MagicMock()
            graph_store._database.run_in_transaction = MagicMock(side_effect=lambda func: func(mock_transaction))

            # Act
            graph_store.add_graph_documents([graph_doc])

            # Assert
            graph_store._database.run_in_transaction.assert_called_once()
            
            # Check node mutations
            node_call = mock_transaction.insert_or_update.call_args_list[0]
            self.assertEqual(node_call.kwargs['table'], 'GraphNode')
            self.assertEqual(node_call.kwargs['columns'], ['id', 'label', 'properties'])
            
            values = list(node_call.kwargs['values'])
            self.assertEqual(len(values), 2)
            
            google_node_val = next(v for v in values if v[1] == "Company")
            sundar_node_val = next(v for v in values if v[1] == "Person")

            self.assertEqual(json.loads(google_node_val[2])['country'], 'USA')

            # Check edge mutations
            edge_call = mock_transaction.insert_or_update.call_args_list[1]
            self.assertEqual(edge_call.kwargs['table'], 'GraphEdge')
            self.assertEqual(edge_call.kwargs['columns'], ['id', 'dest_id', 'edge_id', 'label', 'properties'])
            
            edge_values = edge_call.kwargs['values'][0]
            self.assertEqual(edge_values[0], graph_store._get_int64_hash("Company-Google"))
            self.assertEqual(edge_values[1], graph_store._get_int64_hash("Person-Sundar Pichai"))
            self.assertEqual(edge_values[3], 'IS_CEO_OF')
            self.assertEqual(json.loads(edge_values[4])['start_year'], 2015)

    @patch("google.cloud.spanner_v1.Client")
    def test_add_graph_documents_with_native_types(self, MockSpannerClient):
        """Tests that native types in properties are passed as dicts, not JSON strings."""
        mock_client, _, _ = self._get_mock_db(MockSpannerClient)
        
        with patch.object(SpannerSchemalessGraph, '_create_or_verify_schema'):
            graph_store = SpannerSchemalessGraph(
                instance_id="test-instance", database_id="test-db", client=mock_client
            )

            node = Node(id="N1", type="Type1", properties={"value": 123, "active": False})
            graph_doc = GraphDocument(source=Document(page_content=""), nodes=[node], relationships=[])

            mock_transaction = MagicMock()
            graph_store._database.run_in_transaction = MagicMock(side_effect=lambda func: func(mock_transaction))

            graph_store.add_graph_documents([graph_doc])

            mock_transaction.insert_or_update.assert_called_once()
            call_kwargs = mock_transaction.insert_or_update.call_args.kwargs
            
            # The crucial check: ensure the 'properties' value is a dict, not a string
            passed_values = call_kwargs['values'][0]
            properties_value = passed_values[2]
            self.assertIsInstance(properties_value, str)
            self.assertEqual(json.loads(properties_value)['value'], 123)
            self.assertEqual(json.loads(properties_value)['active'], False)

if __name__ == "__main__":
    unittest.main()
