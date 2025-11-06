import unittest
from unittest.mock import patch, MagicMock
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gemini_graph_transformers.gemini_graph_transformer import GeminiGraphTransformer
from langchain_core.documents import Document

# This allows us to mock the genai Client and Model classes
@patch('google.genai.Client')
class TestGeminiGraphTransformer(unittest.TestCase):

    def test_process_document_with_correct_client_usage(self, MockGenaiClient):
        # Arrange
        # 1. Mock the client and its nested `models` object
        mock_client_instance = MockGenaiClient.return_value
        mock_models_object = MagicMock()
        mock_client_instance.models = mock_models_object

        # 2. Mock the final response from generate_content
        mock_response = MagicMock()
        mock_graph_data = [
            {"id": "TestCorp", "type": "Company", "properties": {"location": "Testville"}}
        ]
        mock_response.text = json.dumps(mock_graph_data)
        mock_models_object.generate_content.return_value = mock_response

        # 3. Initialize the transformer
        transformer = GeminiGraphTransformer(
            project_id="test-project",
            location="test-location",
            node_properties=["location"]
        )
        document = Document(page_content="TestCorp is in Testville.")

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        # 1. Verify the client was configured correctly for Vertex AI
        MockGenaiClient.assert_called_once_with(vertexai=True, project="test-project", location="test-location")

        # 3. Verify generate_content was called with JSON mode
        mock_models_object.generate_content.assert_called_once()
        call_args, call_kwargs = mock_models_object.generate_content.call_args
        config = call_kwargs.get('config')
        self.assertIsNotNone(config)
        self.assertEqual(config.response_mime_type, "application/json")

        # 3. Verify the final parsed graph
        self.assertEqual(len(graph_documents), 1)
        graph = graph_documents[0]
        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.nodes[0].id, "TestCorp")
        self.assertEqual(graph.nodes[0].properties['location'], "Testville")

    def test_preserves_native_types(self, MockGenaiClient):
        """Tests that native data types (int, bool) are preserved in properties."""
        # Arrange
        mock_client_instance = MockGenaiClient.return_value
        mock_models_object = MagicMock()
        mock_client_instance.models = mock_models_object

        mock_response = MagicMock()
        mock_graph_data = [
            {
                "id": "TestNode",
                "type": "TestType",
                "properties": {
                    "name": "Test Name",
                    "age": 42,
                    "is_active": True
                }
            }
        ]
        mock_response.text = json.dumps(mock_graph_data)
        mock_models_object.generate_content.return_value = mock_response

        transformer = GeminiGraphTransformer(project_id="test-project", location="test-location")
        document = Document(page_content="Some text")

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        self.assertEqual(len(graph_documents), 1)
        graph = graph_documents[0]
        self.assertEqual(len(graph.nodes), 1)
        node = graph.nodes[0]

        self.assertEqual(node.properties['name'], "Test Name")
        self.assertIsInstance(node.properties['age'], int)
        self.assertEqual(node.properties['age'], 42)
        self.assertIsInstance(node.properties['is_active'], bool)
        self.assertEqual(node.properties['is_active'], True)

if __name__ == "__main__":
    unittest.main()
