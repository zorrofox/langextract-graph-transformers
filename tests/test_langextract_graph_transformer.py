
import unittest
from unittest.mock import patch, MagicMock
import json
from langchain_core.documents import Document
from langextract_graph_transformers.langextract_graph_transformer import LangExtractGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument
import langextract as lx

class TestLangExtractGraphTransformer(unittest.TestCase):

    def setUp(self):
        self.project_id = "test-project"
        self.location = "test-location"

    @patch('langextract.extract')
    def test_extraction_with_properties(self, mock_extract):
        # Arrange
        transformer = LangExtractGraphTransformer(
            project_id=self.project_id,
            location=self.location,
            node_properties=["sector"],
            relationship_properties=["confidence"]
        )
        doc_content = "FirstEnergy (sector: Utilities) is a company."
        document = Document(page_content=doc_content)

        # Mock the JSON output that the LLM would return
        mock_graph_json = {
            "extractions": [
                {"id": "FirstEnergy", "type": "Company", "properties": {"sector": "Utilities"}},
                {"id": "FE", "type": "StockSymbol", "properties": {}},
                {"source": "FirstEnergy", "target": "FE", "type": "HAS_STOCK_SYMBOL", "properties": {"confidence": 0.9}}
            ]
        }
        mock_json_string = json.dumps(mock_graph_json)

        # Mock the return value of lx.extract
        mock_extraction_result = MagicMock()
        mock_extraction_result.extractions = [
            lx.data.Extraction(
                extraction_class="GraphJSON",
                extraction_text=mock_json_string
            )
        ]
        mock_extract.return_value = mock_extraction_result

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        self.assertEqual(len(graph_documents), 1)
        graph_document = graph_documents[0]
        self.assertIsInstance(graph_document, GraphDocument)

        # Verify the parsed graph content
        self.assertEqual(len(graph_document.nodes), 2)
        self.assertEqual(len(graph_document.relationships), 1)

        # Find the FirstEnergy node and check its properties
        first_energy_node = next((n for n in graph_document.nodes if n.id == "FirstEnergy"), None)
        self.assertIsNotNone(first_energy_node)
        # Verify that the property key is normalized and the value is a string
        self.assertEqual(first_energy_node.properties.get('prop_sector'), "Utilities")

        # Check relationship properties
        relationship = graph_document.relationships[0]
        # Verify that the property key is normalized and the value is a string
        self.assertEqual(relationship.properties.get('prop_confidence'), "0.9")

    @patch('langextract.extract')
    def test_extraction_handles_raw_list_output(self, mock_extract):
        # Arrange
        transformer = LangExtractGraphTransformer(
            project_id=self.project_id,
            location=self.location,
        )
        doc_content = "Apple is a company."
        document = Document(page_content=doc_content)

        # Mock the JSON output as a raw list, which was causing the AttributeError
        mock_graph_list = [
            {"id": "Apple", "type": "Company", "properties": {}},
        ]
        mock_json_string = json.dumps(mock_graph_list)

        # Mock the return value of lx.extract
        mock_extraction_result = MagicMock()
        mock_extraction_result.extractions = [
            lx.data.Extraction(
                extraction_class="GraphJSON",
                extraction_text=mock_json_string
            )
        ]
        mock_extract.return_value = mock_extraction_result

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        self.assertEqual(len(graph_documents), 1)
        graph_document = graph_documents[0]
        self.assertIsInstance(graph_document, GraphDocument)

        # Verify the parsed graph content is correct, even from a raw list
        self.assertEqual(len(graph_document.nodes), 1)
        self.assertEqual(len(graph_document.relationships), 0)
        self.assertEqual(graph_document.nodes[0].id, "Apple")
        self.assertEqual(graph_document.nodes[0].type, "Company")

if __name__ == '__main__':
    unittest.main()
