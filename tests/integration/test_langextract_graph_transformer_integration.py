
import unittest
import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langextract_graph_transformers.langextract_graph_transformer import LangExtractGraphTransformer
from langchain_community.graphs.graph_document import GraphDocument

# Load environment variables from .env file
load_dotenv()

class TestLangExtractGraphTransformerIntegration(unittest.TestCase):

    def test_arbitrary_extraction_with_properties(self):
        # Arrange
        project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        location = os.getenv("VERTEX_AI_LOCATION")

        if not project_id or not location:
            self.skipTest("VERTEX_AI_PROJECT_ID and VERTEX_AI_LOCATION environment variables must be set for integration tests")

        text_content = ("FirstEnergy (NYSE:FE), a major utilities provider, posted its earnings results on Tuesday. "
                        "The company reported $0.53 earnings per share for the quarter.")
        document = Document(page_content=text_content)

        transformer = LangExtractGraphTransformer(
            project_id=project_id,
            location=location,
            node_properties=["sector"], # Ask for the sector of a company
            relationship_properties=["quarter"] # Ask for the quarter of the earnings report
        )

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        self.assertIsInstance(graph_documents, list)
        self.assertEqual(len(graph_documents), 1)
        graph_document = graph_documents[0]
        self.assertIsInstance(graph_document, GraphDocument)
        
        self.assertGreater(len(graph_document.nodes), 0, "No nodes were extracted.")
        self.assertGreater(len(graph_document.relationships), 0, "No relationships were extracted.")

        # Find the FirstEnergy node and check for its sector property
        first_energy_node = next((n for n in graph_document.nodes if n.id == "FirstEnergy"), None)
        self.assertIsNotNone(first_energy_node, "FirstEnergy node not found.")
        self.assertIn("prop_sector", first_energy_node.properties, "'prop_sector' property missing from FirstEnergy node.")
        self.assertIsNotNone(first_energy_node.properties.get("prop_sector"), "'prop_sector' property should have a value.")

        # Find the REPORTED_EARNINGS relationship and check for its quarter property
        reported_earnings_rel = next((r for r in graph_document.relationships if r.type == "REPORTED_EARNINGS"), None)
        self.assertIsNotNone(reported_earnings_rel, "REPORTED_EARNINGS relationship not found.")
        self.assertIn("prop_quarter", reported_earnings_rel.properties, "'prop_quarter' property missing from REPORTED_EARNINGS relationship.")
        self.assertIsNotNone(reported_earnings_rel.properties.get("prop_quarter"), "'prop_quarter' property should have a value.")

if __name__ == '__main__':
    unittest.main()
