import unittest
import os
from dotenv import load_dotenv

from langchain_core.documents import Document

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gemini_graph_transformers.gemini_graph_transformer import GeminiGraphTransformer

load_dotenv()

class TestGeminiGraphTransformerIntegration(unittest.TestCase):

    def setUp(self):
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        if not self.project_id:
            self.skipTest("Integration tests require VERTEX_AI_PROJECT_ID.")

    def test_e2e_extraction_with_correct_sdk(self):
        """Tests the full end-to-end extraction using the correct Vertex AI SDK configuration."""
        # Arrange
        transformer = GeminiGraphTransformer(
            project_id=self.project_id,
            location=self.location,
            node_properties=["headquarters", "sector"],
            relationship_properties=["date", "value_usd"]
        )

        text_content = ("In a major tech deal, Microsoft, a software company based in Redmond, "
                        "officially acquired GitHub for $7.5 billion on October 26, 2018.")
        document = Document(page_content=text_content)

        # Act
        graph_documents = transformer.process_documents([document])

        # Assert
        self.assertEqual(len(graph_documents), 1)
        graph = graph_documents[0]

        self.assertGreater(len(graph.nodes), 1, "Should have extracted at least two nodes.")
        self.assertGreater(len(graph.relationships), 0, "Should have extracted at least one relationship.")

        # Find specific nodes and check their properties
        msft_node = next((n for n in graph.nodes if n.id == "Microsoft"), None)
        github_node = next((n for n in graph.nodes if n.id == "GitHub"), None)

        self.assertIsNotNone(msft_node, "Microsoft node was not extracted.")
        self.assertIsNotNone(github_node, "GitHub node was not extracted.")
        self.assertEqual(msft_node.type, "Company")
        self.assertIn("prop_headquarters", msft_node.properties)
        self.assertEqual(msft_node.properties["prop_headquarters"], "Redmond")

        # Find the specific relationship and check its properties
        acquisition_rel = next((r for r in graph.relationships if r.type == "ACQUIRED"), None)
        self.assertIsNotNone(acquisition_rel, "ACQUIRED relationship was not extracted.")
        self.assertEqual(acquisition_rel.source.id, "Microsoft")
        self.assertEqual(acquisition_rel.target.id, "GitHub")
        self.assertIn("prop_date", acquisition_rel.properties)
        self.assertEqual(acquisition_rel.properties["prop_date"], "October 26, 2018")

if __name__ == "__main__":
    unittest.main()