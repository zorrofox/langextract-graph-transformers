import unittest
import os
import json
import hashlib
from google import genai
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

# Correct, cross-platform path joining
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from gemini_graph_transformers.gemini_graph_transformer import GeminiGraphTransformer
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph

load_dotenv()

class TestFullPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")
        self.location = os.getenv("VERTEX_AI_LOCATION", "us-central1")
        self.instance_id = os.getenv("SPANNER_INSTANCE_ID")
        self.database_id = os.getenv("SPANNER_DATABASE_ID")
        self.node_table = "GraphNode"
        self.edge_table = "GraphEdge"
        self.graph_name = "FinGraph"

        self.graph_store = SpannerSchemalessGraph(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            node_table=self.node_table,
            edge_table=self.edge_table,
            graph_name=self.graph_name,
        )

        self.transformer = GeminiGraphTransformer(
            project_id=self.project_id,
            location=self.location,
            node_properties=["sector", "location"], 
            relationship_properties=["date"]
        )

    def tearDown(self):
        self.graph_store.cleanup()

    def _get_int64_hash(self, input_string: str) -> int:
        sha256_hash = hashlib.sha256(input_string.encode('utf-8')).digest()
        return int.from_bytes(sha256_hash[:8], byteorder='big', signed=True)

    def test_extraction_to_storage_e2e(self):
        """Tests the full pipeline from text -> GraphDocument -> Spanner -> Query."""
        text_content = ("In a major tech deal, Microsoft, a software company based in Redmond, "
                        "officially acquired GitHub for $7.5 billion on October 26, 2018.")
        document = Document(page_content=text_content)

        graph_documents = self.transformer.process_documents([document])
        
        self.assertEqual(len(graph_documents), 1)
        self.assertGreater(len(graph_documents[0].nodes), 1)
        self.assertGreater(len(graph_documents[0].relationships), 0)

        self.graph_store.add_graph_documents(graph_documents)

        microsoft_id = self._get_int64_hash("Company-Microsoft")
        github_id = self._get_int64_hash("Product-GitHub")

        node_query = f"SELECT label, properties FROM {self.node_table} WHERE id = {microsoft_id}"
        node_result = self.graph_store.query(node_query)
        
        self.assertEqual(len(node_result), 1)
        retrieved_node = node_result[0]
        self.assertEqual(retrieved_node['label'], 'Company')
        retrieved_props = retrieved_node['properties']
        self.assertIn('location', retrieved_props)
        self.assertEqual(retrieved_props['location'], 'Redmond')

        edge_query = f"""SELECT label, properties FROM {self.edge_table} 
                        WHERE id = {microsoft_id} AND dest_id = {github_id}"""
        edge_result = self.graph_store.query(edge_query)

        self.assertEqual(len(edge_result), 1)
        retrieved_edge = edge_result[0]
        self.assertEqual(retrieved_edge['label'], 'ACQUIRED')
        retrieved_edge_props = retrieved_edge['properties']
        self.assertIn('date', retrieved_edge_props)
        self.assertEqual(retrieved_edge_props['date'], 'October 26, 2018')

    def test_cleanup_and_readd_race_condition(self):
        """Tests that the graph can be cleaned up and immediately recreated without race conditions."""
        doc1 = Document(page_content="First document.")
        graph_doc1 = GraphDocument(nodes=[Node(id="A", type="Thing")], relationships=[], source=doc1)
        self.graph_store.add_graph_documents([graph_doc1])

        self.graph_store.cleanup()

        doc2 = Document(page_content="Second document.")
        graph_doc2 = GraphDocument(nodes=[Node(id="B", type="Thing")], relationships=[], source=doc2)
        
        try:
            self.graph_store.add_graph_documents([graph_doc2])
        except Exception as e:
            self.fail(f"add_graph_documents failed unexpectedly after cleanup with error: {e}")

        node_b_id = self._get_int64_hash("Thing-B")
        node_query = f"SELECT label FROM {self.node_table} WHERE id = {node_b_id}"
        node_result = self.graph_store.query(node_query)
        self.assertEqual(len(node_result), 1)
        self.assertEqual(node_result[0]['label'], 'Thing')

if __name__ == "__main__":
    unittest.main()
