
import unittest
import os
import uuid
import json
import hashlib
from dotenv import load_dotenv

from langchain_core.documents import Document

# This is a bit of a hack to allow importing from the parent directory
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from langextract_graph_transformers.langextract_graph_transformer import LangExtractGraphTransformer
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph

load_dotenv()

class TestFullPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.instance_id = os.getenv("SPANNER_INSTANCE_ID")
        self.database_id = os.getenv("SPANNER_DATABASE_ID")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")

        if not all([self.instance_id, self.database_id, self.project_id]):
            self.skipTest("Full pipeline integration tests require all environment variables.")
        
        unique_id = str(uuid.uuid4())[:8]
        self.node_table = f"GraphNode_{unique_id}"
        self.edge_table = f"GraphEdge_{unique_id}"
        self.graph_name = f"FinGraph_{unique_id}"

        # 1. Initialize Spanner Graph Storage (creates schema)
        self.graph_store = SpannerSchemalessGraph(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            node_table=self.node_table,
            edge_table=self.edge_table,
            graph_name=self.graph_name,
        )

        # 2. Initialize Graph Transformer
        self.transformer = LangExtractGraphTransformer(
            project_id=self.project_id,
            location=os.getenv("VERTEX_AI_LOCATION", "us-central1"),
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
        # Define source text
        text_content = ("Microsoft, a tech giant headquartered in Redmond, announced its acquisition of Activision Blizzard on January 18, 2022.")
        document = Document(page_content=text_content)

        # 1. Extract graph from text
        graph_documents = self.transformer.process_documents([document])
        self.assertEqual(len(graph_documents), 1)
        self.assertGreater(len(graph_documents[0].nodes), 1)
        self.assertGreater(len(graph_documents[0].relationships), 0)

        # 2. Store the extracted graph in Spanner
        self.graph_store.add_graph_documents(graph_documents)

        # 3. Query the data back from Spanner to verify
        import time
        time.sleep(5) # Allow some time for data to be indexed

        # Query for the 'Microsoft' node
        microsoft_id = self._get_int64_hash("Company-Microsoft")
        node_query = f"SELECT label, properties FROM {self.node_table} WHERE id = {microsoft_id}"
        node_result = self.graph_store.query(node_query)
        
        self.assertEqual(len(node_result), 1)
        retrieved_node = node_result[0]
        self.assertEqual(retrieved_node['label'], 'Company')
        retrieved_props = retrieved_node['properties']
        self.assertIn('prop_location', retrieved_props) # Check if the property was extracted
        self.assertEqual(retrieved_props['prop_location'], 'Redmond')

        # Query for the 'ACQUIRED' relationship
        activision_id = self._get_int64_hash("Company-Activision Blizzard")
        edge_query = f"""SELECT label, properties FROM {self.edge_table} 
                        WHERE id = {microsoft_id} AND dest_id = {activision_id}"""
        edge_result = self.graph_store.query(edge_query)

        self.assertEqual(len(edge_result), 1)
        retrieved_edge = edge_result[0]
        self.assertEqual(retrieved_edge['label'], 'ACQUIRED')
        retrieved_edge_props = retrieved_edge['properties']
        self.assertIn('prop_date', retrieved_edge_props)
        self.assertEqual(retrieved_edge_props['prop_date'], 'January 18, 2022')

if __name__ == "__main__":
    unittest.main()
