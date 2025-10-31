import unittest
import os
import uuid
import json
import hashlib
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph

load_dotenv()

class TestSchemalessGraphStoreIntegration(unittest.TestCase):

    def setUp(self):
        self.instance_id = os.getenv("SPANNER_INSTANCE_ID")
        self.database_id = os.getenv("SPANNER_DATABASE_ID")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")

        if not all([self.instance_id, self.database_id, self.project_id]):
            self.skipTest("Spanner integration tests require SPANNER_INSTANCE_ID, SPANNER_DATABASE_ID, and VERTEX_AI_PROJECT_ID.")
        
        # Use unique table names for each test run to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        self.node_table = f"GraphNode_{unique_id}"
        self.edge_table = f"GraphEdge_{unique_id}"
        self.graph_name = f"FinGraph_{unique_id}"

        self.graph = SpannerSchemalessGraph(
            project_id=self.project_id,
            instance_id=self.instance_id,
            database_id=self.database_id,
            node_table=self.node_table,
            edge_table=self.edge_table,
            graph_name=self.graph_name,
        )

    def tearDown(self):
        self.graph.cleanup()

    def _get_int64_hash(self, input_string: str) -> int:
        sha256_hash = hashlib.sha256(input_string.encode('utf-8')).digest()
        return int.from_bytes(sha256_hash[:8], byteorder='big', signed=True)

    def test_e2e_lifecycle(self):
        """Tests the full lifecycle: schema creation, data insertion, and querying."""
        # 1. Schema creation is implicitly tested by setUp().

        # 2. Prepare and insert data
        source_doc = Document(page_content="Source text for integration test.")
        node1 = Node(id="TestCorp", type="Company", properties={"employees": 1000})
        node2 = Node(id="John Doe", type="Person", properties={"role": "Engineer"})
        relationship = Relationship(source=node1, target=node2, type="EMPLOYS", properties={"duration_years": 5})
        graph_doc = GraphDocument(source=source_doc, nodes=[node1, node2], relationships=[relationship])

        self.graph.add_graph_documents([graph_doc])

        # 3. Query the data back and verify
        import time
        time.sleep(5) 

        # Query for the node
        node_id = self._get_int64_hash("Company-TestCorp")
        node_query = f"SELECT label, properties FROM {self.node_table} WHERE id = {node_id}"
        node_result = self.graph.query(node_query)
        
        self.assertEqual(len(node_result), 1)
        retrieved_node = node_result[0]
        self.assertEqual(retrieved_node['label'], 'Company')
        retrieved_node_props = retrieved_node['properties']
        self.assertEqual(retrieved_node_props['employees'], 1000)

        # Query for the edge
        source_hash_id = self._get_int64_hash("Company-TestCorp")
        edge_query = f"SELECT label, properties FROM {self.edge_table} WHERE id = {source_hash_id}"
        edge_result = self.graph.query(edge_query)

        self.assertEqual(len(edge_result), 1)
        retrieved_edge = edge_result[0]
        self.assertEqual(retrieved_edge['label'], 'EMPLOYS')
        retrieved_edge_props = retrieved_edge['properties']
        self.assertEqual(retrieved_edge_props['duration_years'], 5)

if __name__ == "__main__":
    unittest.main()