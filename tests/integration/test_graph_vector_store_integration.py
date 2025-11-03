
import unittest
import os
import uuid
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node
from langchain_community.embeddings import FakeEmbeddings

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from langchain_google_spanner.schemaless_graph_store import SpannerSchemalessGraph
from langchain_google_spanner.graph_vector_store import SpannerGraphVectorStore

load_dotenv()

class TestGraphVectorStoreIntegration(unittest.TestCase):

    def setUp(self):
        import os

        self.instance_id = os.getenv("SPANNER_INSTANCE_ID")
        self.database_id = os.getenv("SPANNER_DATABASE_ID")
        self.project_id = os.getenv("VERTEX_AI_PROJECT_ID")

        if not all([self.instance_id, self.database_id, self.project_id]):
            self.skipTest("Spanner integration tests require SPANNER_INSTANCE_ID, SPANNER_DATABASE_ID, and VERTEX_AI_PROJECT_ID.")
        
        unique_id = str(uuid.uuid4())[:8]
        self.node_table = f"VectorNode_{unique_id}"
        self.edge_table = f"VectorEdge_{unique_id}"
        self.graph_name = f"VectorGraph_{unique_id}"

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

    def test_e2e_vector_store_lifecycle(self):
        """
        Tests the full lifecycle: 
        1. Add nodes without embeddings.
        2. Use from_existing_graph to populate embeddings.
        3. Perform a similarity search.
        """
        # 1. Add nodes without embeddings
        docs_to_add = [
            "The cat sat on the mat.",
            "The dog chased the ball.",
            "It was a sunny day."
        ]
        nodes = [Node(id=f"doc_{i}", type="document", properties={"text": text}) for i, text in enumerate(docs_to_add)]
        source_doc = Document(page_content="")
        graph_doc = GraphDocument(nodes=nodes, relationships=[], source=source_doc)
        self.graph.add_graph_documents([graph_doc])



        # Per user instruction, removed polling logic.
        # The test now assumes immediate data visibility.
        count_query = f"SELECT COUNT(*) as count FROM {self.node_table} WHERE label = 'document'"
        result = self.graph.query(count_query)
        self.assertEqual(result[0]['count'], 3, "Initial graph data not visible.")
        # 2. Use from_existing_graph to populate embeddings
        embedding_service = FakeEmbeddings(size=768)
        
        vector_store = SpannerGraphVectorStore.from_existing_graph(
            graph=self.graph,
            embedding=embedding_service,
            node_label="document",
            text_properties=["text"],
            embedding_property="embedding"
        )

        self.assertIsInstance(vector_store, SpannerGraphVectorStore)

        # 3. Verify embeddings were populated
        
        populated_nodes_query = f"SELECT properties FROM {self.node_table}"
        populated_nodes = self.graph.query(populated_nodes_query)
        
        self.assertEqual(len(populated_nodes), 3)
        for node in populated_nodes:
            props = node['properties']
            self.assertIn("embedding", props)
            self.assertEqual(len(props['embedding']), 768)

        # 4. Perform a similarity search
        query_text = "A feline was resting."
        results = vector_store.similarity_search(query=query_text, k=1)

        self.assertEqual(len(results), 1)
        # FakeEmbeddings creates embeddings from a hash of the text, so similar
        # texts won't necessarily have similar vectors. This test mainly ensures
        # the search runs without error and returns the correct number of documents.
        self.assertIn(results[0].page_content, docs_to_add)


if __name__ == "__main__":
    unittest.main()
