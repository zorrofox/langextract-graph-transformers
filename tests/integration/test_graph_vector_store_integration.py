
import unittest
from unittest.mock import patch
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

    @patch("langchain_community.embeddings.FakeEmbeddings.embed_documents")
    def test_e2e_vector_store_lifecycle(self, mock_embed_documents):
        """
        Tests the full lifecycle, including the include_label_in_embedding feature.
        """
        # 1. Setup mock for embedding
        embedding_service = FakeEmbeddings(size=768)
        # Since the content is now dynamic, we just mock the return value
        mock_embed_documents.return_value = [embedding_service.embed_query("foo")] * 3

        # 2. Add nodes without embeddings
        docs_to_add = [
            "The cat sat on the mat.",
            "The dog chased the ball.",
            "It was a sunny day."
        ]
        nodes = [Node(id=f"doc_{i}", type="document", properties={"text": text}) for i, text in enumerate(docs_to_add)]
        source_doc = Document(page_content="")
        graph_doc = GraphDocument(nodes=nodes, relationships=[], source=source_doc)
        self.graph.add_graph_documents([graph_doc])

        count_query = f"SELECT COUNT(*) as count FROM {self.node_table} WHERE label = 'document'"
        result = self.graph.query(count_query)
        self.assertEqual(result[0]['count'], 3, "Initial graph data not visible.")

        # 3. Use from_existing_graph to populate embeddings, including the label
        vector_store = SpannerGraphVectorStore.from_existing_graph(
            graph=self.graph,
            embedding=embedding_service,
            text_properties=['text'],
            include_label_in_embedding=True,
        )

        # 4. Assert embed_documents was called correctly
        mock_embed_documents.assert_called_once()
        call_args, _ = mock_embed_documents.call_args
        self.assertIn("document The cat sat on the mat.", call_args[0])
        self.assertIn("document The dog chased the ball.", call_args[0])
        self.assertIn("document It was a sunny day.", call_args[0])

        # 5. Verify embeddings were populated in the database
        populated_nodes_query = f"SELECT embedding FROM {self.node_table}"
        populated_nodes = self.graph.query(populated_nodes_query)
        self.assertEqual(len(populated_nodes), 3)
        for node in populated_nodes:
            self.assertIsNotNone(node["embedding"])
            self.assertEqual(len(node["embedding"]), 768)

        # 6. Perform a similarity search
        vector_store.node_label = "document"
        query_text = "A feline was resting."
        results = vector_store.similarity_search(query=query_text, k=1)

        self.assertEqual(len(results), 1)
        self.assertIn(results[0].page_content, docs_to_add)


        @patch("langchain_community.embeddings.FakeEmbeddings.embed_query")
        @patch("langchain_community.embeddings.FakeEmbeddings.embed_documents")
        def test_similarity_search_across_all_nodes(self, mock_embed_documents, mock_embed_query):
            """Tests similarity search across nodes with different labels."""
            # 1. Setup embeddings
            embedding_service = FakeEmbeddings(size=2) # Small embedding for test
    
            # Define predictable embeddings
            query_emb = [1.0, 0.0] # Query for cat_doc and cat_animal
            cat_doc_emb = [1.0, 0.0] # Perfect match
            dog_doc_emb = [0.0, 1.0] # Orthogonal
            cat_animal_emb = [0.9, 0.1] # Close to cat_doc
            dog_breed_emb = [0.1, 0.9] # Close to dog_doc
    
            # 2. Add nodes with different labels
            nodes_to_add = [
                Node(id="cat_doc", type="document", properties={"text": "A document about a cat."}),
                Node(id="dog_doc", type="document", properties={"text": "A document about a dog."}),
                Node(id="cat_animal", type="animal", properties={"text": "A furry creature that purrs."}),
                Node(id="dog_breed", type="animal", properties={"text": "A loyal companion."}),
            ]
            # Sort nodes by ID to match the ORDER BY id in from_existing_graph
            nodes_to_add.sort(key=lambda node: node.id)
    
            source_doc = Document(page_content="Integration test data source.")
            graph_doc = GraphDocument(nodes=nodes_to_add, relationships=[], source=source_doc)
            self.graph.add_graph_documents([graph_doc])
    
            # Get the nodes from the database in the same order as _populate_embeddings
            with self.graph._database.snapshot() as snapshot:
                results = snapshot.execute_sql(f"SELECT id, properties FROM {self.graph.node_table} ORDER BY id")
                db_nodes = list(results)
    
            # Create the mock embeddings based on the order of the nodes in the database
            mock_embeddings = []
            for node_id, props in db_nodes:
                if props['text'] == 'A document about a cat.':
                    mock_embeddings.append(cat_doc_emb)
                elif props['text'] == 'A document about a dog.':
                    mock_embeddings.append(dog_doc_emb)
                elif props['text'] == 'A furry creature that purrs.':
                    mock_embeddings.append(cat_animal_emb)
                elif props['text'] == 'A loyal companion.':
                    mock_embeddings.append(dog_breed_emb)
            
            mock_embed_query.return_value = query_emb
            mock_embed_documents.return_value = mock_embeddings
    
            # 3. Use from_existing_graph to populate embeddings
            # Embed based on node IDs for predictable mocking
            vector_store = SpannerGraphVectorStore.from_existing_graph(
                graph=self.graph,
                embedding=embedding_service,
                text_properties=['id', 'text'], # Embed based on ID and text for predictable mocking
            )
    
            # 4. Perform a similarity search for a query related to one of the nodes
            query_text = "A purring feline." # This text is not used for embedding, only to trigger mock_embed_query
            results = vector_store.similarity_search(
                embedding=query_emb, # Pass the pre-defined query embedding
                k=2 # Expect 2 results
            )
    
            # 5. Assertions
            self.assertEqual(len(results), 2)
            
            # The top result should be cat_doc, second should be cat_animal
            result_ids = {doc.metadata['id'] for doc in results}
            
            # Get the hashed IDs for assertion using the graph's internal hashing logic
            # The nodes_to_add list is already sorted by ID
            hashed_cat_doc_id = self.graph._get_int64_hash(f"{nodes_to_add[0].type.lower()}-{nodes_to_add[0].id.lower()}")
            hashed_cat_animal_id = self.graph._get_int64_hash(f"{nodes_to_add[2].type.lower()}-{nodes_to_add[2].id.lower()}")
    
            self.assertTrue(hashed_cat_doc_id in result_ids)
            self.assertTrue(hashed_cat_animal_id in result_ids)
            # Check that the content is correctly constructed
            for doc in results:
                if doc.metadata['id'] == 'cat_doc':
                    self.assertIn("A document about a cat.", doc.page_content)
                if doc.metadata['id'] == 'cat_animal':
                    self.assertIn("A furry creature that purrs.", doc.page_content)

if __name__ == "__main__":
    unittest.main()
