from __future__ import annotations

import json
import uuid
import math
from typing import Any, Iterable, List, Optional, Type

from google.cloud import spanner
from google.cloud.spanner_v1 import JsonObject, param_types
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from .schemaless_graph_store import SpannerSchemalessGraph
from langchain_community.graphs.graph_document import Node, GraphDocument


class SpannerGraphVectorStore(VectorStore):
    """
    A VectorStore implementation that uses a SpannerSchemalessGraph as a backend.
    """

    def __init__(
        self,
        graph: SpannerSchemalessGraph,
        embedding: Embeddings,
        node_label: str,
        text_properties: List[str],
        embedding_property: str = "embedding", # This now refers to the dedicated column
    ):
        self._graph = graph
        self._embedding = embedding
        self.node_label = node_label
        self.text_properties = text_properties
        # The embedding_property name is kept for consistency, but it maps to a dedicated column
        self.embedding_property = embedding_property

    @staticmethod
    def _get_value_by_path(data: dict, path: str) -> Any:
        """Retrieves a value from a nested dictionary using a dot-separated path."""
        keys = path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
            if value is None:
                return None
        return value

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Adds texts as nodes to the graph and embeds them.
        The embedding is stored in the dedicated 'embedding' column.
        """
        texts = list(texts)
        if not texts:
            return []
        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = self._embedding.embed_documents(texts)

        nodes = []
        ids = []
        for i, text in enumerate(texts):
            node_id = str(uuid.uuid4())
            ids.append(node_id)
            
            # Embeddings are now a top-level property of the Node
            # to be handled by our modified SpannerSchemalessGraph
            properties = {
                **metadatas[i],
                self.text_properties[0]: text,
                self.embedding_property: embeddings[i],
            }
            node = Node(id=node_id, type=self.node_label, properties=properties)
            nodes.append(node)

        graph_doc = GraphDocument(nodes=nodes, relationships=[])
        self._graph.add_graph_documents([graph_doc])

        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search with the query."""
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Perform a similarity search by vector using Spanner's native vector functions
        on the dedicated `embedding` column.
        """
        query = f"""
        SELECT properties, {self.embedding_property}
        FROM {self._graph.node_table}
        WHERE label = @node_label AND {self.embedding_property} IS NOT NULL
        ORDER BY COSINE_DISTANCE({self.embedding_property}, @query_embedding)
        LIMIT @limit
        """

        params = {
            "node_label": self.node_label,
            "query_embedding": embedding,
            "limit": k,
        }
        param_types = {
            "node_label": spanner.param_types.STRING,
            "query_embedding": spanner.param_types.Array(spanner.param_types.FLOAT64),
            "limit": spanner.param_types.INT64,
        }

        docs = []
        with self._graph._database.snapshot() as snapshot:
            result_stream = snapshot.execute_sql(
                query, params=params, param_types=param_types
            )
            rows = list(result_stream)
            if not rows:
                return []

            for row in rows:
                props, emb = row
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except json.JSONDecodeError:
                        continue
                
                text = " ".join(
                    str(SpannerGraphVectorStore._get_value_by_path(props, key) or "") for key in self.text_properties
                ).strip()
                
                # Metadata no longer needs to filter the embedding property,
                # as it's in a separate column.
                metadata = props
                docs.append(Document(page_content=text, metadata=metadata))

        return docs

    @classmethod
    def from_texts(
        cls: Type[SpannerGraphVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        graph: SpannerSchemalessGraph,
        node_label: str,
        text_properties: List[str],
        embedding_property: str = "embedding",
        **kwargs: Any,
    ) -> SpannerGraphVectorStore:
        """
        Create a SpannerGraphVectorStore from a list of texts.
        """
        store = cls(
            graph=graph,
            embedding=embedding,
            node_label=node_label,
            text_properties=text_properties,
            embedding_property=embedding_property,
        )
        store.add_texts(texts, metadatas, **kwargs)
        return store

    @classmethod
    def from_existing_graph(
        cls: Type[SpannerGraphVectorStore],
        graph: SpannerSchemalessGraph,
        embedding: Embeddings,
        node_label: str,
        text_properties: List[str],
        embedding_property: str = "embedding",
        batch_size: int = 100,
    ) -> SpannerGraphVectorStore:
        """
        Create a SpannerGraphVectorStore from an existing Spanner graph,
        populating the dedicated `embedding` column for nodes that are missing it.
        """
        print(f"Starting to populate embeddings for nodes with label '{node_label}'...")

        def _fetch_and_update_in_batches(transaction) -> int:
            query = f"""
            SELECT id, properties
            FROM {graph.node_table}
            WHERE label = @node_label AND {embedding_property} IS NULL
            """
            params = {"node_label": node_label}
            param_types = {"node_label": spanner.param_types.STRING}

            result_stream = transaction.execute_sql(
                query, params=params, param_types=param_types
            )

            nodes_to_process = []
            for row in result_stream:
                node_id, properties = row
                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except json.JSONDecodeError:
                        properties = {}
                else:
                    properties = dict(properties)
                nodes_to_process.append({"node_id": node_id, "properties": properties})

            if not nodes_to_process:
                print("No nodes found to update.")
                return 0

            print(f"Found {len(nodes_to_process)} nodes to update.")

            texts_to_embed = [
                " ".join(
                    str(SpannerGraphVectorStore._get_value_by_path(node["properties"], p) or "") for p in text_properties
                ).strip()
                for node in nodes_to_process
            ]
            embeddings = embedding.embed_documents(texts_to_embed)

            nodes_to_update = []
            for i, node in enumerate(nodes_to_process):
                sanitized_embedding = [x if math.isfinite(x) else 0.0 for x in embeddings[i]]
                nodes_to_update.append((node["node_id"], sanitized_embedding))

            transaction.update(
                table=graph.node_table,
                columns=("id", embedding_property),
                values=nodes_to_update,
            )
            return len(nodes_to_update)

        graph._database.run_in_transaction(_fetch_and_update_in_batches)

        print("Initialization complete. Returning VectorStore instance.")
        return cls(
            graph=graph,
            embedding=embedding,
            node_label=node_label,
            text_properties=text_properties,
            embedding_property=embedding_property,
        )