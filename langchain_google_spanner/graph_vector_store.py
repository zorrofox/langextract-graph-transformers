from __future__ import annotations

import json
import uuid
import math
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
        embedding_property: str = "embedding",
    ):
        self._graph = graph
        self._embedding = embedding
        self.node_label = node_label
        self.text_properties = text_properties
        self.embedding_property = embedding_property

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
            metadata = metadatas[i]
            properties = {
                **metadata,
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
        """Perform a similarity search by vector using Spanner's native vector functions."""

        query = f"""
        SELECT properties
        FROM {self._graph.node_table}
        WHERE label = @node_label
        ORDER BY COSINE_DISTANCE(FLOAT64_ARRAY(JSON_QUERY(properties, '$.{self.embedding_property}')), @query_embedding)
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
                props = row[0]
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except json.JSONDecodeError:
                        continue

                text = " ".join(
                    str(props.get(key, "")) for key in self.text_properties
                ).strip()
                metadata = {
                    k: v for k, v in props.items() if k != self.embedding_property
                }
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
        populating embeddings for nodes that are missing them.
        """
        print(f"Starting to populate embeddings for nodes with label '{node_label}'...")

        def _fetch_and_update_in_batches(transaction) -> int:
            query = f"""
            SELECT id, properties
            FROM {graph.node_table}
            WHERE label = @node_label
            AND JSON_VALUE(properties, '$.{embedding_property}') IS NULL
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
                    str(node["properties"].get(p, "")) for p in text_properties
                ).strip()
                for node in nodes_to_process
            ]
            embeddings = embedding.embed_documents(texts_to_embed)

            nodes_to_update = []
            for i, node in enumerate(nodes_to_process):
                props = node.get("properties", {})
                props[embedding_property] = [float(x) for x in embeddings[i]]
                nodes_to_update.append((node["node_id"], node_label, props))

            print(nodes_to_update)

            transaction.update(
                table=graph.node_table,
                columns=("id", "label", "properties"),
                values=[(n[0], n[1], JsonObject(n[2])) for n in nodes_to_update],
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
