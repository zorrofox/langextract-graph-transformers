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
        text_properties: List[str],
    ):
        self._graph = graph
        self._embedding = embedding
        self.text_properties = text_properties
        self.embedding_property = "embedding" # Hardcoded to match schema

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
        node_label: str,
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
            
            properties = {
                **metadatas[i],
                self.text_properties[0]: text,
                self.embedding_property: embeddings[i],
            }
            node = Node(id=node_id, type=node_label, properties=properties)
            nodes.append(node)

        graph_doc = GraphDocument(nodes=nodes, relationships=[])
        self._graph.add_graph_documents([graph_doc])

        return ids

    def similarity_search(
        self, query: str, 
        k: int = 4,
        node_label: Optional[str] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Run similarity search with the query."""
        embedding = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(embedding, k, **kwargs)

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, node_label: Optional[str] = None, **kwargs: Any
    ) -> List[Document]:
        """
        Perform a similarity search by vector using Spanner's native vector functions.
        If node_label is provided, the search is filtered to nodes with that label.
        Otherwise, the search is performed across all nodes.
        """
        base_query = f"""
        SELECT properties, {self.embedding_property}
        FROM {self._graph.node_table}
        """
        
        params = {
            "query_embedding": embedding,
            "limit": k,
        }
        param_types = {
            "query_embedding": spanner.param_types.Array(spanner.param_types.FLOAT64),
            "limit": spanner.param_types.INT64,
        }

        where_clauses = [f"{self.embedding_property} IS NOT NULL"]
        if node_label:
            where_clauses.append("label = @node_label")
            params["node_label"] = node_label
            param_types["node_label"] = spanner.param_types.STRING

        query = f"{base_query} WHERE {' AND '.join(where_clauses)} ORDER BY COSINE_DISTANCE({self.embedding_property}, @query_embedding) LIMIT @limit"

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
                    str(self._get_value_by_path(props, key) or "") for key in self.text_properties
                ).strip()
                
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
        )
        store.add_texts(texts, metadatas, **kwargs)
        return store

    @classmethod
    def from_existing_graph(
        cls: Type[SpannerGraphVectorStore],
        graph: SpannerSchemalessGraph,
        embedding: Embeddings,
        text_properties: List[str],
        node_label: Optional[str] = None,
        include_label_in_embedding: bool = False,
        batch_size: int = 1000,
    ) -> SpannerGraphVectorStore:
        """
        Create a SpannerGraphVectorStore from an existing Spanner graph,
        populating the dedicated embedding column for nodes that are missing it.
        """
        embedding_property = "embedding" # Hardcoded to match schema
        print(f"Starting to populate '{embedding_property}' column for nodes...")
        if node_label:
            print(f"Filtering for node_label: '{node_label}'")
        if include_label_in_embedding:
            print("Including node label in content for embedding.")

        total_updated_count = 0
        last_processed_id = None
        
        while True:
            # ... (query building logic remains the same) ...
            base_query = f"SELECT id, label, properties FROM {graph.node_table}"
            where_clauses = [f"{embedding_property} IS NULL"]
            params = {}
            param_types = {}

            if node_label:
                where_clauses.append("label = @node_label")
                params["node_label"] = node_label
                param_types["node_label"] = spanner.param_types.STRING

            if last_processed_id:
                where_clauses.append("id > @last_processed_id")
                params["last_processed_id"] = last_processed_id
                param_types["last_processed_id"] = spanner.param_types.INT64
            
            params["batch_size"] = batch_size
            param_types["batch_size"] = spanner.param_types.INT64

            query = f"{base_query} WHERE {' AND '.join(where_clauses)} ORDER BY id LIMIT @batch_size"

            with graph._database.snapshot() as snapshot:
                result_stream = snapshot.execute_sql(query, params=params, param_types=param_types)
                nodes_to_process = list(result_stream)

            num_found = len(nodes_to_process)
            if num_found == 0:
                print("No more nodes to update.")
                break

            print(f"Found {num_found} nodes to process in this batch...")

            parsed_nodes = []
            for node_id, label, properties in nodes_to_process:
                if isinstance(properties, str):
                    try:
                        properties = json.loads(properties)
                    except json.JSONDecodeError:
                        properties = {}
                else:
                    properties = dict(properties)
                parsed_nodes.append({"node_id": node_id, "label": label, "properties": properties})

            texts_to_embed = []
            valid_nodes_for_embedding = []
            for node in parsed_nodes:
                text_parts = []
                for p in text_properties:
                    if p == "id":
                        text_parts.append(str(node["node_id"]))
                    elif p == "label":
                        text_parts.append(str(node["label"]))
                    else:
                        text_parts.append(str(cls._get_value_by_path(node["properties"], p) or ""))
                
                content = " ".join(text_parts).strip()

                if include_label_in_embedding:
                    content = f"{node['label']} {content}".strip()
                
                if content:
                    texts_to_embed.append(content)
                    valid_nodes_for_embedding.append(node)

            if not valid_nodes_for_embedding:
                print("No nodes with valid text content in this batch. Skipping.")
                last_processed_id = parsed_nodes[-1]["node_id"]
                continue

            embeddings = embedding.embed_documents(texts_to_embed)

            nodes_to_update = []
            for i, node in enumerate(valid_nodes_for_embedding):
                sanitized_embedding = [x if math.isfinite(x) else 0.0 for x in embeddings[i]]
                nodes_to_update.append((node["node_id"], sanitized_embedding))

            def _update_batch(transaction):
                transaction.update(
                    table=graph.node_table,
                    columns=("id", embedding_property),
                    values=nodes_to_update,
                )
            
            graph._database.run_in_transaction(_update_batch)
            num_updated = len(nodes_to_update)
            total_updated_count += num_updated
            print(f"Successfully updated {num_updated} nodes. Total updated so far: {total_updated_count}")

            last_processed_id = parsed_nodes[-1]["node_id"]

        print(f"Initialization complete. Total nodes updated: {total_updated_count}")
        return cls(
            graph=graph,
            embedding=embedding,
            text_properties=text_properties,
        )
