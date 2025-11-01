from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import hashlib

from google.cloud.spanner_v1 import Client, JsonObject
from google.cloud.spanner_v1.database import Database
from google.cloud.spanner_v1.pool import AbstractSessionPool
from google.cloud.spanner_v1.transaction import Transaction

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

class SpannerSchemalessGraph:
    """A graph store for Google Cloud Spanner using a schema-less, high-performance data model."""

    def __init__(
        self,
        instance_id: str,
        database_id: str,
        node_table: str = "GraphNode",
        edge_table: str = "GraphEdge",
        graph_name: str = "FinGraph",
        project_id: Optional[str] = None,
        client: Optional[Client] = None,
        pool: Optional[AbstractSessionPool] = None,
    ):
        self._client = client or Client(project=project_id)
        self._instance_id = instance_id
        self._database_id = database_id
        self.node_table = node_table
        self.edge_table = edge_table
        self.graph_name = graph_name

        self._database = self._client.instance(self._instance_id).database(
            self._database_id,
            pool=pool,
        )
        
        self._create_or_verify_schema()

    def _get_int64_hash(self, input_string: str) -> int:
        """Creates a deterministic 64-bit integer hash for a given string."""
        sha256_hash = hashlib.sha256(input_string.encode('utf-8')).digest()
        return int.from_bytes(sha256_hash[:8], byteorder='big', signed=True)

    def _create_or_verify_schema(self) -> None:
        """Checks for and creates the tables and property graph based on the official template."""
        try:
            table_ddls = []
            with self._database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    f"""SELECT t.table_name FROM information_schema.tables AS t
                        WHERE t.table_catalog = '' AND t.table_schema = '' 
                        AND t.table_name IN ('{self.node_table}', '{self.edge_table}')"""
                )
                existing_tables = {row[0] for row in results}

            if self.node_table not in existing_tables:
                table_ddls.append(f"""CREATE TABLE {self.node_table} (
                      id INT64 NOT NULL,
                      label STRING(MAX),
                      properties JSON,
                    ) PRIMARY KEY (id)""")

            if self.edge_table not in existing_tables:
                 table_ddls.append(f"""CREATE TABLE {self.edge_table} (
                      id INT64 NOT NULL,
                      dest_id INT64 NOT NULL,
                      edge_id INT64 NOT NULL,
                      label STRING(MAX),
                      properties JSON,
                    ) PRIMARY KEY (id, dest_id, edge_id),
                      INTERLEAVE IN PARENT {self.node_table} ON DELETE CASCADE""")
            
            if table_ddls:
                op_tables = self._database.update_ddl(ddl_statements=table_ddls)
                op_tables.result() # Wait for table creation to complete

            with self._database.snapshot() as snapshot:
                graph_results = snapshot.execute_sql(
                    f"SELECT 1 FROM information_schema.property_graphs WHERE PROPERTY_GRAPH_NAME = '{self.graph_name}'"
                )
                if not list(graph_results):
                    graph_ddl = f"""CREATE PROPERTY GRAPH {self.graph_name}
                          NODE TABLES (
                            {self.node_table}
                              DYNAMIC LABEL (label)
                              DYNAMIC PROPERTIES (properties)
                          )
                          EDGE TABLES (
                            {self.edge_table}
                              SOURCE KEY (id) REFERENCES {self.node_table}(id)
                              DESTINATION KEY (dest_id) REFERENCES {self.node_table}(id)
                              DYNAMIC LABEL (label)
                              DYNAMIC PROPERTIES (properties)
                          )"""
                    op_graph = self._database.update_ddl(ddl_statements=[graph_ddl])
                    op_graph.result() # Wait for graph creation to complete

        except Exception as e:
            print(f"An error occurred during schema verification/creation: {e}")

    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """Adds graph documents to the Spanner database."""
        self._create_or_verify_schema()

        def insert_data(transaction: Transaction) -> None:
            node_columns = ["id", "label", "properties"]
            edge_columns = ["id", "dest_id", "edge_id", "label", "properties"]

            node_mutations = []
            edge_mutations = []

            for doc in graph_documents:
                for node in doc.nodes:
                    node_id = self._get_int64_hash(f"{node.type}-{node.id}")
                    properties = node.properties or {}
                    if baseEntityLabel:
                        properties["baseEntityLabel"] = True
                    if include_source:
                        properties["source"] = {
                            "page_content": doc.source.page_content,
                            "metadata": doc.source.metadata,
                        }
                    node_mutations.append((node_id, node.type, JsonObject(properties)))

                for rel in doc.relationships:
                    source_hash_id = self._get_int64_hash(
                        f"{rel.source.type}-{rel.source.id}"
                    )
                    target_hash_id = self._get_int64_hash(
                        f"{rel.target.type}-{rel.target.id}"
                    )
                    edge_hash_id = self._get_int64_hash(
                        f"{source_hash_id}-{rel.type}-{target_hash_id}"
                    )

                    properties = rel.properties or {}
                    edge_mutations.append(
                        (
                            source_hash_id,
                            target_hash_id,
                            edge_hash_id,
                            rel.type,
                            JsonObject(properties),
                        )
                    )

            if node_mutations:
                transaction.insert_or_update(
                    table=self.node_table,
                    columns=node_columns,
                    values=node_mutations,
                )
            if edge_mutations:
                transaction.insert_or_update(
                    table=self.edge_table,
                    columns=edge_columns,
                    values=edge_mutations,
                )

        self._database.run_in_transaction(insert_data)

    def query(self, query: str) -> List[Dict[str, Any]]:
        """Executes a GoogleSQL query against the database."""
        with self._database.snapshot() as snapshot:
            result_stream = snapshot.execute_sql(query)
            rows = list(result_stream)
            if not rows:
                return []
            
            field_names = [field.name for field in result_stream.fields]
            results = [dict(zip(field_names, row)) for row in rows]
        return results

    def refresh_schema(self) -> None:
        self._create_or_verify_schema()

    def cleanup(self) -> None:
        """Deletes the graph, node, and edge tables safely."""
        ddl_statements = [
            f"DROP PROPERTY GRAPH IF EXISTS {self.graph_name}",
            f"DROP TABLE IF EXISTS {self.edge_table}",
            f"DROP TABLE IF EXISTS {self.node_table}",
        ]
        try:
            op = self._database.update_ddl(ddl_statements=ddl_statements)
            op.result(timeout=200)
        except Exception as e:
            # This is expected if the resources don't exist, so we don't log it.
            pass