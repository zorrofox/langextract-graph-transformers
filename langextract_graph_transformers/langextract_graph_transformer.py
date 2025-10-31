from typing import List, Optional, Any
import langextract as lx
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_core.documents import Document
import json
import re
import logging

# Suppress the specific ABSL warning about prompt alignment
logging.getLogger('absl').setLevel(logging.ERROR)

class LangExtractGraphTransformer:
    """
    A graph transformer that uses langextract to extract graph structures from documents.
    It supports both schema-driven extraction and arbitrary (schema-less) extraction.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        node_properties: Optional[List[str]] = None,
        relationship_properties: Optional[List[str]] = None,
        model_id: str = "gemini-2.5-pro",
    ):
        self.project_id = project_id
        self.location = location
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.model_id = model_id
        self.model_config = {
            "vertexai": True,
            "project": project_id,
            "location": location,
        }

    def _get_arbitrary_example(self) -> lx.data.ExampleData:
        """Provides a high-quality example for arbitrary graph extraction, including properties."""
        example_text = ("FirstEnergy (NYSE:FE), a major utilities provider, posted its earnings results on Tuesday. "
                        "The company reported $0.53 earnings per share for the quarter.")
        
        example_json_output = json.dumps({
            "extractions": [
                {"id": "FirstEnergy", "type": "Company", "properties": {"sector": "Utilities"}},
                {"id": "FE", "type": "StockSymbol", "properties": {}},
                {"id": "$0.53", "type": "EarningsPerShare", "properties": {}},
                {"source": "FirstEnergy", "target": "FE", "type": "HAS_STOCK_SYMBOL", "properties": {"confidence": 1.0}},
                {"source": "FirstEnergy", "target": "$0.53", "type": "REPORTED_EARNINGS", "properties": {"quarter": "Q1"}}
            ]
        }, indent=2)

        return lx.data.ExampleData(
            text=example_text,
            extractions=[
                lx.data.Extraction(
                    extraction_class="GraphJSON",
                    extraction_text=example_json_output
                )
            ]
        )

    def process_documents(self, documents: List[Document]) -> List[GraphDocument]:
        """
        Processes a list of documents to extract graph structures.
        """
        example = self._get_arbitrary_example()
        
        results = []
        for document in documents:
            graph_document = self._process_single_document(document, example)
            results.append(graph_document)
        return results

    def _process_single_document(self, document: Document, example: lx.data.ExampleData) -> GraphDocument:
        """
        Processes a single document to extract a graph structure using the 'meta-extraction' method.
        """
        prompt = """
        You are an expert at building knowledge graphs. 
        From the provided text, extract all meaningful entities as nodes and the relationships between them.
        Your output MUST be a single valid JSON object with a single key, "extractions".
        The value of "extractions" must be an array of JSON objects, where each object is either a node or a relationship.
        Each node and relationship object should contain an optional "properties" object for its attributes.
        Do not include any other text or markdown formatting in your response.
        """

        if self.node_properties:
            prompt += f"\nFor nodes, you should extract the following properties when available: {self.node_properties}"
        if self.relationship_properties:
            prompt += f"\nFor relationships, you should extract the following properties when available: {self.relationship_properties}"

        result = lx.extract(
            text_or_documents=document.page_content,
            prompt_description=prompt,
            examples=[example], # The 'meta' example guides the LLM
            model_id=self.model_id,
            language_model_params=self.model_config,
        )

        nodes = []
        relationships = []
        if result.extractions and result.extractions[0].extraction_text:
            try:
                graph_json_string = result.extractions[0].extraction_text
                graph_data_obj = json.loads(graph_json_string)
                
                # Handle both dict and list outputs from the LLM
                if isinstance(graph_data_obj, dict):
                    graph_data = graph_data_obj.get("extractions", [])
                elif isinstance(graph_data_obj, list):
                    graph_data = graph_data_obj
                else:
                    graph_data = []
                
                node_map = {}
                for item in graph_data:
                    if "source" not in item and "id" in item and "type" in item:
                        node_id = item["id"]
                        if node_id not in node_map:
                            node = Node(
                                id=node_id,
                                type=item["type"],
                                properties={f"prop_{str(k).lower()}": str(v) for k, v in item.get("properties", {}).items()}
                            )
                            nodes.append(node)
                            node_map[node_id] = node
                
                for item in graph_data:
                    if "source" in item and "target" in item and "type" in item:
                        source_node = node_map.get(item["source"])
                        target_node = node_map.get(item["target"])
                        if source_node and target_node:
                            relationships.append(
                                Relationship(
                                    source=source_node,
                                    target=target_node,
                                    type=item["type"],
                                    properties={f"prop_{str(k).lower()}": str(v) for k, v in item.get("properties", {}).items()}
                                )
                            )
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Failed to parse graph from document: {e}")
                pass

        return GraphDocument(nodes=nodes, relationships=relationships, source=document)
