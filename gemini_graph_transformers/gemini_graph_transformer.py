from typing import List, Optional, Any, Dict
import json
import logging

from google import genai
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

class GeminiGraphTransformer:
    """
    A graph transformer that uses the native JSON mode of Google's Gemini models 
    via the Vertex AI backend to reliably extract graph structures from documents.
    """

    def __init__(
        self,
        project_id: str,
        location: str,
        node_properties: Optional[List[str]] = None,
        relationship_properties: Optional[List[str]] = None,
        model_name: str = "gemini-2.5-pro",
    ):
        self.node_properties = node_properties
        self.relationship_properties = relationship_properties
        self.model_name = model_name
        
        # Correctly initialize the client for Vertex AI usage
        self.client = genai.Client(vertexai=True, project=project_id, location=location)

    def process_documents(self, documents: List[Document]) -> List[GraphDocument]:
        """Processes a list of documents to extract graph structures."""
        results = []
        for document in documents:
            graph_document = self._process_single_document(document)
            results.append(graph_document)
        return results

    def _process_single_document(self, document: Document) -> GraphDocument:
        """Processes a single document using Gemini's native JSON mode via Vertex AI."""
        prompt = self._build_prompt()
        
        full_prompt = f"{prompt}\n\nText to process:\n---\n{document.page_content}\n---"

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            graph_data = json.loads(response.text)
        except (json.JSONDecodeError, Exception) as e:
            logging.warning(f"Failed to parse JSON from model response: {e}. Response text: {getattr(response, 'text', '')}")
            return GraphDocument(nodes=[], relationships=[], source=document)

        nodes = []
        relationships = []
        node_map = {}

        for item in graph_data:
            if "source" not in item and "id" in item and "type" in item:
                node_id = item["id"]
                if node_id not in node_map:
                    node = Node(
                        id=node_id,
                        type=item["type"],
                        properties={k: v for k, v in item.get("properties", {}).items()}
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
                            properties={k: v for k, v in item.get("properties", {}).items()}
                        )
                    )

        graph_document = GraphDocument(nodes=nodes, relationships=relationships, source=document)
        return graph_document

    def _build_prompt(self) -> str:
        """Builds the prompt for the LLM, including property instructions."""
        prompt = """You are an expert at building knowledge graphs. 
From the provided text, extract all meaningful entities as nodes and the relationships between them.
Your output MUST be a single, valid JSON array of nodes and relationships.
Each node must have an "id" and a "type".
Each relationship must have a "source" (the ID of the source node), a "target" (the ID of the target node), and a "type".
Nodes and relationships can have an optional "properties" object for additional attributes.
Do not include any other text, comments, or markdown formatting in your response.

Here is an example:

Text to process:
---
In a major tech deal, Google, a software company, officially acquired YouTube for $1.65 billion on October 9, 2006.
---

JSON:
[    
    {{
        "id": "Google",
        "type": "Company",
        "properties": {{
            "sector": "Software",
            "location": "Mountain View"
        }}
    }},
    {{
        "id": "YouTube",
        "type": "Product"
    }},
    {{
        "source": "Google",
        "target": "YouTube",
        "type": "ACQUIRED",
        "properties": {{
            "date": "October 9, 2006",
            "value_usd": 1650000000
        }}
    }}
]
"""

        if self.node_properties:
            prompt += f"\nFor nodes, you should extract the following properties when available: {self.node_properties}"
        if self.relationship_properties:
            prompt += f"\nFor relationships, you should extract the following properties when available: {self.relationship_properties}"
        
        return prompt
