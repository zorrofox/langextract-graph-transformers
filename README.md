# LangExtract Graph Transformer

This project provides a `LangExtractGraphTransformer`, a Python class designed to extract graph structures (nodes and relationships) from text documents using Google's `langextract` library and powerful language models like Gemini.

Inspired by the functionality of LangChain's `LLMGraphTransformer`, this implementation is specifically tailored to leverage the example-driven nature of the `langextract` library to perform robust, arbitrary (schema-less) graph extraction.

## Features

- **Arbitrary Graph Extraction**: Dynamically extracts nodes and relationships from text without a predefined schema.
- **Property Extraction**: Capable of extracting properties for both nodes and relationships.
- **Robust Parsing**: Handles variations in LLM output to reliably parse graph data.
- **Spanner Compatible**: Normalizes property keys and values to ensure compatibility with strongly-typed databases like Google Cloud Spanner.

## How It Works: The "Meta-Extraction" Strategy

After extensive iteration, this transformer uses a "meta-extraction" technique to achieve reliable, arbitrary graph extraction. Instead of asking the model to extract individual entities, we instruct it to return a single, complete JSON object that represents the entire graph for a given document. This approach respects the example-driven design of the `langextract` library while achieving the flexibility of schema-less extraction.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd langextract-graph-transformers
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Configure Environment Variables:**
    Create a `.env` file in the root of the project and add your Google Cloud project details:
    ```
    VERTEX_AI_PROJECT_ID="your-gcp-project-id"
    VERTEX_AI_LOCATION="your-gcp-location" # e.g., us-central1
    ```

4.  **Authenticate with Google Cloud:**
    Ensure your local environment is authenticated to use Vertex AI. The simplest way is to use the `gcloud` CLI:
    ```bash
    gcloud auth application-default login
    ```

## Usage Example

Here's how to use the `LangExtractGraphTransformer` to extract a graph from a piece of text.

```python
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langextract_graph_transformers.langextract_graph_transformer import LangExtractGraphTransformer

# Load environment variables
load_dotenv()

# 1. Initialize the Transformer
project_id = os.getenv("VERTEX_AI_PROJECT_ID")
location = os.getenv("VERTEX_AI_LOCATION")

transformer = LangExtractGraphTransformer(
    project_id=project_id,
    location=location,
    # Optionally ask for specific properties to be extracted
    node_properties=["sector", "location"],
    relationship_properties=["date", "confidence"]
)

# 2. Create a Document to process
text_content = ("Microsoft, a tech giant headquartered in Redmond, announced its acquisition of Activision Blizzard on January 18, 2022.")
document = Document(page_content=text_content)

# 3. Process the document to get the graph
graph_documents = transformer.process_documents([document])

# 4. Inspect the result
if graph_documents:
    graph = graph_documents[0]
    print("---" Nodes "---")
    for node in graph.nodes:
        print(node)
    
    print("\n---" Relationships "---")
    for rel in graph.relationships:
        print(rel)

# Expected Output (will vary based on model generation):
#
# --- Nodes ---
# id='Microsoft' type='Company' properties={'prop_sector': 'Tech', 'prop_location': 'Redmond'}
# id='Activision Blizzard' type='Company' properties={}
#
# --- Relationships ---
# source=id='Microsoft' type='Company' properties={...} target=id='Activision Blizzard' type='Company' properties={} type='ACQUIRED' properties={'prop_date': 'January 18, 2022'}

```

## Running Tests

This project includes both unit and integration tests.

- **Unit tests** mock the `langextract` API to verify the internal logic of the transformer.
- **Integration tests** make real calls to the Vertex AI API and will incur costs.

To run all tests:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m unittest discover tests
```

To run only unit tests:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m unittest tests/test_langextract_graph_transformer.py
```
