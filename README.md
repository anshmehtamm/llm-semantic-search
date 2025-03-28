# LLM-Powered Semantic Search for E-commerce

This project implements and evaluates an LLM-powered semantic search system for e-commerce product discovery, based on the proposal by Ansh Mehta and Shivam Sah. It leverages sentence embeddings and vector databases to understand user intent beyond simple keyword matching.

## Project Structure

```
llm-semantic-search/
├── data/                     # Dataset storage (Amazon Shopping Queries Dataset)
├── index/                    # Storage for generated FAISS index and ID map
├── src/                      # Source code modules
│   ├── config.py             # Configuration (paths, model names, etc.)
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── embedding_generator.py # Embedding generation using SentenceTransformers
│   ├── vector_store.py       # Vector database management (FAISS implementation)
│   ├── search_engine.py      # Core search logic combining embeddings and vector search
│   ├── evaluation.py         # Implementation of P@K, MRR, nDCG metrics
│   └── utils.py              # Utility functions (logging, timing)
├── scripts/                  # Executable scripts
│   ├── generate_embeddings.py # Script to build and save the search index
│   └── run_evaluation.py      # Script to evaluate the search engine performance
├── app.py                    # Streamlit web application for interactive search
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repo-url>
    cd llm-semantic-search
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `faiss-cpu` is included. If you have a compatible NVIDIA GPU and CUDA installed, you can install `faiss-gpu` instead for better performance: `pip uninstall faiss-cpu && pip install faiss-gpu`.*

4.  **Download Dataset:**
    *   Obtain the Amazon Shopping Queries Dataset (following instructions from the source, e.g., Amazon Science).
    *   Place the relevant dataset files (e.g., product catalog, queries, judgments) into the `data/` directory.
    *   **Crucially:** Update the file paths and loading logic in `src/config.py` and `src/data_loader.py` to match the actual names and structure of your dataset files. The current `data_loader.py` contains placeholder logic.

## Usage

1.  **Generate Embeddings and Build Index:**
    *   This step processes the product data, generates embeddings, and creates the FAISS index. It needs to be run once initially or whenever the product data or embedding model changes.
    *   **Warning:** This can be computationally intensive and time-consuming, especially on large datasets. Ensure you have sufficient RAM/GPU memory.
    ```bash
    python scripts/generate_embeddings.py
    ```
    *   This will save an index file (e.g., `index/all-MiniLM-L6-v2_faiss.index`) and an ID map (`index/all-MiniLM-L6-v2_id_map.json`).

2.  **Run Evaluation:**
    *   This script loads the evaluation queries and judgments, uses the search engine (with the pre-built index) to get results for each query, and calculates metrics (P@K, MRR, nDCG).
    *   Ensure the evaluation data loading in `src/data_loader.py` is correctly configured for your dataset format.
    ```bash
    python scripts/run_evaluation.py
    ```
    *   Results will be printed to the console and saved to a JSON file in the project root.

3.  **Run the Streamlit Prototype:**
    *   This launches a simple web interface for interactive searching.
    *   Make sure you are in the project root directory (`llm-semantic-search/`).
    ```bash
    streamlit run app.py
    ```
    *   Open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).

## Configuration

Key parameters can be adjusted in `src/config.py`:

*   `EMBEDDING_MODEL_NAME`: Choose different SentenceTransformer models from HuggingFace.
*   `DATA_DIR`, `PRODUCT_DATA_PATH`, etc.: **Must be updated** to point to your dataset files.
*   `INDEX_DIR`, `INDEX_FILE`, `ID_MAP_FILE`: Location for index files.
*   `DEFAULT_SEARCH_K`: Default number of results for search.
*   `EVALUATION_K`: K values used for evaluation metrics.
*   `DEVICE`: Set to `"cuda"` if using GPU, otherwise `"cpu"`. 