# LLM-Powered Semantic Search for E-commerce

This project implements and evaluates an LLM-powered semantic search system for e-commerce product discovery, based on the proposal by Ansh Mehta and Shivam Sah. It leverages sentence embeddings and vector databases to understand user intent beyond simple keyword matching.


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