import sys
import os
import streamlit as st
import logging
import pandas as pd 

# Adjust path to import from src - Assuming running streamlit from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.search_engine import SemanticSearchEngine
    from src import config
    from src.utils import setup_logger

except ModuleNotFoundError:
    st.error("Could not import project modules. Make sure you are running Streamlit from the project root directory (`llm-semantic-search/`) and all dependencies are installed.")
    st.stop()


# --- Application Setup ---
setup_logger()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="E-commerce Semantic Search", layout="wide")
st.title("🛍️ LLM-Powered E-commerce Semantic Search")
st.markdown("Enter a product query below to find semantically similar items.")

# --- Load Search Engine ---
@st.cache_resource
def load_engine():
    logger.info("Loading Semantic Search Engine for Streamlit app...")
    try:
        engine = SemanticSearchEngine()
        if engine.vector_store.index is None:
            st.error("Vector Index not found or failed to load. Please run `scripts/generate_embeddings.py` first.")
            return None
        logger.info("Search Engine loaded successfully.")
        return engine
    except Exception as e:
        logger.error(f"Error loading search engine in Streamlit app: {e}", exc_info=True)
        st.error(f"Failed to load the search engine: {e}")
        return None

search_engine = load_engine()

# --- Optional: Load Product Data for Display ---
@st.cache_data
def load_product_details():
    logger.info("Loading product details for display...")
    try:
        products_df = pd.read_csv(config.PRODUCT_DATA_PATH, usecols=['product_id', 'product_title']) # Adjust columns
        products_df.set_index('product_id', inplace=True)
        logger.info(f"Loaded details for {len(products_df)} products.")
        return products_df.to_dict('index')
    except FileNotFoundError:
        st.warning("Product details file not found. Search results will only show IDs.")
        return {}
    except Exception as e:
        logger.error(f"Error loading product details: {e}")
        st.warning(f"Could not load product details: {e}")
        return {}

product_details = load_product_details()

# --- Search Interface ---
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input("Enter your search query:", placeholder="e.g., 'comfortable shoes for running'")

with col2:
    k = st.number_input("Number of results:", min_value=1, max_value=50, value=config.DEFAULT_SEARCH_K)

search_button = st.button("Search")

# --- Display Results ---
if search_button and query and search_engine:
    st.markdown("---")
    st.subheader("Search Results")

    with st.spinner("Searching..."):
        try:
            results_ids, results_scores = search_engine.perform_search(query, k=k)

            if results_ids:
                results_data = []
                for pid, score in zip(results_ids, results_scores):
                    details = product_details.get(pid, {'product_title': 'N/A'})
                    results_data.append({
                        "Product ID": pid,
                        "Product Title": details.get('product_title', 'N/A'),
                        "Similarity Score": f"{1 - score:.4f}" if score is not None else "N/A"
                        # Alternative score display: f"{score:.4f} (Distance)"
                    })

                results_df = pd.DataFrame(results_data)
                st.dataframe(results_df)
            else:
                st.warning("No results found for your query.")

        except Exception as e:
            logger.error(f"Error during search execution in Streamlit: {e}", exc_info=True)
            st.error(f"An error occurred during the search: {e}")

elif search_button and not query:
    st.warning("Please enter a search query.")
elif search_button and not search_engine:
     st.error("Search engine is not available. Please check the logs.")


st.sidebar.info(f"""
**Model:** `{config.EMBEDDING_MODEL_NAME}`
**Index:** `{os.path.basename(config.INDEX_FILE)}` ({search_engine.vector_store.index.ntotal if search_engine and search_engine.vector_store.index else 'N/A'} vectors)
""")
