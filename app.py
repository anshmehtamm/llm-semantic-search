# streamlit_app.py

import sys
import os
import streamlit as st
import logging
import pandas as pd
from pathlib import Path
import time # For simulating delays if needed

# Adjust path to import from src - Assuming running streamlit from project root
# Ensure this path adjustment works for your structure.
# It assumes 'streamlit_app.py' is in the project root alongside the 'src' folder.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
PROJECT_ROOT = Path(__file__).parent

try:
    from src.search_engine import SemanticSearchEngine
    from src import config # Use config for defaults and paths
    from src.utils import setup_logger
    # Try importing the text preparation function for consistency display if needed
    from src.semantic_search_engine import prepare_text_for_product
except ModuleNotFoundError as e:
    st.error(f"""
    **Error importing project modules:** {e}.

    Please ensure:
    1. You are running Streamlit from the project root directory (`llm-semantic-search/`).
       Command: `streamlit run streamlit_app.py`
    2. All dependencies are installed (`pip install -r requirements.txt`).
    3. The `src` directory exists and contains the necessary files (`config.py`, `search_engine.py`, etc.).
    """)
    st.stop()
except ImportError as e:
    # Catch potential import errors within the modules themselves
     st.error(f"""
    **Error during module import:** {e}.

    This might indicate an issue within the imported files or missing dependencies.
    Check the console logs for more details.
    """)
     st.stop()


# --- Application Setup ---
setup_logger() # Configure logging
logger = logging.getLogger(__name__)

st.set_page_config(page_title="E-commerce Semantic Search", layout="wide")

st.markdown("<h1 style='text-align: center;'>🛍️ Advanced E-commerce Semantic Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Search for products using natural language and compare semantic search with cross-encoder re-ranking.</p>", unsafe_allow_html=True)
st.divider()

# --- Load Search Engine ---
# Cache the engine resource to avoid reloading models on every interaction
@st.cache_resource
def load_engine(model_type='finetuned'):
    """Loads the SemanticSearchEngine, attempts to load specified reranker."""
    logger.info(f"Attempting to load Semantic Search Engine (Reranker type: {model_type})...")
    start_time = time.time()
    try:
        # Pass model_type to constructor to load correct reranker
        engine = SemanticSearchEngine(model_type=model_type)
        if engine.vector_store.index is None:
            st.error("Vector Index not found or failed to load. Please run `scripts/generate_embeddings.py` first.")
            logger.error("Vector index loading failed.")
            return None
        if not hasattr(engine, 'all_products_details') or engine.all_products_details is None:
             st.warning("Product details failed to load during engine initialization. Display will be limited.")
             logger.warning("Product details missing in loaded engine.")

        # Check if the desired reranker actually loaded
        if model_type != 'none' and engine.cross_encoder is None:
            st.warning(f"Requested reranker type '{model_type}' failed to load (check logs/model path). Re-ranking will be disabled.")
            logger.warning(f"Cross-encoder (type: {model_type}) failed to load.")

        end_time = time.time()
        logger.info(f"Search Engine loaded in {end_time - start_time:.2f} seconds.")
        return engine
    except Exception as e:
        logger.error(f"Fatal error loading search engine in Streamlit app: {e}", exc_info=True)
        st.error(f"Fatal error loading the search engine: {e}. Please check logs.")
        return None

# --- Load Product Data (for display - separate cache) ---
# Cache the data itself
@st.cache_data
def load_product_display_data(locale='us'):
    """Loads product data specifically required for display."""
    logger.info("Loading product details for display...")
    try:
        # Load only necessary columns for display to save memory
        # Add 'product_image_url' if you have that column
        cols_to_load = ['product_id', 'product_title', 'product_description', 'product_bullet_point', 'product_brand', 'product_color']
        # Determine correct path from config
        product_file_path = config.PRODUCT_DATA_PATH # Assuming Parquet or adjust based on actual file
        if not os.path.exists(product_file_path):
             raise FileNotFoundError(f"Product file not found at {product_file_path}")

        if str(product_file_path).endswith('.parquet'):
            df = pd.read_parquet(product_file_path, columns=cols_to_load)
        elif str(product_file_path).endswith('.csv'):
             df = pd.read_csv(product_file_path, usecols=lambda c: c in cols_to_load) # Load only needed cols
        else:
             raise ValueError(f"Unsupported product file format: {product_file_path}")

        df_locale = df[df['product_locale'] == locale].copy()
        logger.info(f"Loaded details for {len(df_locale)} products for locale '{locale}'.")
        df_locale.set_index('product_id', inplace=True)
        # Convert to dict for faster lookup if needed, or keep as df
        return df_locale # Return DataFrame directly
    except FileNotFoundError:
        st.warning(f"Product data file not found ({config.PRODUCT_DATA_PATH}). Product details will not be shown.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        logger.error(f"Error loading product display data: {e}", exc_info=True)
        st.warning(f"Could not load product details: {e}")
        return pd.DataFrame()

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Search Configuration")
cfg_k_initial = st.sidebar.number_input(
    "Initial Candidates (k_initial)",
    min_value=5, max_value=200, value=config.RERANK_CANDIDATE_COUNT, step=5,
    help="Number of results fetched by the initial semantic search."
)
cfg_k_final = st.sidebar.number_input(
    "Final Results (k_final)",
    min_value=1, max_value=50, value=config.DEFAULT_SEARCH_K, step=1,
    help="Number of results to display after optional re-ranking."
)

cfg_rerank = st.sidebar.toggle("Enable Re-ranking", value=True, help="Use a Cross-Encoder model to re-rank the initial semantic search results.")

cfg_model_type = 'none' # Default if reranking is off
if cfg_rerank:
    cfg_model_type = st.sidebar.radio(
        "Re-ranker Model",
        options=['finetuned', 'base'],
        index=0, # Default to 'finetuned'
        format_func=lambda x: "Fine-Tuned (Recommended)" if x == 'finetuned' else "Base MS-MARCO",
        help="Select which Cross-Encoder model to use for re-ranking."
    )

# Load engine based on sidebar selection - this triggers cache reload if model_type changes
search_engine = load_engine(model_type=cfg_model_type if cfg_rerank else 'none')

# Display model/index info from the loaded engine
st.sidebar.divider()
st.sidebar.markdown("**Engine Status**")
if search_engine:
    st.sidebar.success("✅ Engine Loaded")
    st.sidebar.info(f"""
    **Embedding Model:** `{config.EMBEDDING_MODEL_NAME}`
    **Index:** `{os.path.basename(config.INDEX_FILE)}`
    **Vectors:** `{search_engine.vector_store.index.ntotal if search_engine.vector_store.index else 'N/A'}`
    **Reranker Loaded:** `{'Yes (' + cfg_model_type + ')' if cfg_rerank and search_engine.cross_encoder else ('No' if cfg_rerank else 'Disabled')}`
    """)
    # Load product details after engine is loaded
    product_details_df = load_product_display_data()

else:
    st.sidebar.error("❌ Engine Failed to Load")
    product_details_df = pd.DataFrame() # Ensure it's defined as empty

# --- Main Search Interface ---
query = st.text_input("Enter your search query:", placeholder="e.g., 'matcha tea ceremony set bamboo'", key="query_input")
search_button = st.button("🔍 Search", type="primary")

# --- Search Execution and Display ---
if search_button and query and search_engine:
    st.markdown("---")
    st.subheader("Processing Search...")

    results_placeholder = st.empty() # Placeholder for results display

    with st.spinner("Performing initial semantic search..."):
        try:
            # --- Stage 1: Semantic Search ---
            logger.info(f"Performing Stage 1 Search: k={cfg_k_initial}, query='{query}'")
            # Get more candidates if reranking might happen
            num_to_fetch = cfg_k_initial if cfg_rerank and search_engine.cross_encoder else cfg_k_final
            query_embedding = search_engine.embedder.encode([query], show_progress_bar=False)

            if query_embedding is None or len(query_embedding) == 0:
                st.error("Failed to generate query embedding.")
                st.stop()

            initial_ids, initial_scores = search_engine.vector_store.search(query_embedding[0], k=num_to_fetch)
            logger.info(f"Stage 1 Search returned {len(initial_ids)} results.")

            if not initial_ids:
                st.warning("No results found from initial semantic search.")
                st.stop()

            # Prepare initial results data (convert distance to similarity maybe?)
            # Score: Lower distance is better. Display as is for clarity.
            initial_results = [{"product_id": pid, "score": score} for pid, score in zip(initial_ids, initial_scores)]

        except Exception as e:
            logger.error(f"Error during Stage 1 search: {e}", exc_info=True)
            st.error(f"An error occurred during semantic search: {e}")
            st.stop() # Stop execution if stage 1 fails

    # --- Stage 2: Re-ranking (Optional) ---
    reranked_results = None # Initialize to None
    if cfg_rerank and search_engine.cross_encoder and search_engine.products_df_for_rerank is not None:
        with st.spinner(f"Re-ranking top {len(initial_ids)} candidates with '{cfg_model_type}' model..."):
            try:
                logger.info(f"Performing Stage 2 Re-ranking: model_type='{cfg_model_type}'")
                rerank_pairs = []
                valid_ids_for_rerank = []
                for pid in initial_ids:
                    try:
                        product_text = search_engine.products_df_for_rerank.loc[pid, 'product_text']
                        rerank_pairs.append([query, product_text])
                        valid_ids_for_rerank.append(pid)
                    except KeyError:
                        logger.warning(f"Product ID {pid} from initial search not found in lookup data for reranking.")

                if rerank_pairs:
                    cross_scores = search_engine.cross_encoder.predict(rerank_pairs, show_progress_bar=False) # No bar needed here usually
                    # Score: Higher score is better.
                    sorted_reranked = sorted(zip(valid_ids_for_rerank, cross_scores), key=lambda x: x[1], reverse=True)
                    # Prepare final list based on k_final
                    reranked_results = [{"product_id": pid, "score": float(score)} for pid, score in sorted_reranked[:cfg_k_final]]
                    logger.info(f"Stage 2 Re-ranking complete. Top {cfg_k_final} results selected.")
                else:
                    logger.warning("No valid pairs found for re-ranking after data lookup.")
                    st.warning("Re-ranking could not be performed (no product text found for initial results). Showing semantic search results only.")

            except Exception as e:
                logger.error(f"Error during Stage 2 re-ranking: {e}", exc_info=True)
                st.error(f"An error occurred during re-ranking: {e}. Showing semantic search results only.")
                reranked_results = None # Ensure it's None if error occurs

    # --- Display Results ---
    st.subheader("Search Results Comparison")

    # Helper function to display product details nicely
    def display_product(rank, result_data, score_label="Score"):
        product_id = result_data["product_id"]
        score = result_data["score"]
        details = {}
        if not product_details_df.empty:
             try:
                 details = product_details_df.loc[product_id].to_dict()
             except KeyError:
                 logger.warning(f"Product details not found for ID: {product_id} during display.")
                 details = {} # Ensure details is a dict

        title = details.get('product_title', 'N/A')
        brand = details.get('product_brand', 'N/A')
        color = details.get('product_color', 'N/A')
        description = details.get('product_description', 'N/A')
        bullets = details.get('product_bullet_point', None)

        # Use an expander for details
        with st.expander(f"**{rank}. {title}** (ID: {product_id}) - {score_label}: {score:.4f}", expanded=(rank <= 3)): # Expand top 3
            st.markdown(f"**Brand:** {brand}")
            st.markdown(f"**Color:** {color}")
            if description and pd.notna(description):
                 st.markdown(f"**Description:** {description[:300]}{'...' if len(description)>300 else ''}") # Truncate long description
            if bullets and pd.notna(bullets):
                 st.markdown("**Key Features:**")
                 # Split bullet points if they are separated by a specific character (e.g., '\n' or '▶︎')
                 # Adjust the splitter as needed based on your data format
                 bullet_list = str(bullets).split('▶︎') # Example splitter
                 for i, bullet in enumerate(bullet_list):
                      if bullet.strip():
                          st.markdown(f"- {bullet.strip()}")
                      if i >= 4 : # Limit number of displayed bullets
                          st.markdown("- ...")
                          break
            # Add image display if you have 'product_image_url' column
            # image_url = details.get('product_image_url', None)
            # if image_url:
            #     st.image(image_url, width=150)


    # Display side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4>Semantic Search Results (Vector Similarity)</h4>", unsafe_allow_html=True)
        st.caption(f"Top {len(initial_results)} based on embedding distance (lower score is better).")
        if not initial_results:
            st.info("No initial semantic search results.")
        else:
            # Display up to k_final results from initial search for comparison
            for i, res in enumerate(initial_results[:cfg_k_final]):
                display_product(i + 1, res, score_label="Distance")

    with col2:
        st.markdown(f"<h4>Re-ranked Results ({'Enabled' if cfg_rerank else 'Disabled'})</h4>", unsafe_allow_html=True)
        if cfg_rerank and reranked_results:
             st.caption(f"Top {len(reranked_results)} re-ranked by '{cfg_model_type}' model (higher score is better).")
             for i, res in enumerate(reranked_results):
                 display_product(i + 1, res, score_label="Relevance Score")
        elif cfg_rerank and search_engine.cross_encoder is None:
             st.warning(f"Re-ranking enabled, but '{cfg_model_type}' model failed to load.")
        elif cfg_rerank and not reranked_results:
             st.info("Re-ranking did not produce results (check logs).")
        else:
             st.info("Re-ranking is disabled in the sidebar configuration.")


elif search_button and not query:
    st.warning("Please enter a search query.")
elif search_button and not search_engine:
     st.error("Search engine is not available. Please check the application logs.")

# Add some padding at the bottom
st.markdown("<br><br>", unsafe_allow_html=True)