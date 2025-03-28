import numpy as np
import logging
from . import config
from .utils import timeit

logger = logging.getLogger(__name__)

# --- Core Metric Implementations ---

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int):
    """Calculates Precision@K."""
    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k: return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return hits / len(retrieved_k) # Use len(retrieved_k) instead of K for cases where fewer than K results are returned

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int):
    """Calculates Recall@K."""
    if not relevant_ids: return 1.0 # Or 0.0? Define behavior for no relevant docs. Let's say 1.0 if retrieved is also empty, 0.0 otherwise.
    if not relevant_ids and not retrieved_ids: return 1.0
    if not relevant_ids: return 0.0

    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k: return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def average_precision(retrieved_ids: list, relevant_ids: set):
    """Calculates Average Precision (AP)."""
    if not relevant_ids: return 1.0 if not retrieved_ids else 0.0 # Handle edge case
    if not retrieved_ids: return 0.0

    ap = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            precision_at_i = hits / (i + 1)
            ap += precision_at_i

    if hits == 0: return 0.0
    return ap / hits # Original definition uses |relevant_ids|, but dividing by hits is common in IR. Let's use hits.

def mean_reciprocal_rank(retrieved_ids_list: list, relevant_ids_list: list):
    """Calculates Mean Reciprocal Rank (MRR)."""
    total_rr = 0.0
    query_count = 0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        if not relevant: # Skip queries with no relevant documents defined? Or assign RR=0? Let's skip for now.
           # logger.warning("Query skipped in MRR calculation due to empty relevant set.")
           continue
        query_count += 1
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                total_rr += 1.0 / (i + 1)
                break # Only consider the first relevant hit
        # If no relevant doc found in retrieved list, RR for this query is 0, so total_rr doesn't change.

    if query_count == 0:
        logger.warning("MRR calculation had no valid queries.")
        return 0.0
    return total_rr / query_count

def ndcg_at_k(retrieved_ids: list, relevance_scores: dict, k: int):
    """Calculates Normalized Discounted Cumulative Gain (nDCG)@K."""
    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]

    # Calculate DCG@K for the retrieved list
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        # Get relevance score, default to 0 if not found (treats irrelevant/not judged as 0)
        relevance = relevance_scores.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2) # log base 2, index starts from 1 (i+2 because i starts at 0)

    # Calculate Ideal DCG@K (IDCG@K)
    # Sort the relevance scores of *all* judged documents for this query (descending)
    # Take the top K scores from this ideal ranking.
    ideal_sorted_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted_scores[:k]):
        idcg += relevance / np.log2(i + 2)

    if idcg == 0:
        # If ideal DCG is 0, nDCG is 0 (unless DCG is also 0, then debatable: 0 or 1). Let's use 0.
        return 0.0
    else:
        return dcg / idcg

# --- Evaluation Runner ---

@timeit
def evaluate_search_engine(search_engine, evaluation_data, k_values=config.EVALUATION_K):
    """
    Runs the evaluation pipeline on the search engine.

    Args:
        search_engine (SemanticSearchEngine): The search engine instance to evaluate.
        evaluation_data (list): List of dicts from load_evaluation_data.
        k_values (list[int]): List of K values for metric calculation.

    Returns:
        dict: A dictionary containing aggregated evaluation metrics.
    """
    all_retrieved_ids = []
    all_relevant_ids = []
    all_relevance_scores = []
    query_details = [] # Store per-query results if needed

    logger.info(f"Starting evaluation on {len(evaluation_data)} queries...")

    max_k = max(k_values) if k_values else config.DEFAULT_SEARCH_K

    for item in evaluation_data:
        query = item['query']
        query_id = item['query_id']
        relevant_ids = item['relevant_product_ids']
        relevance_scores = item['relevance_scores'] # {product_id: score}

        # Perform search - retrieve enough results for the max K needed
        retrieved_ids, _ = search_engine.perform_search(query, k=max_k)

        all_retrieved_ids.append(retrieved_ids)
        all_relevant_ids.append(relevant_ids)
        all_relevance_scores.append(relevance_scores)
        query_details.append({
            'query_id': query_id,
            'query': query,
            'retrieved': retrieved_ids,
            'relevant': relevant_ids,
            'scores': relevance_scores
        })

    logger.info("Search complete for all evaluation queries. Calculating metrics...")

    # --- Calculate Metrics ---
    metrics = {}

    # P@K and Recall@K
    for k in k_values:
        precisions = [precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        metrics[f'P@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0

    # MRR
    # We need to filter pairs where relevant_ids is empty for the standard MRR calculation
    valid_mrr_pairs = [(ret, rel) for ret, rel in zip(all_retrieved_ids, all_relevant_ids) if rel]
    if valid_mrr_pairs:
         mrr_retrieved, mrr_relevant = zip(*valid_mrr_pairs)
         metrics['MRR'] = mean_reciprocal_rank(list(mrr_retrieved), list(mrr_relevant))
    else:
         metrics['MRR'] = 0.0
         logger.warning("MRR calculation skipped as no queries had defined relevant documents.")


    # nDCG@K
    for k in k_values:
        ndcgs = [ndcg_at_k(ret, scores, k) for ret, scores in zip(all_retrieved_ids, all_relevance_scores)]
        metrics[f'nDCG@{k}'] = np.mean(ndcgs) if ndcgs else 0.0


    logger.info("Evaluation metrics calculated.")
    return metrics


# Example usage (for testing)
if __name__ == '__main__':
    logger.info("Testing evaluation metrics...")

    # Test case 1
    retrieved1 = ['p1', 'p3', 'p2', 'p5', 'p4']
    relevant1 = {'p1', 'p2', 'p4'}
    scores1 = {'p1': 3, 'p2': 2, 'p3': 0, 'p4': 3, 'p5': 1, 'p6': 2} # Includes a relevant doc (p6) not retrieved

    # Test case 2
    retrieved2 = ['p7', 'p8']
    relevant2 = {'p9'}
    scores2 = {'p7': 1, 'p8': 0, 'p9': 3}

     # Test case 3 (no relevant docs)
    retrieved3 = ['p10', 'p11']
    relevant3 = set()
    scores3 = {'p10':0, 'p11': 0}

    # --- Individual Metric Tests ---
    k = 3
    print(f"\n--- Test Case 1 (k={k}) ---")
    print(f"Retrieved: {retrieved1}")
    print(f"Relevant: {relevant1}")
    print(f"Scores: {scores1}")
    p_at_k = precision_at_k(retrieved1, relevant1, k)
    r_at_k = recall_at_k(retrieved1, relevant1, k)
    ap = average_precision(retrieved1, relevant1)
    ndcg_at_k_val = ndcg_at_k(retrieved1, scores1, k)
    print(f"P@{k}: {p_at_k:.4f}")   # Expected: 2/3 = 0.6667 (p1, p2 hit within top 3)
    print(f"Recall@{k}: {r_at_k:.4f}") # Expected: 2/3 = 0.6667 (p1, p2 hit / total relevant p1,p2,p4)
    print(f"AP: {ap:.4f}") # Expected: (1/1 + 2/3 + 3/5) / 3 = (1 + 0.6667 + 0.6) / 3 = 2.2667 / 3 = 0.7556 (using hits in denom)
                          #           (1/1 + 2/3 + 3/5) / 3 = 0.7556 # Correct AP calc
                          # AP = (P@1_if_hit + P@2_if_hit + ... P@N_if_hit) / num_relevant_docs
                          # Hits at pos 1 (p1), 3 (p2), 5 (p4)
                          # AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.6667 + 0.6) / 3 = 2.2667 / 3 = 0.7556
    print(f"nDCG@{k}: {ndcg_at_k_val:.4f}")
    # DCG@3 = score(p1)/log2(2) + score(p3)/log2(3) + score(p2)/log2(4)
    # DCG@3 = 3/1 + 0/1.585 + 2/2 = 3 + 0 + 1 = 4.0
    # Ideal order: p1(3), p4(3), p2(2), p6(2), p5(1), p7(1), p3(0), p8(0), p10(0), p11(0) ...
    # Ideal scores: 3, 3, 2, 2, 1, 1, 0, 0 ...
    # IDCG@3 = 3/log2(2) + 3/log2(3) + 2/log2(4)
    # IDCG@3 = 3/1 + 3/1.585 + 2/2 = 3 + 1.8927 + 1 = 5.8927
    # nDCG@3 = 4.0 / 5.8927 = 0.6788

    # --- MRR Test ---
    all_ret = [retrieved1, retrieved2, retrieved3]
    all_rel = [relevant1, relevant2, relevant3]
    mrr = mean_reciprocal_rank(all_ret, all_rel)
    # RR1 = 1/1 (p1 found at pos 1)
    # RR2 = 0 (p9 not found)
    # RR3 = skipped or 0 (no relevant) -> using skip -> MRR = (1.0 + 0.0) / 2 = 0.5
    print(f"\n--- MRR Test ---")
    print(f"MRR: {mrr:.4f}") # Expected: (1/1 + 0) / 2 = 0.5 (excluding query 3)

    # --- Test Evaluation Runner (requires a mock search engine) ---
    class MockSearchEngine:
        def perform_search(self, query, k):
            # Simple mock: return fixed results based on query
            if "query 1" in query: return retrieved1, [0.1] * len(retrieved1)
            if "query 2" in query: return retrieved2, [0.2] * len(retrieved2)
            if "query 3" in query: return retrieved3, [0.3] * len(retrieved3)
            return [], []

    mock_eval_data = [
        {'query_id': 'q1', 'query': 'query 1', 'relevant_product_ids': relevant1, 'relevance_scores': scores1},
        {'query_id': 'q2', 'query': 'query 2', 'relevant_product_ids': relevant2, 'relevance_scores': scores2},
         {'query_id': 'q3', 'query': 'query 3', 'relevant_product_ids': relevant3, 'relevance_scores': scores3},
    ]
    mock_engine = MockSearchEngine()
    print("\n--- Testing evaluate_search_engine ---")
    results = evaluate_search_engine(mock_engine, mock_eval_data, k_values=[3, 5])
    print("Evaluation Results:")
    import json
    print(json.dumps(results, indent=2))

    # Expected rough values (re-check calculations if needed):
    # P@3: mean(0.6667, 0, 0) = 0.2222
    # Recall@3: mean(0.6667, 0, 0) = 0.2222
    # MRR: 0.5 (as calculated above)
    # nDCG@3: mean(0.6788, 0, 0) = 0.2263
    # P@5: mean(3/5, 0, 0) = mean(0.6, 0, 0) = 0.2
    # Recall@5: mean(3/3, 0, 0) = mean(1.0, 0, 0) = 0.3333
    # nDCG@5: mean(ndcg5_q1, 0, 0)
    #   DCG5_q1 = 4.0 + score(p5)/log2(5) + score(p4)/log2(6) = 4.0 + 1/2.32 + 3/2.58 = 4.0 + 0.43 + 1.16 = 5.59
    #   IDCG5_q1 = 5.8927 + score(p6)/log2(5) + score(p5)/log2(6) = 5.8927 + 2/2.32 + 1/2.58 = 5.8927 + 0.86 + 0.39 = 7.14
    #   nDCG5_q1 = 5.59 / 7.14 = 0.7829
    #   nDCG@5 = mean(0.7829, 0, 0) = 0.2610 