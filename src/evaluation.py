import numpy as np
import logging
from . import config
from .utils import timeit

logger = logging.getLogger(__name__)

def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int):
    """Calculates Precision@K."""
    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k: return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return hits / len(retrieved_k) 

def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int):
    """Calculates Recall@K."""
    if not relevant_ids: return 1.0 
    if not relevant_ids and not retrieved_ids: return 1.0
    if not relevant_ids: return 0.0

    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]
    if not retrieved_k: return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)

def average_precision(retrieved_ids: list, relevant_ids: set):
    """Calculates Average Precision (AP)."""
    if not relevant_ids: return 1.0 if not retrieved_ids else 0.0 
    if not retrieved_ids: return 0.0

    ap = 0.0
    hits = 0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            precision_at_i = hits / (i + 1)
            ap += precision_at_i

    if hits == 0: return 0.0
    return ap / hits 

def mean_reciprocal_rank(retrieved_ids_list: list, relevant_ids_list: list):
    """Calculates Mean Reciprocal Rank (MRR)."""
    total_rr = 0.0
    query_count = 0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        if not relevant: 
           continue
        query_count += 1
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                total_rr += 1.0 / (i + 1)
                break 

    if query_count == 0:
        logger.warning("MRR calculation had no valid queries.")
        return 0.0
    return total_rr / query_count

def ndcg_at_k(retrieved_ids: list, relevance_scores: dict, k: int):
    """Calculates Normalized Discounted Cumulative Gain (nDCG)@K."""
    if k <= 0: return 0.0
    retrieved_k = retrieved_ids[:k]

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k):
        relevance = relevance_scores.get(doc_id, 0)
        dcg += relevance / np.log2(i + 2) 

    ideal_sorted_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(ideal_sorted_scores[:k]):
        idcg += relevance / np.log2(i + 2)

    if idcg == 0:
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
    query_details = [] 

    logger.info(f"Starting evaluation on {len(evaluation_data)} queries...")

    max_k = max(k_values) if k_values else config.DEFAULT_SEARCH_K

    for item in evaluation_data:
        query = item['query']
        query_id = item['query_id']
        relevant_ids = item['relevant_product_ids']
        relevance_scores = item['relevance_scores']

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


    metrics = {}

    # P@K and Recall@K
    for k in k_values:
        precisions = [precision_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        recalls = [recall_at_k(ret, rel, k) for ret, rel in zip(all_retrieved_ids, all_relevant_ids)]
        metrics[f'P@{k}'] = np.mean(precisions) if precisions else 0.0
        metrics[f'Recall@{k}'] = np.mean(recalls) if recalls else 0.0

    # MRR
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

