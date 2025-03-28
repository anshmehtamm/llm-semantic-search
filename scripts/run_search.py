import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.search_engine import SemanticSearchEngine

def main():
    # Initialize the search engine
    search_engine = SemanticSearchEngine()
    
    # Example search
    query = "Coffee Machine with pods"
    print(f"\nPerforming search for: '{query}'")
    results, scores = search_engine.perform_search(query, k=5)
    print(f"Results: {results}")
    print(f"Scores: {scores}")

if __name__ == "__main__":
    main() 