from helper import get_embedding, get_opensearch_client
from pprint import pprint


def keyword_search(query_text, index_name="pdf_content_index", top_k=20):
    """
    Perform keyword search using OpenSearch.

    Args:
        query_text (str): The query text to search for
        index_name (str): OpenSearch index name
        top_k (int): Number of results to return

    Returns:
        list: Search results with metadata
    """
    client = get_opensearch_client("localhost", 9200)

    try:
        # Create a keyword search query
        search_query = {
            "size": top_k,
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            },
            "_source": ["content", "content_type", "file_name"],
        }

        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        
        print(f"Keyword search found {len(hits)} results\n")
        return hits
        
    except Exception as e:
        print(f"✗ Keyword search error: {str(e)}")
        return []


def semantic_search(query_text, index_name="pdf_content_index", top_k=20):
    """
    Perform semantic search using vector embeddings (KNN).

    Args:
        query_text (str): The query text to search for
        index_name (str): OpenSearch index name
        top_k (int): Number of results to return

    Returns:
        list: Search results with metadata
    """
    client = get_opensearch_client("localhost", 9200)

    try:
        # Get embedding for the query using Ollama
        print(f"Generating embedding for query: '{query_text}'...")
        query_embedding = get_embedding(query_text)
        
        # Verify embedding
        if not query_embedding or len(query_embedding) != 768:
            print(f"✗ Invalid embedding: expected 768 dimensions, got {len(query_embedding) if query_embedding else 0}")
            return []
        
        print(f"✓ Embedding generated successfully (768 dimensions)\n")

        # Create a semantic search query using KNN
        search_query = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": top_k
                    }
                }
            },
            "_source": ["content", "content_type", "file_name"],
        }

        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        
        print(f"Semantic search found {len(hits)} results\n")
        return hits
        
    except Exception as e:
        print(f"✗ Semantic search error: {str(e)}")
        return []


def hybrid_search(query_text, index_name="pdf_content_index", top_k=20):
    """
    Perform hybrid search combining keyword and semantic search with weighted scoring.

    Args:
        query_text (str): The query text to search for
        index_name (str): OpenSearch index name
        top_k (int): Number of results to return

    Returns:
        list: Search results with combined scores
    """
    client = get_opensearch_client("localhost", 9200)

    try:
        # Get embedding for the query
        print(f"Generating embedding for query: '{query_text}'...")
        query_embedding = get_embedding(query_text)
        
        # Verify embedding
        if not query_embedding or len(query_embedding) != 768:
            print(f"✗ Invalid embedding: expected 768 dimensions, got {len(query_embedding) if query_embedding else 0}")
            return []
        
        print(f"✓ Embedding generated successfully (768 dimensions)\n")

        # Create a hybrid search query combining KNN and keyword search
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": top_k
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": query_text,
                                    "boost": 1.2
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": ["content", "content_type", "file_name"],
        }

        response = client.search(index=index_name, body=search_query)
        hits = response["hits"]["hits"]
        
        print(f"Hybrid search found {len(hits)} results\n")
        return hits
        
    except Exception as e:
        print(f"✗ Hybrid search error: {str(e)}")
        
        # Fallback to keyword search
        print("Falling back to keyword search...\n")
        try:
            fallback_query = {
                "size": top_k,
                "query": {
                    "match": {
                        "content": {
                            "query": query_text,
                            "fuzziness": "AUTO"
                        }
                    }
                },
                "_source": ["content", "content_type", "file_name"],
            }
            response = client.search(index=index_name, body=fallback_query)
            hits = response["hits"]["hits"]
            
            print(f"Fallback search found {len(hits)} results\n")
            return hits
            
        except Exception as e2:
            print(f"✗ Fallback search error: {str(e2)}")
            return []


def format_search_results(results, max_content_length=300):
    """
    Format search results for display.

    Args:
        results (list): Search results from OpenSearch
        max_content_length (int): Maximum content length to display

    Returns:
        None (prints formatted results)
    """
    if not results:
        print("No results found.\n")
        return

    print("="*80)
    print(f"SEARCH RESULTS ({len(results)} results)")
    print("="*80)

    for i, hit in enumerate(results, 1):
        score = hit.get("_score", 0)
        source = hit.get("_source", {})
        
        content = source.get("content", "")
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
        
        content_type = source.get("content_type", "text")
        file_name = source.get("file_name", "unknown")
        
        print(f"\n[Result {i}]")
        print(f"  Score: {score:.4f}")
        print(f"  Type: {content_type}")
        print(f"  File: {file_name}")
        print(f"  Content: {content}\n")

    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("OPENSEARCH RETRIEVAL TEST")
    print("="*80 + "\n")

    query = "Compare RAG vs fine-tuning"
    index = "pdf_content_index"

    # Test keyword search
    print("1. KEYWORD SEARCH")
    print("-" * 80)
    keyword_results = keyword_search(query, index_name=index, top_k=5)
    format_search_results(keyword_results)

    # Test semantic search
    print("2. SEMANTIC SEARCH")
    print("-" * 80)
    semantic_results = semantic_search(query, index_name=index, top_k=5)
    format_search_results(semantic_results)

    # Test hybrid search
    print("3. HYBRID SEARCH")
    print("-" * 80)
    hybrid_results = hybrid_search(query, index_name=index, top_k=5)
    format_search_results(hybrid_results)