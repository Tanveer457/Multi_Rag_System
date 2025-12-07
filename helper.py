import requests
from opensearchpy import OpenSearch

def get_embedding(prompt, model="nomic-embed-text"):
    url = "http://localhost:11434/api/embeddings"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json().get("embedding")

def get_opensearch_client(host="localhost", port=9200):
    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=30,
        max_retries=4,
        retry_on_timeout=True,
        use_ssl=False,
        verify_certs=False
    )

    try:
        if client.ping():
            print("Connected to OpenSearch")
        else:
            print("⚠ OpenSearch responded but ping failed")
    except Exception as e:
        print("❌ Failed to connect to OpenSearch:", e)

    return client


if __name__ == "__main__":
    client = get_opensearch_client()
    print(client)
