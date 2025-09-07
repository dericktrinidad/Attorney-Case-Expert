# utils/retriever/weaviate_retriever.py

from typing import List, Dict, Any, Optional
import weaviate


class WeaviateRetriever:
    """
    Retriever wrapper around Weaviate hybrid/vector search.
    """

    def __init__(self, url: str, api_key: Optional[str] = None, index: str = "Movies", alpha: float = 0.6):
        self.url = url
        self.api_key = api_key
        self.index = index
        self.alpha = alpha

        if api_key:
            auth = weaviate.auth.AuthApiKey(api_key=api_key)
            self.client = weaviate.Client(url, auth_client_secret=auth)
        else:
            self.client = weaviate.Client(url)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search (semantic + keyword).
        Returns a list of dicts with movie properties and scores.
        """
        try:
            res = (
                self.client.query
                .get(self.index, ["title", "overview", "year"])
                .with_hybrid(query=query, alpha=self.alpha)
                .with_limit(top_k)
                .do()
            )

            hits = res.get("data", {}).get("Get", {}).get(self.index, [])
            results = []
            for h in hits:
                results.append({
                    "title": h.get("title"),
                    "overview": h.get("overview"),
                    "year": h.get("year"),
                    "_score": h.get("_additional", {}).get("score"),
                })
            return results

        except Exception as e:
            print(f"[WeaviateRetriever] Error during retrieval: {e}")
            return []
