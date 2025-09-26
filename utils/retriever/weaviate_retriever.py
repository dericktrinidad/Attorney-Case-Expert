# utils/retriever/weaviate_retriever.py  (v4)
from typing import List, Dict, Any, Optional
import time
import urllib.request
import weaviate
from weaviate.classes.query import MetadataQuery
from sentence_transformers import SentenceTransformer

class WeaviateRetriever:
    def __init__(
        self,
        host: str = "127.0.0.1",
        http_port: int = 8080,
        grpc_port: int = 50051,
        index: str = "Cases",
        alpha: float = 0.6,
        use_tls: bool = False,
        api_key: Optional[str] = None,
        wait_ready_seconds: int = 30,
    ):
        self.index = index
        self.alpha = alpha
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        conn_kwargs = dict(
            http_host=host, http_port=http_port, http_secure=use_tls,
            grpc_host=host, grpc_port=grpc_port, grpc_secure=use_tls,
        )
        if api_key:
            from weaviate.auth import AuthApiKey
            conn_kwargs["auth_credentials"] = AuthApiKey(api_key=api_key)

        self.client = weaviate.connect_to_custom(**conn_kwargs)
        print("✅ Connected to Weaviate client")
        
        if self.index not in self.client.collections.list_all():
            self.client.close()
            raise RuntimeError(
                f"Collection '{self.index}' not found. Run your ingest to create/load it."
            )
        self.collection = self.client.collections.get(self.index)

    def retrieve_hybrid(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_vec = self.model.encode(query_text, normalize_embeddings=True).tolist()
        res = self.collection.query.hybrid(
            query=query_text,
            vector=query_vec,
            alpha=self.alpha,
            limit=top_k,
            return_metadata=MetadataQuery(score=True),
        )
        
        out: List[Dict[str, Any]] = self._format_results(res)
        return out

    def retrieve_bm25(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        res = self.collection.query.bm25(query=query_text, limit=top_k, return_metadata=MetadataQuery(score=True))
        out: List[Dict[str, Any]] = self._format_results(res)
        return out
    
    def retrieve_semantic(self, query_text: str, top_k: int = 10,) -> List[Dict[str, Any]]:
        vec = self.model.encode(query_text, normalize_embeddings=True).tolist()
        res = self.collection.query.near_vector(vec, limit=top_k,
                                                return_metadata=MetadataQuery(score=True))
        out: List[Dict[str, Any]] = self._format_results(res)
        return out
    
    def _format_results(self, res):
        out= []
        for obj in (res.objects or []):
            p = obj.properties or {}
            txt = p.get("text", "")
            out.append({
                "title": p.get("title"),
                # "court": p.get("court"),
                "date_filed": p.get("date_filed"),
                "url": p.get("url"),
                "text": txt,
                "_score": getattr(obj.metadata, "score", None),
            })  
        return out
        
        
    def close(self):
        try:
            self.client.close()
        except Exception:
            pass