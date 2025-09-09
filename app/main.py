# app/main.py
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

# v4 retriever (must pass http_port + grpc_port inside it)
from utils.retriever.weaviate_retriever import WeaviateRetriever


# --------- Config from env ---------
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "weaviate")     # in Docker compose network; use 127.0.0.1 on host
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_CLASS = os.getenv("WEAVIATE_CLASS", "Movies")
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.6"))
USE_TLS = os.getenv("WEAVIATE_TLS", "false").lower() in ("1", "true", "yes")

# --------- Response models ---------
class Hit(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    director: Optional[str] = None
    cast: Optional[str] = None
    country: Optional[str] = None
    release_year: Optional[int] = None
    rating: Optional[str] = None
    duration: Optional[str] = None
    listed_in: Optional[str] = None
    _score: Optional[float] = None

class SearchResponse(BaseModel):
    results: List[Hit]

# --------- Lifespan ---------
@asynccontextmanager
async def lifespan(app: FastAPI):
    retriever = None
    try:
        retriever = WeaviateRetriever(
            host=WEAVIATE_HOST,
            http_port=WEAVIATE_HTTP_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
            index=WEAVIATE_CLASS,
            alpha=HYBRID_ALPHA,
            use_tls=USE_TLS,
        )
        print("[startup] Retriever initialized")
    except Exception as e:
        print(f"[startup] Retriever init failed: {e}")
        retriever = None

    app.state.retriever = retriever
    try:
        yield
    finally:
        try:
            if app.state.retriever:
                app.state.retriever.close()
        except Exception:
            pass

app = FastAPI(title="CinemaRAG Retriever API", lifespan=lifespan)

# --------- Routes ---------
@app.get("/health")
def health():
    """Always returns 200 so you can verify the server is up even if Weaviate isn't."""
    return {
        "ok": True,
        "weaviate_http": f"{'https' if USE_TLS else 'http'}://{WEAVIATE_HOST}:{WEAVIATE_HTTP_PORT}",
        "weaviate_grpc": f"{WEAVIATE_HOST}:{WEAVIATE_GRPC_PORT}",
        "index": WEAVIATE_CLASS,
        "retriever_initialized": bool(getattr(app.state, "retriever", None)),
    }

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1), top_k: int = Query(10, ge=1, le=100)):
    r = getattr(app.state, "retriever", None)
    if r is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    return {"results": r.retrieve(q, top_k)}