# ingest/ingest_movies.py
import os, time, urllib.request
from tqdm import tqdm
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from sentence_transformers import SentenceTransformer

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")   # "weaviate" if running inside the app container
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
CSV_PATH = os.getenv("CSV_PATH", "./data/raw/netflix/netflix_titles.csv")
CLASS = os.getenv("WEAVIATE_CLASS", "Movies")

print(f"CONNECTING TO CLIENT {WEAVIATE_HTTP_PORT}")
client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST, http_port=WEAVIATE_HTTP_PORT, http_secure=False,
    grpc_host=WEAVIATE_HOST, grpc_port=WEAVIATE_GRPC_PORT, grpc_secure=False,
)

print("✅ Connected to Weaviate client")
        
try:
    # Create collection if missing (BYO vectors => VectorConfig.none)
    if CLASS not in client.collections.list_all():
        client.collections.create(
            name=CLASS,
            properties=[
                Property(name="type", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="director", data_type=DataType.TEXT),
                Property(name="cast", data_type=DataType.TEXT),
                Property(name="country", data_type=DataType.TEXT),
                Property(name="release_year", data_type=DataType.INT),
                Property(name="rating", data_type=DataType.TEXT),
                Property(name="duration", data_type=DataType.TEXT),
                Property(name="listed_in", data_type=DataType.TEXT),
                Property(name="description", data_type=DataType.TEXT),
            ],
            # ✅ correct in v4:
            vector_config=Configure.Vectors.self_provided(),
        )
    else:
        print(f"Collection '{CLASS}' already exists")

    collections = client.collections.list_all()
    print("Collections:", collections)
    if CLASS in collections:
        movies = client.collections.get(CLASS)
        total = movies.aggregate.over_all(total_count=True).total_count
        print(f"Collection '{CLASS}' has {total} objects")
    else:
        print(f"⚠️ Collection '{CLASS}' not found yet")
        
    movies = client.collections.get(CLASS)

    # Load & clean CSV
    df = pd.read_csv(CSV_PATH).drop(columns=["show_id", "date_added"], errors="ignore").fillna("")
    if "release_year" in df.columns:
        df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)

    # Embeddings (BYO vectors)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def build_text(row):
        return f"{row.get('title','')} {row.get('description','')} {row.get('cast','')} {row.get('listed_in','')}"

    inserted = 0
    with movies.batch.dynamic() as batch:
        for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Ingesting {CLASS}"):
            vec = model.encode(build_text(r), normalize_embeddings=True).tolist()
            uid = generate_uuid5(f"{r.get('title','')}::{r.get('release_year',0)}")
            props = {
                "type": r.get("type",""),
                "title": r.get("title",""),
                "director": r.get("director",""),
                "cast": r.get("cast",""),
                "country": r.get("country",""),
                "release_year": int(r.get("release_year", 0)),
                "rating": r.get("rating",""),
                "duration": r.get("duration",""),
                "listed_in": r.get("listed_in",""),
                "description": r.get("description",""),
            }
            batch.add_object(properties=props, uuid=uid, vector=vec)
            inserted += 1

    total = movies.aggregate.over_all(total_count=True).total_count
    print(f"✅ Ingested {inserted} objects. Collection total: {total}")

finally:
    client.close()
    print("Client is Closed")