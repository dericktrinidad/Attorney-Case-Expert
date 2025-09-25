import os, time, urllib.request
from tqdm import tqdm
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from numpy import float32
import re
# load_dotenv()
# TOKEN = os.getenv("COURTLISTENER_API_KEY")
# point to your env file

df_path = "./data/raw/all_opinions.csv"
df = pd.read_csv(df_path)

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")   # "weaviate" if running inside the app container
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
# CSV_PATH = os.getenv("CSV_PATH", "./data/raw/netflix/netflix_titles.csv")
CLASS = os.getenv("WEAVIATE_CLASS", "Cases")

print(f"CONNECTING TO CLIENT {WEAVIATE_HTTP_PORT}")
client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST, http_port=WEAVIATE_HTTP_PORT, http_secure=False,
    grpc_host=WEAVIATE_HOST, grpc_port=WEAVIATE_GRPC_PORT, grpc_secure=False,
)

print("✅ Connected to Weaviate client")


def norm_date(x):
    if pd.isna(x) or str(x).strip() == "":
        return "1970-01-01T00:00:00Z"
    try:
        dt = pd.to_datetime(x, errors="coerce", utc=True)
        if pd.isna(dt):
            return "1970-01-01T00:00:00Z"
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return "1970-01-01T00:00:00Z"
    
def _clean_txt(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\n\r\t\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
        
            
try:
    # Create collection if missing (BYO vectors => VectorConfig.none)
    if CLASS not in client.collections.list_all():
        client.collections.create(
            name=CLASS,
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                # Property(name="court", data_type=DataType.TEXT),
                Property(name="date_filed", data_type=DataType.DATE),
                Property(name="url", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
            ],
            # ✅ correct in v4:
            vector_config=Configure.Vectors.self_provided(),
        )
        print(f"Collection '{CLASS}' created")
    else:
        print(f"Collection '{CLASS}' already exists")


        
    cases = client.collections.get(CLASS)

    # Embeddings (BYO vectors)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # N_DATA = 1000
    class_name = "LegalOpinion"
    with cases.batch.dynamic() as batch:
        # batch.batch_size = 32
        # for i, doc in tqdm(df.iloc[:N_DATA].iterrows(), total=N_DATA, desc="Ingesting to Weaviate"):
        for i, doc in tqdm(df.iterrows(), total=len(df), desc="Ingesting to Weaviate"):
        
            uid = generate_uuid5(f"{doc.get('title','')}::{doc.get('date_filed',0)}::{doc.get('id',0)}::{doc.get('absolute_url','')}::{i}")
            vec = model.encode(doc["text"]).astype(float32)
            prop = {
                "title": doc["case_name"],
                # "court": doc["court"] or "",
                "date_filed": norm_date(doc["date_filed"]) or "1970-01-01T00:00:00Z",
                "url": doc["absolute_url"],
                "text": _clean_txt(doc["text"]),
                }
            
            batch.add_object(
                properties= prop,
                vector=vec,
                uuid=uid,
            )
    collections = client.collections.list_all()
    # print("Collections:", collections)
    if CLASS in collections:
        class_obj = client.collections.get(CLASS)
        total = class_obj.aggregate.over_all(total_count=True).total_count
        print(f"Collection '{CLASS}' has {total} objects")
    else:
        print(f"⚠️ Collection '{CLASS}' not found yet")
finally:
    client.close()
    print("Client is Closed")