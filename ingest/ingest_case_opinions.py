import os, time, urllib.request
from tqdm import tqdm
# import pandas as pd
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
load_dotenv("/home/cheemyubuntu/Documents/Projects/LLMS/Attorney-Case-Expert/courtlistener.env")

# now TOKEN is available
TOKEN = os.getenv("TOKEN")
H = {"Authorization": f"token {TOKEN}"}
print("TOKEN: ", H)
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

def _clean_txt(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\n\r\t\f\v]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def stream_opinions(limit=100_000, start_url="https://www.courtlistener.com/api/rest/v4/opinions/??jurisdiction=ny&order_by=-date_filed"):
    url, seen = start_url, 0
    # with tqdm(total=limit, desc="Fetching Court Opinions") as pbar:
    while url and seen < limit:
        try:
            r = requests.get(url, headers=H, timeout=60); r.raise_for_status()
            j = r.json()
            # url = j.get("next")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for UTL {url}")
            # url = None
            continue
        for op in j["results"]:
            text = op.get("plain_text") #or op.get("html") or op.get("html_with_citations")
            title = op.get('absolute_url').split('/')[-2]
            if not text:
                continue
            text = _clean_txt(text)
            court = op.get("court")
            if isinstance(court, dict):
                court = court.get("name") or court.get("id") or ""
            yield {
                "id": str(op["id"]),
                "title": title,
                "court": op.get("court"),
                "date_filed": op.get("date_filed"),
                "url": "https://www.courtlistener.com" + (op.get("absolute_url") or ""),
                "text": text,
            }
            seen += 1
            if seen >= limit: 
                break
            
        url = j.get("next")
        if not url:
            break
        time.sleep(0.25)
        
            
try:
    # Create collection if missing (BYO vectors => VectorConfig.none)
    if CLASS not in client.collections.list_all():
        client.collections.create(
            name=CLASS,
            properties=[
                Property(name="title", data_type=DataType.TEXT),
                Property(name="court", data_type=DataType.TEXT),
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
    
    class_name = "LegalOpinion"
    N_CASES = 1000
    with cases.batch.dynamic() as batch:
        # batch.batch_size = 32
        for doc in tqdm(stream_opinions(limit=N_CASES), total=N_CASES, desc="Ingesting to Weaviate"):
            uid = generate_uuid5(f"{doc.get('title','')}::{doc.get('date_filed',0)}::{doc.get('id',0)}")
            
            vec = model.encode(doc["text"]).astype(float32)
            
            prop = {
                "title": doc["title"],
                "court": doc["court"] or "",
                "date_filed": doc["date_filed"] or "1970-01-01T00:00:00Z",
                "url": doc["url"],
                "text": doc["text"],
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