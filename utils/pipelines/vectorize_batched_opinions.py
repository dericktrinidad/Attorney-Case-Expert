import os, time, urllib.request
from tqdm import tqdm
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5
from utils.retriever.weaviate_retriever import WeaviateRetriever
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import requests
from numpy import float32
import re
from typing import List
# load_dotenv()
# TOKEN = os.getenv("COURTLISTENER_API_KEY")
# point to your env file

df_path = "./data/raw/all_opinions.csv"
df = pd.read_csv(df_path)

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")   # "weaviate" if running inside the app container
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
# CSV_PATH = os.getenv("CSV_PATH", "./data/raw/netflix/netflix_titles.csv")



class TokenizeOpinions:
    def __init__(self, retriever:WeaviateRetriever):
        self.retriever = retriever
        print(f"CONNECTING TO CLIENT {WEAVIATE_HTTP_PORT}")
        self.client = self.init_client()
        print("✅ Connected to Weaviate client")
        self.tok = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
    def init_client(self):
        return weaviate.connect_to_custom(
            http_host=WEAVIATE_HOST, http_port=WEAVIATE_HTTP_PORT, http_secure=False,
            grpc_host=WEAVIATE_HOST, grpc_port=WEAVIATE_GRPC_PORT, grpc_secure=False,
        )
        
    def norm_date(self, x):
        if pd.isna(x) or str(x).strip() == "":
            return "1970-01-01T00:00:00Z"
        try:
            dt = pd.to_datetime(x, errors="coerce", utc=True)
            if pd.isna(dt):
                return "1970-01-01T00:00:00Z"
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return "1970-01-01T00:00:00Z"
        
    def _clean_txt(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"[\n\r\t\f\v]+", " ", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()
            
    def ingest(self, top_opinions: List, index: str='batched_opinions'):
        try:
            # Create collection if missing (BYO vectors => VectorConfig.none)
            if index not in self.client.collections.list_all():
                self.client.collections.create(
                    name=index,
                    properties=[
                        Property(name="doc_id", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.NUMBER),
                        Property(name="chunk_count", data_type=DataType.NUMBER),
                        Property(name="title", data_type=DataType.TEXT),
                        Property(name="date_filed", data_type=DataType.DATE),
                        Property(name="url", data_type=DataType.TEXT),
                        Property(name="text", data_type=DataType.TEXT),
                    ],

                    vector_config=Configure.Vectors.self_provided(),
                )
                print(f"Collection '{index}' created")
            else:
                print(f"Collection '{index}' already exists")
                
            cases = self.client.collections.get(index)
            
            
            with cases.batch.dynamic() as batch:
                for i, doc in tqdm(enumerate(top_opinions), total=len(df), desc="Ingesting to Weaviate"):
                
                    uid = generate_uuid5(f"{doc.get('title','')}::{doc.get('date_filed',0)}::{doc.get('id',0)}::{doc.get('absolute_url','')}::{i}")
                    vec = self.tok.encode(doc["text"]).astype(float32)
                    prop = {
                        "title": doc["title"],
                        "date_filed": self.norm_date(doc["date_filed"]) or "1970-01-01T00:00:00Z",
                        "url": doc["absolute_url"],
                        "text": self._clean_txt(doc["text"]),
                        }
                    
                    batch.add_object(
                        properties= prop,
                        vector=vec,
                        uuid=uid,
                    )
                    
            collections = self.client.collections.list_all()
            # print("Collections:", collections)
            if index in collections:
                class_obj = self.client.collections.get(index)
                total = class_obj.aggregate.over_all(total_count=True).total_count
                print(f"Collection '{index}' has {total} objects")
            else:
                print(f"⚠️ Collection '{index}' not found yet")
        finally:
            self.client.close()
            print("Client is Closed")
            
    def chunk_text(self, text: str, max_tokens=1000, overlap=200) -> List[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(ids):
            end = min(start + max_tokens, len(ids))
            chunk_ids = ids[start:end]
            chunks.append(self.tok.decode(chunk_ids))
            if end == len(ids):
                break
            start = end - overlap  # sliding window with overlap
            if start < 0: start = 0
        return chunks