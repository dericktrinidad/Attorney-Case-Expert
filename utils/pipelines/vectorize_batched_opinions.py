import os, time, urllib.request
from tqdm import tqdm
from uuid import uuid5, NAMESPACE_URL
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
from typing import List, Dict, Any
# load_dotenv()
# TOKEN = os.getenv("COURTLISTENER_API_KEY")
# point to your env file

df_path = "./data/raw/all_opinions.csv"
df = pd.read_csv(df_path)

WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")   # "weaviate" if running inside the app container
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
# CSV_PATH = os.getenv("CSV_PATH", "./data/raw/netflix/netflix_titles.csv")



class VectorizeOpinions:
    def __init__(self):
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
        
    def stable_doc_id(self, doc: Dict[str, Any]) -> str:
        base = f"{doc.get('title','')}::{doc.get('date_filed',0)}::{doc.get('id',0)}::{doc.get('absolute_url','')}"
        return str(uuid5(NAMESPACE_URL, base))
    def stable_chunk_uuid(self, doc_id: str, chunk_index: int) -> str:
        return str(uuid5(NAMESPACE_URL, f"{doc_id}::chunk::{chunk_index}"))

    def ingest(self, opinions_path: str, index: str='Cases'):
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
            
            opinions_df = pd.read_csv(opinions_path)
            with cases.batch.dynamic() as batch:
                with tqdm(total=len(opinions_df), desc="Ingesting to Weaviate") as pbar:
                    for i, doc in opinions_df.iterrows():
                        full_text = self._clean_txt(doc["text"])
                        title =  doc.get("case_name", "")
                        
                        if not full_text:
                            pbar.update(1)
                            continue
                        
                        doc_id = self.stable_doc_id(doc)
                        chunks = self.chunk_text(full_text, max_words=300, overlap=40)
                        n = len(chunks)
                        for ci, chunk in enumerate(chunks):
                            # tqdm.write(f"[{i}] {title} (chunk {ci+1}/{n})")
                            
                            vec = self.tok.encode(chunk).astype(float32)
                            uid = self.stable_chunk_uuid(doc_id, ci)
                            props = {
                                "doc_id": doc_id,
                                "chunk_index": ci,
                                "chunk_count": n,
                                "title": title,
                                "date_filed": self.norm_date(doc.get("date_filed")) or "1970-01-01T00:00:00Z",
                                "url": doc.get("absolute_url") or doc.get("url") or "",
                                "text": chunk,
                                }
                            batch.add_object(
                                properties=props,
                                vector=vec,
                                uuid=uid
                                )
                            pbar.set_postfix_str(f"{title[:40]} (chunk {ci+1}/{len(chunks)})")
                        pbar.update(1)
                        
            class_obj = self.client.collections.get(index)
            total = class_obj.aggregate.over_all(total_count=True).total_count
            print(f"Collection '{index}' has {total} objects")
        finally:
            self.client.close()
            print("Client is Closed")
            
    def chunk_text(self, text: str, max_words=300, overlap=40) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_words, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end == len(words):
                break
            start = end - overlap  # sliding window with overlap
        return chunks