import argparse
import torch
from utils.retriever import WeaviateRetriever
from utils.telemetry import init_tracing
from utils.pipelines.ragservice import RAGService
from utils.pipelines.vectorize_batched_opinions import VectorizeOpinions 
from utils.models.llm.hf_infer import HFModelManager, HFLoadConfig, GenerateConfig

torch.cuda.empty_cache()

def build_service():
    #Initialize Arize-Pheonix http://127.0.0.1:6006
    init_tracing(service_name="ace-app-1")
    retriever = WeaviateRetriever()
    cfg = GenerateConfig()
    model_kwargs = {
        'base_model_id': "Qwen/Qwen2.5-7B-Instruct",
        'irac_model_id': "./finetuning/models/ace-irac-lora-qwen7b"
    }
    llm_cfg = HFLoadConfig(**model_kwargs)
    return RAGService(cfg=cfg, llm_cfg=llm_cfg, retriever=retriever)

def ingest_data(path: str):
    print(f"[INFO] Ingesting data from {path}")
    # retriever = WeaviateRetriever()
    vectorizer = VectorizeOpinions()
    vectorizer.ingest(path)

def run_query(query: str):
    svc = build_service()
    out = svc.run_pipeline(query)

def build_parser():
    parser = argparse.ArgumentParser(description="Ace RAG Service CLI")
    parser.add_argument("--ingest",
        type=str,
        help="Path to CSV file with raw opinions to ingest."
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to query against the RAG pipeline."
    )
    return parser, parser.parse_args()

def main():
    parser, args = build_parser()
    if args.ingest:
        ingest_data(args.ingest)
    elif args.query:
        run_query(args.query)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
