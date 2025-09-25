from utils.retriever import WeaviateRetriever
from utils.telemetry import init_tracing
from utils.pipelines.ragservice import RAGService
from utils.models.llm.hf_infer import HFModelManager, HFLoadConfig, GenerateConfig
import torch
torch.cuda.empty_cache()

def build_service():
    #Initialize Arize-Pheonix
    init_tracing(service_name="ace-app")
    retriever = WeaviateRetriever()
    cfg = GenerateConfig()
    llm_cfg = HFLoadConfig(model_id="Qwen/Qwen2.5-7B-Instruct")
    return RAGService(cfg=cfg, llm_cfg=llm_cfg, retriever=retriever)

def main():
    svc = build_service()
    query = "What are the consiquences for theft?"
    out = svc.run_pipeline(query)
    print(out)
    
if __name__ == "__main__":
    main()