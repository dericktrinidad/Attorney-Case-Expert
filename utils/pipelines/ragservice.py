from typing import Dict, Any, List
# from utils.telemetry.decorators import instrument_llm, instrument_retriever
from utils.telemetry import instrument_retriever, instrument_llm
from utils.models.llm.hf_infer import HFModelManager, HFLoadConfig, GenerateConfig
from utils.retriever.weaviate_retriever import WeaviateRetriever
from utils.pipelines.prompt_engineering import initial_prompt, summarize_opinion_prompt, summarize_irac_prompt
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"


class RAGService:
    def __init__(self, llm_cfg:HFLoadConfig, cfg:GenerateConfig, retriever: WeaviateRetriever):
        self.llm = HFModelManager(llm_cfg)
        self.cfg = cfg
        self.retriever = retriever
        
    @instrument_retriever(
        name="hybrid_search",
        input_getter=lambda query, **_: query,
        output_getter=lambda out: out,  
    )
    def hybrid_search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_hybrid(query, top_k=k)
    
    @instrument_retriever(
        name="keyword_search",
        input_getter=lambda query, **_: query,
        output_getter=lambda out: out,  
    )
    def keyword_search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_bm25(query, top_k=k)
    
    @instrument_retriever(
        name="semantic_search",
        input_getter=lambda query, **_: query,
        output_getter=lambda out: out,  
    )
    def keyword_search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_semantic(query, top_k=k)
    
    @instrument_llm(
        name="generate_text",
        input_getter=lambda query, **_: query,
        output_getter=lambda out: out,       # record the return value
    )
    def llm_inference(self, prompt: str) -> str:
        out = self.llm.make_inference(prompt, GenerateConfig())
        return out
    @instrument_llm(
        name="generate_irac_summary",
        input_getter=lambda query, **_: query,
        output_getter=lambda out: out,       # record the return value
    )
    def llm_irac_inference(self, prompt: str) -> str:
        out = self.llm.make_irac_inference(prompt, GenerateConfig())
        return out

    def refine_output(self, out):
        return out.split("[END USER PROMPT]")[-1]

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        prompt = initial_prompt(query)
        simplified_prompt = self.llm_inference(prompt)
        cleaned_simplified_prompt = self.refine_output(simplified_prompt)
        
        hits = self.hybrid_search(cleaned_simplified_prompt, k=10)
        top_docs = {doc.get('title'): [] for doc in hits[:3] } #get highest score docs
        for doc in hits:
            title = doc.get('title')
            text = doc.get('text')
            score = doc.get('_score')
            print(f"Top K Docs:\nTitle = {title}\nScore = {score}\nText = {text[:100]}")
            if title in set(top_docs.keys()):
                top_docs[title].append(text)

        best_doc = list(top_docs.items())[0]
        best_doc_text = ' '.join(best_doc[1])
        best_doc_info = f"Title:{best_doc[0]} Text: {best_doc_text}"
        prompt = summarize_irac_prompt(best_doc_info,  query)
        out = self.llm_irac_inference(prompt)
        print("Final Prompt Output:\n")
        refined_final_output = self.refine_output(out) 
        print(refined_final_output)
        # best_doc_str = f"Title: {best_doc.get('title')} Opinion:{best_doc.get('text')}"
        # opinion_prompt = summarize_opinion_prompt(best_doc_str, query)
                
        # return {"answer": opinion_prompt, "source": best_doc_str}
    
