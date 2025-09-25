from utils.retriever import WeaviateRetriever
from utils.telemetry import init_tracing, instrument_retriever, instrument_llm
from utils.models.llm.hf_infer import HFModelManager, HFLoadConfig, GenerateConfig

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,expandable_segments:True"

#Initialize Arize-Pheonix 
init_tracing()

@instrument_llm(
    name="test_llm",
    input_getter=lambda query, **_: query,
    output_getter=lambda out: out
)
def test_hf_model(query, model_id="Qwen/Qwen2.5-14B-Instruct"):
    mgr = HFModelManager(HFLoadConfig(model_id=model_id))
    out = mgr.make_inference(query, GenerateConfig())
    print(out)
    return out
    
@instrument_retriever(
    name='test_retriever',
    input_getter=lambda query, **_: query,
    output_getter=lambda out: out
)
def test_retriever(query):
    wr = WeaviateRetriever()
    out = wr.retrieve_hybrid(query)
    wr.close()
    return out

def main():
    query = "Is a brief seizure to check ID permissible absent reasonable suspicion?"
    prompt = f'''SYSTEM: You extract search terms for a legal vector+BM25 hybrid retriever.

    Rules:
    - Return ONLY a JSON object. No prose.
    - Include: keywords (unigrams), key_phrases (multi-word), entities (case names, statutes), negatives (stop-words to ignore), expansions (common synonyms/abbrevs).
    - Preserve exact legal phrases (e.g., “reasonable suspicion”, “stop and frisk”).
    - Use lowercase except proper names/citations.
    - No hallucinations; don’t invent case names/citations.

    USER QUERY: "{query}"

    OUTPUT JSON SCHEMA:
    {{
    "keywords": [string],
    "key_phrases": [string],
    "entities": [string],
    "expansions": [string],
    "negatives": [string]
    }}'''
    messages = [{"role": "user", "content": prompt}]
    case_keywords = test_hf_model(messages, model_id="Qwen/Qwen2.5-14B-Instruct")
    
    out = test_retriever(case_keywords)
    
    for doc in out:
        title = doc.get("title", "Untitled")
        score = doc.get("_score", None)
        print(f"{title} (score={score:.4f})" if score is not None else title)
if __name__ == "__main__":
    main()