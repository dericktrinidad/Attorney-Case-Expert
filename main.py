#TODO: from utils.common import load_config, setup_logging
from utils.retriever import WeaviateRetriever
#TODO: from utils.models import XGBRecommender, FinalLLM
#TODO: from utils.features import to_xgb_features
#TODO: from utils.pipelines import RecommendPipeline

def main():
    query = "space comedy adventure"
    wr = WeaviateRetriever()
    out = wr.retrieve(query)
    print(out)

    wr.close()

if __name__ == "__main__":
    main()