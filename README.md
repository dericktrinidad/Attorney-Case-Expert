# CinemaRAG 🎬

CinemaRAG is a retrieval-augmented movie recommendation system that combines hybrid search with Weaviate, XGBoost ranking, and a final LLM reasoning layer.

- Retriever → performs hybrid semantic + keyword search over a movie database (plots, genres, metadata).

- Ranker (XGBoost) → scores candidate movies using engineered features (retrieval scores, metadata signals).

- LLM layer → refines results, adds natural-language explanations, and personalizes recommendations.

- User Interface → lightweight API/UI for querying (e.g., “recommend me a sci-fi movie with time travel”).

- The project demonstrates how to integrate vector databases, ML models, and LLMs into a clean modular pipeline for professional RAG applications.

### 🚀 Setup

Make sure you are using Python 3.10 (recommended for best compatibility). You can create a virtual environment first:

#### Create Conda Setup
```Bash
conda create -n {ENV_NAME} python=3.10 -y
conda activate {ENV_NAME}
```
#### Create Virtual environment
```Bash
python3.10 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```
#### Install dependencies:
```Bash
git clone https://github.com/your-username/cinemarag.git
cd cinemarag

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 🗂️ WorkFlow Hierarchy


```Bash
project/
├─ main.py
├─ pyproject.toml
├─ README.md
├─ .env
├─ configs/
│  └─ config.yaml
├─ utils/
│  ├─ __init__.py
│  ├─ common/
│  │  ├─ logging.py
│  │  ├─ types.py
│  │  └─ config.py
│  ├─ ui/
│  │  ├─ __init__.py
│  │  └─ api.py                # FastAPI or Streamlit entry (optional)
│  ├─ retriever/
│  │  ├─ __init__.py
│  │  ├─ base.py               # interfaces/protocols
│  │  └─ weaviate_retriever.py
│  ├─ features/
│  │  ├─ __init__.py
│  │  └─ featurizer.py         # convert retrieval → features for XGBoost
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ xgboost_model.py      # train/load/predict
│  │  └─ llm.py                # final LLM post-processing
│  └─ pipelines/
│     ├─ __init__.py
│     └─ recommend.py          # orchestrates retriever → XGB → LLM
├─ tests/
│  ├─ test_retriever.py
│  ├─ test_featurizer.py
│  ├─ test_xgb.py
│  └─ test_pipeline.py
├─ data/                       # gitignored
│  ├─ raw/
│  └─ processed/
└─ models_store/               # saved XGB, tokenizers, etc. (gitignored)
```