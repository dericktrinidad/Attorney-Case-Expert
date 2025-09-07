# CinemaRAG 🎬

CinemaRAG is a retrieval-augmented movie recommendation system that combines hybrid search with Weaviate, XGBoost ranking, and a final LLM reasoning layer.

- Retriever → performs hybrid semantic + keyword search over a movie database (plots, genres, metadata).

- Ranker (XGBoost) → scores candidate movies using engineered features (retrieval scores, metadata signals).

- LLM layer → refines results, adds natural-language explanations, and personalizes recommendations.

- User Interface → lightweight API/UI for querying (e.g., “recommend me a sci-fi movie with time travel”).

- The project demonstrates how to integrate vector databases, ML models, and LLMs into a clean modular pipeline for professional RAG applications.

## CinemaRAG Workflow Diagram
![CinemaRAG Workflow][docs/images/workflow.svg]

The CinemaRAG workflow starts with user input, which is cleaned and embedded for retrieval. A hybrid search over the vector database surfaces the most relevant documents. These results are combined with recommendations from an XGBoost model and formatted into an augmented prompt. Finally, the LLM generates the output response, tailored to the user’s preferences.


## 🚀 Setup

Make sure you are using Python 3.10 (recommended for best compatibility). You can create a virtual environment first:

### Create Conda Setup
```Bash
conda create -n {ENV_NAME} python=3.10 -y
conda activate {ENV_NAME}
```
### Create Virtual environment
```Bash
python3.10 -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
```
### Install dependencies:
```Bash
git clone https://github.com/your-username/cinemarag.git
cd cinemarag

pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 🗂️ WorkFlow Hierarchy


```Bash
project/
├─ main.py                 # Entry point – wires together pipeline
├─ pyproject.toml          # Project metadata & dependencies
├─ README.md               # Project description & setup guide
├─ configs/                # YAML/JSON configs for retriever, models, logging
│  └─ config.yaml
├─ utils/                  # Core modules
│  ├─ common/              # Shared utilities (logging, config loaders, types)
│  ├─ ui/                  # User interface layer (FastAPI, Streamlit, etc.)
│  ├─ retriever/           # Retriever logic (Weaviate client, hybrid search)
│  ├─ features/            # Feature engineering for XGBoost
│  ├─ models/              # XGBoost + LLM wrappers
│  └─ pipelines/           # Orchestration of retriever → ranker → LLM
├─ tests/                  # Unit tests for each module
├─ data/                   # Raw & processed datasets (gitignored)
│  ├─ raw/
│  └─ processed/
└─ models_store/           # Saved ML models / embeddings (gitignored)
```