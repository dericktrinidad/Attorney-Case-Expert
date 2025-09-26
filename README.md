# Attorney Case Expert (ACE) 🕵️‍♂️

## Problem Statement

Investigators and attorneys face an overwhelming volume of court opinions spread across jurisdictions and databases. Manually searching and reviewing these opinions is slow, labor-intensive, and often impractical, limiting the ability to quickly uncover relevant precedents.

Building an AI-driven Case Opinion Assistant (COA) requires overcoming unique challenges with unstructured legal text, including:

- Lengthy documents: Parsing hundreds of pages per opinion.

- Complex legal language: Interpreting statutes, citations, and procedural context.

- Inconsistent formatting: Handling variations across courts and jurisdictions.

- Redundancy: Identifying duplicate or overlapping opinions across sources.

- Relevance: Surfacing the most applicable rulings without overwhelming noise.


## Solution – Attorney Case Expert (ACE)

ACE streamlines legal research by enabling natural-language search over a vectorized database of case opinions. Attorneys and investigators can ask questions such as “find recent fraud cases involving wire transfers in the last five years”, and ACE retrieves the most relevant opinions, ranks them by legal context, and generates concise LLM-driven summaries with citations. This empowers legal professionals to identify precedents faster, reduce manual review time, and focus on building stronger arguments.

## 🚀 Setup
Recommended Version == Python 3.11

### Create Conda Setup
```Bash
conda create -n {ENV_NAME} python=3.11 -y
conda activate {ENV_NAME}
```

### Install dependencies:
```Bash
git clone https://github.com/dericktrinidad/Attorney-Case-Expert.git
```
##### Pip Install Pytorch
```Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

##### Pip Install CinemaRAG dependencies
```Bash
pip install -r requirements.txt
```

## Fine-Tuning LLMs

```Bash
python ./finetuning/train_ace_irac_lora.py \
  --model_id Qwen/Qwen2.5-7B-Instruct \
  --data_path ./data/raw/ace_irac_sft.jsonl \
  --output_dir ./finetuning/ace-irac-lora-qwen7b \
  --epochs 2 --batch_size 1 --grad_accum 16 --lr 2e-4 \
  --qlora_r 8 --use_qlora
```

## Managing Docker

Build containers with docker-compose.yaml
```Bash
sudo docker-compose -f docker/docker-compose.yml up --build -d
```
Stop & Remove Containers
```Bash
sudo docker-compose -f docker/docker-compose.yml down
```
View Logs
```Bash
sudo docker-compose -f docker/docker-compose.yml logs -f
```