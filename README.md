# Attorney Case Expert (ACE) 🕵️‍♂️

## Problem Statement

Legal teams are inundated with vast volumes of court opinions scattered across jurisdictions and databases. Traditional workflows require attorneys and investigators to manually sift through lengthy, complex documents, slowing down decision-making and inflating research costs.

Key challenges include:

- Lengthy documents: Opinions can span hundreds of pages with critical insights buried deep within.

- Complex legal language: Statutes, citations, and procedural context demand precise interpretation.

- Inconsistent formatting: Variations across courts and jurisdictions complicate automated processing.

- Redundancy: Duplicate or overlapping opinions waste valuable time.

- Relevance filtering: Surfacing the most applicable rulings without overwhelming noise.

# Solution - Attorney Case Expert (ACE)

Attorney Case Expert (ACE) is an AI-powered legal research platform designed to accelerate precedent discovery and case analysis. ACE converts unstructured legal text into a searchable, vectorized database, enabling lawyers, investigators, and compliance teams to work more efficiently.

With ACE, users gain access to:

Natural-language legal search: Query in plain English (e.g., “What fraud cases involving wire transfers have been decided in the last five years?”).

Context-aware retrieval: Results ranked by legal context and precedent relevance, not just keywords.

LLM-driven summaries: Concise, IRAC-formatted explanations with citations, allowing rapid assessment of each case.

Scalability in design: Built with modular pipelines, batched ingestion, and retriever–generator architecture to handle large legal corpora.

Practical value for professionals: Reduces manual review time and highlights precedents that directly support legal arguments.

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

## Ingest Case Opinions

Run python script to populate weaviate vector database with case opinions

``` Bash
python main.py --ingest 'path/to/opinions.csv'
```

## Run Pipeline

``` Bash
python main.py --query 'Input Query Here'
```

## Run Example

Prompt Example
``` Bash
python main.py --query "Do U.S. courts have jurisdiction over murders committed on the high seas if the vessel has no national flag or is held by pirates, and does it matter whether the offender or victim is an American citizen?"
```

Retriever (Top K = 5)
``` Bash
>>>United States v. Holmes (score=0.4000)
>>>United States v. Holmes (score=0.3347)
>>>United States v. Holmes (score=0.2905)
>>>United States v. Flores (score=0.1729)
>>>United States v. Furlong (score=0.1532)
>>>United States v. Furlong (score=0.1532)
```

IRAC Model Summary Output
``` Bash
# Issue
Whether U.S. courts have jurisdiction over murder on the high seas when committed on a vessel with no national flag or under piratical control, and whether citizenship of the offender or victim affects jurisdiction.

# Rule
Under the Act of April 30, 1790, U.S. courts have jurisdiction over felonies such as murder or robbery committed on the high seas, regardless of vessel nationality or offender/victim citizenship. Jurisdiction is excluded only when the vessel is in fact and in right the property of a foreign state subject to its control.

# Application
In United States v. Holmes and related precedents (Klintock’s Case, Furlong), the Court held that offenses committed aboard stateless or piratical vessels fall within U.S. jurisdiction. Citizenship of the parties is immaterial. The burden of proving a foreign national character for the vessel lies with the defendant.

# Conclusion
The courts of the United States have jurisdiction over such offenses.  
The controlling case is **United States v. Holmes**.
```


