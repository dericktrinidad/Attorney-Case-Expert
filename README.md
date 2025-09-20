# Attorney Case Expert (ACE)🕵️‍♂️

## Problem Statement

Investigative departments face an overwhelming volume of case files stored across physical archives and digital silos. Manually searching and reviewing these documents is slow, labor-intensive, and often impractical, limiting investigators’ ability to quickly uncover relevant information.

Building an AI-driven Document Assistant (ADA) requires overcoming unique challenges with unstructured and sensitive case data, including:

Scanned PDFs and Images: Converting legacy documents into reliable text with OCR.

Unlabeled Records: Inferring context and metadata when files lack clear structure.

Inconsistent Formatting: Normalizing handwritten notes, forms, and irregular layouts.

Redundancy: Detecting and consolidating duplicate or near-duplicate records.

Confidentiality: Enforcing strict role-based access and data security.


## Solution – Attorney Case Expert (ACE)

ACE streamlines legal research by enabling natural-language search over a vectorized case law database. Attorneys and investigators can ask questions such as “find recent fraud cases involving wire transfers in the last five years”, and ACE retrieves the most relevant opinions, ranks them by context, and generates concise LLM-driven summaries with citations and actionable insights.

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

## Managing Docker

Build containers with docker-compose.yaml
```Bash & Start
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