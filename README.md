# **Agentic-Academic-Peer-Review-Assistant**

An AI-powered system for automated academic peer review.

This tool analyzes research papers to provide structured, reviewer-style feedback. It performs citation quality checks, novelty search, plagiarism detection, factual consistency analysis, and claim mapping, then synthesizes the results into a professional review report using an **agentic multi-step pipeline**. Designed for researchers, educators, and institutions to accelerate the peer review process.

---

## Key Features

* **Automated Analysis Suite**

  * **Novelty Search:** Retrieve and compare papers using FAISS + semantic embeddings.
  * **Plagiarism Detection:** Detect exact and paraphrase overlaps via semantic similarity + string matching.
  * **Factual Checks:** Validate numerical values and units for consistency and plausibility.
  * **Claim Mapping:** Extract and match scientific claims against prior publications.
  * **Citation Alert (via GROBID):** Parse references, check citation quality, and flag missing/incorrect citations.

* **Agentic Review Pipeline**

  * **Planner Agent:** Determines which analytical tools should run for the paper.
  * **Reviewer Agent:** Evaluates the paper using tool outputs and retrieved literature.
  * **Critic Agent:** Reviews the draft and identifies weak reasoning or missing evidence.
  * **Meta Reviewer:** Produces the final structured peer review.

* **Enhanced Retrieval Features**

  * **RAG-based Literature Retrieval:** Uses FAISS + SentenceTransformers embeddings to retrieve relevant research papers for context.
  * **Deep Search Mode:** On demand, fetch up to *N* new papers (via ArXiv, Semantic Scholar, CrossRef), store locally, and rebuild FAISS index for fresh comparisons.

* **LLM-Powered Review Synthesis**

  * **Structured Review Generation:** Summarize findings into section-wise feedback including strengths, weaknesses, suggestions, and final recommendation.

* **User-Friendly Interface**

  * **Web App:** Upload a PDF, optionally enable deep search, and receive a detailed review report.
  * **Exportable Results:** Outputs JSON artifacts (novelty, plagiarism, claim mapping, factual checks, citations) and a consolidated review report.

---

## Technology Stack

* **Core Libraries:** `PyPDF2`, `requests`, `argparse`, `json`, `re`
* **NLP & Embeddings:** `sentence_transformers` (`all-MiniLM-L6-v2`), `faiss`, `scikit-learn`
* **Agent Framework:** `LangGraph`
* **Citation Parsing & Alerts:** `grobid`
* **Claim & Factual Analysis:** `pint`, regex-based extraction
* **Backend:** `Flask`
* **Frontend:** `React`
* **LLM Integration:** `google.generativeai` (Gemini), `groq`, Hugging Face Inference API

---

## Website Overview

<img width="1920" height="1080" alt="agenticai1" src="https://github.com/user-attachments/assets/1e039f93-c44f-4774-b21a-8470cb319f82" />
<img width="1920" height="1080" alt="agenticai2" src="https://github.com/user-attachments/assets/7d911f49-a1bd-422f-b421-a9567a9e9657" />
<img width="1848" height="1080" alt="agenticai3" src="https://github.com/user-attachments/assets/3797911c-205a-440a-a4b6-d7a001e6b0af" />



---

## Quick Start

1. **Clone the repository and install dependencies**

```bash
git clone https://github.com/somethingshawnx/Automated-Academic-Peer-Review-Assistant.git
cd Agentic-Academic-Peer-Review-Assistant
pip install -r requirements.txt
```

2. **Set up environment variables (for LLM integration)**

Create a `.env` file in the project root with:

```bash
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
HF_API_KEY=your_key_here
```

*Note: These are optional but required for LLM-based review synthesis.*

3. **Run PDF parsing (extract text + citations using GROBID)**

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.2
```

(Keep GROBID running in a separate terminal.)

```bash
python utils/pdf_parse.py
```

4. **Build FAISS index for similarity search**

```bash
python utils/faiss_index.py \
    --pdf_dir data/pdfs \
    --index_path data/faiss_indexes/global_index.bin \
    --mapping_path data/faiss_indexes/global_mapping.json \
    --metadata_path data/metadata.json
```

5. **Run the application**

```bash
python app.py
```

6. **Run the react app**

Open:

```bash
npm run dev
```

---

## High-Level Architecture
![peerarch](https://github.com/user-attachments/assets/540535ce-deaa-436e-ae4d-65e8b221a161)

---

## Deep Search Flow

If the user enables **Deep Search**:

1. Fetch up to *N* new papers (ArXiv, Semantic Scholar, CrossRef).
2. Save PDFs + metadata locally.
3. Rebuild FAISS index with new data.
4. Run the pipeline again with updated literature corpus.

---

## Roadmap & Future Work

* **Scalability:** Containerize with Docker and add background workers for large-scale reviews.
* **Improved Claim Extraction:** Use advanced NLP/LLM models for precise claim detection.
* **Richer Novelty Detection:** Combine dense embeddings (FAISS) with sparse retrieval (BM25) for hybrid search.
