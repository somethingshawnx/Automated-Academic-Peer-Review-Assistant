import os
import json
import argparse
import requests
import feedparser
import time
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -------- Setup --------
DATA_DIR = Path("data")
TXT_DIR = DATA_DIR / "parsed_text"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"
for d in [TXT_DIR, PDF_DIR, INDEX_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Embedding model (for FAISS index)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


# -------- Fetch from ArXiv (paginated) --------
def fetch_arxiv(keyword, max_results=300):
    base_url = "http://export.arxiv.org/api/query?"
    results = []
    per_page = 150

    for start in range(0, max_results, per_page):
        query = f"search_query=all:{keyword}&start={start}&max_results={per_page}"
        resp = requests.get(base_url + query)
        feed = feedparser.parse(resp.text)

        for entry in feed.entries:
            paper_id = entry.id.split("/abs/")[-1]
            pdf_link = f"http://arxiv.org/pdf/{paper_id}.pdf"
            results.append({
                "title": entry.title.strip(),
                "abstract": entry.summary.strip(),
                "link": entry.link,
                "published": entry.published,
                "pdf_url": pdf_link
            })

        time.sleep(1)

    return results[:max_results]


# -------- Fetch from Semantic Scholar --------
def fetch_semantic_scholar(keyword, max_results=300):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    results = []
    per_page = 100

    for offset in range(0, max_results, per_page):
        params = {
            "query": keyword,
            "offset": offset,
            "limit": per_page,
            "fields": "title,abstract,url,openAccessPdf,publicationDate"
        }
        resp = requests.get(url, params=params)
        data = resp.json()

        for paper in data.get("data", []):
            pdf_url = paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
            results.append({
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "link": paper.get("url", ""),
                "published": paper.get("publicationDate", ""),
                "pdf_url": pdf_url
            })

        time.sleep(1)

    return results[:max_results]


# -------- Fetch from CrossRef --------
def fetch_crossref(keyword, max_results=300):
    url = "https://api.crossref.org/works"
    results = []
    per_page = 100

    for offset in range(0, max_results, per_page):
        params = {"query": keyword, "rows": per_page, "offset": offset}
        resp = requests.get(url, params=params)
        items = resp.json().get("message", {}).get("items", [])

        for item in items:
            title = " ".join(item.get("title", []))
            abstract = item.get("abstract", "")
            link = item.get("URL", "")
            published = item.get("created", {}).get("date-time", "")
            pdf_url = None
            if "link" in item and len(item["link"]) > 0:
                pdf_url = item["link"][0].get("URL")

            results.append({
                "title": title,
                "abstract": abstract,
                "link": link,
                "published": published,
                "pdf_url": pdf_url
            })

        time.sleep(1)

    return results[:max_results]


# -------- Save TXT + PDF --------
def save_papers(papers, topic):
    all_metadata = []
    for idx, paper in enumerate(papers, start=1):
        txt_path = TXT_DIR / f"{topic}_paper_{idx}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Title: {paper['title']}\n")
            f.write(f"Published: {paper['published']}\n")
            f.write(f"Link: {paper['link']}\n\n")
            f.write(f"Abstract:\n{paper['abstract']}\n")

        pdf_path = None
        if paper.get("pdf_url"):
            try:
                pdf_resp = requests.get(paper["pdf_url"], timeout=15)
                if pdf_resp.status_code == 200 and "pdf" in pdf_resp.headers.get("Content-Type", "").lower():
                    pdf_path = PDF_DIR / f"{topic}_paper_{idx}.pdf"
                    with open(pdf_path, "wb") as pdf_file:
                        pdf_file.write(pdf_resp.content)
            except Exception as e:
                print(f"[PDF Failed] {paper['title']} — {e}")

        meta = paper.copy()
        meta["txt_path"] = str(txt_path)
        meta["pdf_path"] = str(pdf_path) if pdf_path else None
        all_metadata.append(meta)

    cache_path = CACHE_DIR / f"{topic}_papers.json"
    with open(cache_path, "w", encoding="utf-8") as jf:
        json.dump(all_metadata, jf, indent=4, ensure_ascii=False)

    return all_metadata


# -------- Build FAISS index --------
def build_faiss_index(papers, topic):
    texts = [p["title"] + " " + (p["abstract"] or "") for p in papers]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / f"{topic}_index.faiss"))
    print(f"[INFO] Saved FAISS index for topic '{topic}'")


# -------- Smart Fetch (with caching) --------
def smart_fetch(topic, max_papers=15):
    cache_path = CACHE_DIR / f"{topic}_papers.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= max_papers:
            print(f"[Cache Hit] Loaded {len(cached)} papers for {topic}")
            return cached[:max_papers]

    print(f"[Cache Miss] Fetching new papers for {topic}...")

    papers = []
    sources = [fetch_arxiv, fetch_semantic_scholar, fetch_crossref]

    for src in sources:
        needed = max_papers - len(papers)
        if needed <= 0:
            break
        papers += src(topic, max_results=needed)

    papers = papers[:max_papers]
    saved = save_papers(papers, topic)
    build_faiss_index(saved, topic)
    return saved




# -------- Entry point (CLI + importable) --------
def fetch_and_add_papers(topic, max_papers=15):
    """Fetch + save max_papers total from all sources."""
    papers = smart_fetch(topic, max_papers=max_papers)
    if not papers:
        print("[WARN] No papers found.")
        return []

    # ✅ no extra arg
    saved_meta = save_papers(papers, topic)

    if saved_meta:
        build_faiss_index(saved_meta, topic)

    return saved_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Fetch Papers with Cache + FAISS")
    parser.add_argument("--keyword", type=str, required=True, help="Search keyword")
    args = parser.parse_args()

    results = smart_fetch(args.keyword)
    print(f"[Done] {len(results)} papers stored for topic: {args.keyword}")