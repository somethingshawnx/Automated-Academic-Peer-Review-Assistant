import os
import json
import re
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader  

# ---------------- CONFIG ----------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PARSED_TEXT_DIR = "data/parsed_text"
PAPERS_JSON = "data/metadata.json"
DEFAULT_CLAIM_SIM_THRESHOLD = 0.70
MIN_SENT_LEN = 30
CLAIM_KEYWORDS = [
    "we propose", "we present", "this paper", "our contribution",
    "we show", "we demonstrate", "we introduce", "in this work",
    "we report", "we observe", "we develop", "we design"
]
# ----------------------------------------

def extract_text_from_pdf_fn(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"[WARN] Failed to parse PDF {pdf_path}: {e}")
    return text.strip() if text else None

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text)
    sents = re.split(r'(?<=[\.\?\!])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) >= MIN_SENT_LEN]

def extract_claims_by_keywords(sentences):
    claims = []
    for s in sentences:
        if any(kw in s.lower() for kw in CLAIM_KEYWORDS):
            claims.append(s)
    return claims

def extract_text_from_paper_meta(paper_meta, idx):
    pdf_path = paper_meta.get("pdf_path") or paper_meta.get("pdf")
    if pdf_path and os.path.exists(pdf_path):
        try:
            return extract_text_from_pdf_fn(pdf_path)
        except Exception:
            pass

    txt_path = paper_meta.get("txt_path") or paper_meta.get("text_path")
    if txt_path and os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    fallback = os.path.join(PARSED_TEXT_DIR, f"paper_{idx+1}.txt")
    if os.path.exists(fallback):
        with open(fallback, "r", encoding="utf-8") as f:
            return f.read()
    return None

def gather_existing_claims(similar_list, papers_metadata):
    existing_claims = []
    for entry in similar_list:
        meta = None

        if isinstance(entry, dict):
            if "index" in entry:
                idx = int(entry.get("index"))
                if 0 <= idx < len(papers_metadata):
                    meta = papers_metadata[idx]
            else:
                meta = {
                    "title": entry.get("title", ""),
                    "pdf_path": entry.get("pdf_path"),
                    "txt_path": entry.get("text_path")
                }

        if not meta:
            continue

        text = extract_text_from_paper_meta(meta, 0)
        if not text:
            continue

        sents = split_into_sentences(text)
        claims = extract_claims_by_keywords(sents)
        if not claims:
            claims = sorted(sents, key=len, reverse=True)[:3]

        for c in claims:
            existing_claims.append({
                "paper_title": meta.get("title", ""),
                "claim": c,
                "link": meta.get("link", "")   
            })

    return existing_claims

def extract_new_claims_from_new_pdf(new_pdf_path):
    text = None
    if os.path.exists(new_pdf_path):
        try:
            text = extract_text_from_pdf_fn(new_pdf_path)
        except Exception:
            text = None

    if not text:
        basename = os.path.splitext(os.path.basename(new_pdf_path))[0]
        alt = os.path.join(PARSED_TEXT_DIR, f"{basename}.txt")
        if os.path.exists(alt):
            with open(alt, "r", encoding="utf-8") as f:
                text = f.read()

    if not text:
        raise RuntimeError("Could not extract text for the new PDF.")

    sents = split_into_sentences(text)
    new_claims = extract_claims_by_keywords(sents)
    if not new_claims:
        new_claims = sorted(sents, key=len, reverse=True)[:5]
    return new_claims, text

def embed_texts(model, texts):
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

def map_claims(new_claims, existing_claims, model, claim_threshold=DEFAULT_CLAIM_SIM_THRESHOLD):
    if not new_claims:
        return []

    if not existing_claims:
        return [{"claim": c, "is_novel": True,
                 "matched_claim": None, "matched_paper": None,
                 "similarity": 0.0} for c in new_claims]

    new_emb = embed_texts(model, new_claims)
    existing_texts = [c["claim"] for c in existing_claims]
    existing_emb = embed_texts(model, existing_texts)

    sim_matrix = cosine_similarity(new_emb, existing_emb)
    mappings = []
    for i, c in enumerate(new_claims):
        best_j = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_j])
        best_match = existing_claims[best_j]
        is_novel = best_score < claim_threshold
        mappings.append({
            "claim": c,
            "is_novel": bool(is_novel),
            "matched_claim": best_match["claim"],
            "matched_paper_title": best_match.get("paper_title", ""),
            "matched_paper_link": best_match.get("link", ""),  
            "similarity": round(best_score, 4)
        })
    return mappings

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Step 5 — Claim Extraction & Mapping")
    parser.add_argument("--new_pdf", required=True, help="Path to the new PDF")
    parser.add_argument("--similar_json", required=True, help="Path to novelty.json (similar papers)")
    parser.add_argument("--claim_threshold", type=float, default=DEFAULT_CLAIM_SIM_THRESHOLD, help="Similarity threshold")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="SentenceTransformer model")
    parser.add_argument("--out_dir", required=False, help="Output directory for results")   # NEW
    args = parser.parse_args()

    similar_data = load_json(args.similar_json)
    similar_list = similar_data.get("results", similar_data)

    papers_meta = load_json(PAPERS_JSON)

    print("[STEP5] Extracting new paper claims...")
    new_claims, _ = extract_new_claims_from_new_pdf(args.new_pdf)
    print(f"[STEP5] Found {len(new_claims)} candidate claims in the new paper.")

    print("[STEP5] Gathering claims from similar papers...")
    existing_claims = gather_existing_claims(similar_list, papers_meta)
    print(f"[STEP5] Collected {len(existing_claims)} claims from {len(similar_list)} similar papers.")

    print(f"[STEP5] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    print("[STEP5] Mapping claims...")
    mappings = map_claims(new_claims, existing_claims, model, claim_threshold=args.claim_threshold)

    # Use --out_dir if provided, else fallback to pdf name
    if args.out_dir:
        out_dir = args.out_dir
    else:
        base = os.path.splitext(os.path.basename(args.new_pdf))[0]
        out_dir = os.path.join("data/results", base)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "claim_mapping.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "new_pdf": args.new_pdf,
            "mappings": mappings,
            "num_new_claims": len(new_claims),
            "num_existing_claims": len(existing_claims)
        }, f, indent=2, ensure_ascii=False)

    print(f"[STEP5] Saved claim mapping to: {out_path}")

    # Print summary
    for m in mappings:
        print("\nClaim:", m["claim"])
        print("Similarity:", m["similarity"], "| Novel:", m["is_novel"])
        print("Matched paper:", m["matched_paper_title"], "| Matched claim:", m["matched_claim"])

if __name__ == "__main__":
    main()
