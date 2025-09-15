import os
import json
import argparse
import faiss
import numpy as np
from PyPDF2 import PdfReader
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

with open("data/metadata.json", "r", encoding="utf-8") as f:
    raw_meta = json.load(f)

METADATA = {}
if isinstance(raw_meta, list):
    for entry in raw_meta:
        pdf_path = entry.get("pdf_path")
        if not pdf_path:   # skip None or missing
            continue
        fname = os.path.basename(pdf_path)
        METADATA[fname] = entry
else:
    METADATA = raw_meta

FAISS_INDEX = "data/faiss_indexes/global_index.bin"
FAISS_MAPPING = "data/faiss_indexes/global_mapping.json"

# ---------- Utils ----------
def extract_text_from_pdf(pdf_path, max_chars=20000):
    """Extract text from a PDF."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
    return text[:max_chars].strip()

def split_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks (words)."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

def calculate_exact_overlap(chunk, ref_chunk, threshold=0.85):
    """Exact string overlap."""
    score = SequenceMatcher(None, chunk, ref_chunk).ratio()
    return score if score >= threshold else None

# ---------- Plagiarism Check ----------
def run_plagiarism_check(test_pdf, output_file, top_k=5):
    print(f"[INFO] Extracting text from: {test_pdf}")
    test_text = extract_text_from_pdf(test_pdf)
    test_chunks = split_into_chunks(test_text)

    # Load FAISS index + mapping
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_MAPPING):
        raise FileNotFoundError(" No global FAISS index found. Please run faiss_index.py first.")

    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_MAPPING, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if isinstance(mapping, list):
        mapping = {str(i): entry for i, entry in enumerate(mapping)}

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode test chunks
    print(f"[INFO] Encoding {len(test_chunks)} test chunks...")
    test_embeddings = model.encode(test_chunks, convert_to_numpy=True, show_progress_bar=True)
    test_embeddings = normalize(test_embeddings)

    exact_matches, paraphrase_matches = [], []

    # Search each chunk in FAISS
    print("[INFO] Running FAISS search for paraphrase overlap...")
    D, I = index.search(test_embeddings, top_k)

    for chunk_idx, chunk in enumerate(test_chunks):
        for sim, ref_idx in zip(D[chunk_idx], I[chunk_idx]):
            if ref_idx == -1:
                continue
            ref_entry = mapping[str(ref_idx)]
            score = float(sim)
            if score >= 0.70:  # semantic threshold
                
                fname = os.path.basename(ref_entry["pdf_path"])
                meta_entry = METADATA.get(fname, {})
                paraphrase_matches.append({
                    "chunk": chunk,
                    "score": score,
                    "pdf_name":  fname,
                    "link": meta_entry.get("link", None),
                    "type": "paraphrase_overlap"
                })

    # Optional: exact overlap
    print("[INFO] Checking for exact overlaps...")
    for ref_idx, ref_entry in mapping.items():
        if not ref_entry.get("text_path") or not os.path.exists(ref_entry["text_path"]):
            continue
        with open(ref_entry["text_path"], "r", encoding="utf-8") as f:
            ref_text = f.read()
        ref_chunks = split_into_chunks(ref_text)

        for chunk in test_chunks:
            for r_chunk in ref_chunks:
                score_exact = calculate_exact_overlap(chunk, r_chunk)
                if score_exact:
                    
                    fname = os.path.basename(ref_entry["pdf_path"])
                    meta_entry = METADATA.get(fname, {})
                    exact_matches.append({
                        "chunk": chunk,
                        "score": score_exact,
                        "pdf_name": fname,
                        "link": meta_entry.get("link", None),
                        "type": "exact_overlap"
                    })

    # Summary
    result = {
        "paper": test_pdf,
        "exact_overlap": exact_matches,
        "paraphrase_overlap": paraphrase_matches,
        "summary": {
            "exact_overlap_count": len(exact_matches),
            "paraphrase_overlap_count": len(paraphrase_matches),
            "plagiarism_risk": "HIGH" if len(paraphrase_matches) + len(exact_matches) > 20 else "LOW"
        }
    }

    # Save JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plagiarism Check (using FAISS global index)")
    parser.add_argument("--test-pdf", type=str, required=True, help="Path to input PDF")
    parser.add_argument("--output", type=str, required=True, help="Path to save JSON results")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top matches to retrieve")
    args = parser.parse_args()

    run_plagiarism_check(args.test_pdf, args.output, args.top_k)
