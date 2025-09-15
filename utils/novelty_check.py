import argparse
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ---------- Config ----------
FAISS_INDEX = "data/faiss_indexes/global_index.bin"
FAISS_MAPPING = "data/faiss_indexes/global_mapping.json"

# ---------- Helpers ----------
def extract_text_from_pdf(pdf_path, max_chars=2000):
    """Extract text from a PDF (first N chars)."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:5]:  # only first 5 pages for speed
            text += page.extract_text() or ""
        return text[:max_chars]
    except Exception as e:
        return f"ERROR reading {pdf_path}: {e}"

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

def label_novelty(score: float) -> str:
    """Interpret similarity score into novelty category."""
    if score >= 0.70:
        return "Not Novel (very similar)"
    elif score >= 0.50:
        return "Partially Novel (some overlap)"
    else:
        return "Highly Novel (no strong match)"

# ---------- Novelty Check ----------
def novelty_check(input_pdf, top_k=5, output_path=None):
    if not os.path.exists(FAISS_INDEX) or not os.path.exists(FAISS_MAPPING):
        raise FileNotFoundError("❌ No global FAISS index found. Please run faiss_index.py first.")

    # Load FAISS index + mapping
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_MAPPING, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Ensure mapping is a dict
    if isinstance(mapping, list):
        mapping = {str(i): entry for i, entry in enumerate(mapping)}

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Extract & encode query
    query_text = extract_text_from_pdf(input_pdf)
    query_emb = model.encode([query_text], convert_to_numpy=True)
    query_emb = normalize(query_emb)

    # Search in FAISS
    D, I = index.search(query_emb, top_k)
    sims, ids = D[0], I[0]

    results = []
    for sim, idx in zip(sims, ids):
        if idx == -1:  # no match
            continue
        entry = mapping[str(idx)]
        results.append({
            "similarity": float(sim),
            "novelty": label_novelty(float(sim)),
            "pdf_path": entry.get("pdf_path", ""),
            "text_path": entry.get("text_path", ""),
            "refs_path": entry.get("refs_path", "")
        })

    # Console output
    print("\n📊 [NOVELTY CHECK RESULTS]")
    print("------------------------------------------------------------")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {os.path.basename(r['pdf_path'])}")
        print(f"   Similarity: {r['similarity']:.4f}")
        print(f"   Novelty: {r['novelty']}")
        print("------------------------------------------------------------")

    # Save JSON
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report = {"pdf": input_pdf, "results": results}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to {output_path}")

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novelty Check (using FAISS global index)")
    parser.add_argument("input_pdf", help="Path to the research paper PDF")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar papers to show")
    parser.add_argument("--output", type=str, help="Path to save JSON results")

    args = parser.parse_args()
    novelty_check(args.input_pdf, args.top_k, args.output)
