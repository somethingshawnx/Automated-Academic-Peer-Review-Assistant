import os
import json
import argparse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

FAISS_DIR = "data/faiss_indexes"
PDF_DIR = "data/pdfs"
METADATA_PATH = "data/metadata.json"

def extract_text_from_pdf(pdf_path, max_chars=2000):
    """Extract text from first few pages of a PDF."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[:5]:
            text += page.extract_text() or ""
        return text[:max_chars]
    except Exception as e:
        return f"ERROR reading {pdf_path}: {e}"

def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

def build_faiss_index(pdf_dir, index_path, mapping_path, metadata_path=None):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    # Load metadata if available
    metadata_dict = {}
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_meta = json.load(f)
            if isinstance(raw_meta, list):  # list of papers
                for paper in raw_meta:
                    pdf_path = paper.get("pdf_path")
                    if pdf_path:  # only include if valid
                        metadata_dict[os.path.basename(pdf_path)] = paper
            elif isinstance(raw_meta, dict):  # single paper
                pdf_path = raw_meta.get("pdf_path")
                if pdf_path:
                    metadata_dict[os.path.basename(pdf_path)] = raw_meta

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    vectors, mapping = [], {}
    idx = 0

    for fname in os.listdir(pdf_dir):
        if not fname.endswith(".pdf"):
            continue
        pdf_path = os.path.join(pdf_dir, fname)
        text_excerpt = extract_text_from_pdf(pdf_path)

        # Embed
        vec = model.encode([text_excerpt], convert_to_numpy=True)
        vec = normalize(vec)
        vectors.append(vec)

        # Attach metadata if available
        meta = metadata_dict.get(fname, {})
        mapping[idx] = {
            "pdf_path": pdf_path,
            "text_excerpt": text_excerpt,
            "title": meta.get("title"),
            "abstract": meta.get("abstract"),
            "link": meta.get("link"),
            "published": meta.get("published"),
        }
        idx += 1

    if not vectors:
        raise ValueError("No PDFs found for indexing!")

    vectors = np.vstack(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(vectors)

    faiss.write_index(index, index_path)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"FAISS index built: {index_path}")
    print(f"Mapping saved: {mapping_path}")
    print(f"Indexed {len(vectors)} PDFs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index for research papers")
    parser.add_argument("--pdf_dir", type=str, default=PDF_DIR, help="Folder containing PDFs")
    parser.add_argument("--index_path", type=str, default=os.path.join(FAISS_DIR, "global_index.bin"))
    parser.add_argument("--mapping_path", type=str, default=os.path.join(FAISS_DIR, "global_mapping.json"))
    parser.add_argument("--metadata_path", type=str, default=METADATA_PATH, help="Optional metadata.json")

    args = parser.parse_args()
    build_faiss_index(args.pdf_dir, args.index_path, args.mapping_path, args.metadata_path)
