import os
import json
import argparse
import requests
import fitz
from pathlib import Path

GROBID_URL = "http://localhost:8070/api/processReferences"

# ---------- Utility ----------
def ensure_folders():
    Path("data/parsed_text").mkdir(parents=True, exist_ok=True)
    Path("data/references").mkdir(parents=True, exist_ok=True)

# ---------- Extract full text ----------
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to extract text from {pdf_path}: {e}")
        return None

# ---------- Extract references ----------
def extract_references_with_grobid(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": f}
            resp = requests.post(GROBID_URL, files=files, timeout=60)
        if resp.status_code == 200:
            return resp.text
        else:
            print(f"[WARNING] GROBID failed for {pdf_path} — Status {resp.status_code}")
            return None
    except Exception as e:
        print(f"[ERROR] GROBID request failed for {pdf_path}: {e}")
        return None

# ---------- Main processing ----------
def process_pdfs():
    ensure_folders()
    pdf_dir = Path("data/pdfs")
    results = []

    for pdf_file in pdf_dir.glob("*.pdf"):
        pdf_path = str(pdf_file)
        topic = pdf_file.stem.split("_")[0]  # e.g., "nlp" from "nlp_paper_1.pdf"
        print(f"[PROCESSING] {pdf_path}")

        # Extract full text
        full_text = extract_text_from_pdf(pdf_path)
        if full_text:
            text_path = Path("data/parsed_text") / f"{pdf_file.stem}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(full_text)
            print(f"[Saved text] {text_path}")
        else:
            text_path = None

        # Extract references
        refs_xml = extract_references_with_grobid(pdf_path)
        if refs_xml:
            refs_path = Path("data/references") / f"{pdf_file.stem}_refs.xml"
            with open(refs_path, "w", encoding="utf-8") as f:
                f.write(refs_xml)
            print(f"[Saved references] {refs_path}")
        else:
            refs_path = None

        results.append({
            "topic": topic,
            "pdf_path": pdf_path,
            "text_path": str(text_path) if text_path else None,
            "refs_path": str(refs_path) if refs_path else None
        })

    # Save summary JSON
    with open("data/pdf_processing_summary.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=4)
    print(f"[Saved summary] data/pdf_processing_summary.json")

# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2 — Parse PDFs & Extract References")
    parser.parse_args()
    process_pdfs()
