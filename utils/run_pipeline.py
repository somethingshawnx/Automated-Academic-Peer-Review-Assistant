import os
import argparse
import subprocess

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {cmd}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description="Full Peer Review Pipeline")
    parser.add_argument("--pdf_url", type=str, help="URL to download paper (arXiv/DOI)", required=False)
    parser.add_argument("--pdf_path", type=str, help="Local PDF path", required=False)
    parser.add_argument("--out_dir", type=str, default="data/results", help="Output directory")
    parser.add_argument("--topic", type=str, default="general", help="Research topic (for factual checks)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # === Step 1: Download PDF if URL provided ===
    if args.pdf_url:
        pdf_path = os.path.join(args.out_dir, "paper.pdf")
        run_cmd(f"python utils/download_pdf.py --url {args.pdf_url} --output {pdf_path}")
    elif args.pdf_path:
        pdf_path = args.pdf_path
    else:
        print("Error: Provide either --pdf_url or --pdf_path")
        exit(1)

    # === Step 2: Citation Analysis ===
    citation_out = os.path.join(args.out_dir, "citation_report.json")
    run_cmd(
        f"python utils/grobid_citation_alerts.py {pdf_path} --output {citation_out}"
    )

    # === Step 3: Novelty Check (FAISS global index) ===
    novelty_out = os.path.join(args.out_dir, "novelty.json")
    run_cmd(
        f"python utils/novelty_check.py {pdf_path} "
        f"--top_k 5 "
        f"--output {novelty_out}"
    )

    # === Step 4: Plagiarism Check ===
    plagiarism_out = os.path.join(args.out_dir, "plagiarism.json")
    run_cmd(
        f"python utils/plagiarism_check.py "
        f"--test-pdf {pdf_path} "
        f"--output {plagiarism_out}"
    )

    # === Step 5: Factual Check ===
    factual_out = os.path.join(args.out_dir, "factual.json")
    run_cmd(
        f"python utils/factual_check.py "
        f"--path {pdf_path} "
        f"--topic {args.topic} "
        f"--output {factual_out}"
    )

    # === Step 6: Claim Mapping ===
    claim_out = os.path.join(args.out_dir, "claim_mapping.json")
    run_cmd(
        f"python utils/claim_mapping.py "
        f"--new_pdf {pdf_path} "
        f"--similar_json {novelty_out} "
        f"--claim_threshold 0.70 "
        f"--out_dir {args.out_dir}"
    )

    # === Step 7: Review Synthesis ===
    review_out = os.path.join(args.out_dir, "review.txt")
    run_cmd(
        f"python utils/llm_review_synthesis.py "
        f"--paper_dir {args.out_dir} "
        f"--output {review_out}"
    )

    print(f"\nFull pipeline complete! Results saved in: {args.out_dir}")

if __name__ == "__main__":
    main()
