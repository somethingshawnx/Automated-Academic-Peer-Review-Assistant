#!/usr/bin/env python3
"""
Updated llm_review_synthesis.py
- Computes novelty from novelty.json "results" similarities (max/top-k)
- Uses plagiarism/paraphrase overlaps and claim mapping to produce strong evidence
- Provides configurable thresholds and explicit exact-match override (reject)
- Better edge-case handling and human-readable evidence
"""
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from textwrap import dedent
from statistics import mean

# ---------------- Config (tweak as needed) ----------------
NOVELTY_SCORE_FUNC = lambda best_sim: max(0, min(10, int(round((1.0 - best_sim) * 10))))  # best_sim in [0,1]
NOVELTY_WARN_THRESHOLD = 0.4      # if best_sim >= 0.6 we should warn (i.e. not novel)
PLAGIARISM_SCORE_THRESH = 0.7    # paraphrase similarity >= this is considered high paraphrase overlap
EXACT_MATCH_THRESHOLD = 0.995    # treat >= this as an exact/near-exact match -> auto-reject
TOP_EVIDENCE = 3                  # how many top overlaps/claims to show
# ---------------------------------------------------------

def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def top_k_overlaps(plagiarism_json, k=TOP_EVIDENCE):
    """
    Return top k paraphrase/exact overlaps sorted by score descending
    """
    if not plagiarism_json:
        return []
    overlaps = []
    for o in plagiarism_json.get("paraphrase_overlap", []) + plagiarism_json.get("exact_overlap", []):
        # unify fields
        score = float(o.get("score", o.get("similarity", 0.0)))
        pdf_path = o.get("pdf_path") or o.get("matched_pdf") or o.get("source", "")
        snippet = (o.get("chunk") or o.get("text") or "")[:300]
        overlaps.append({"score": score, "pdf_path": pdf_path, "snippet": snippet, "type": o.get("type", "paraphrase_overlap")})
    overlaps.sort(key=lambda x: x["score"], reverse=True)
    return overlaps[:k]

def top_k_duplicate_claims(claims_json, k=TOP_EVIDENCE):
    if not claims_json or not isinstance(claims_json.get("mappings"), list):
        return []
    dups = [m for m in claims_json["mappings"] if m.get("similarity", 0.0) >= EXACT_MATCH_THRESHOLD]
    # if none exact, show top by similarity
    if not dups:
        mappings = sorted(claims_json["mappings"], key=lambda x: x.get("similarity", 0.0), reverse=True)
        return mappings[:k]
    return dups[:k]

def compute_novelty_score(novelty_json):
    """
    Compute novelty score from novelty.json.
    expected novelty_json format: {"pdf": "...", "results": [ {"similarity": 0.73, ...}, ... ] }
    We'll compute:
      - best_sim (max similarity)
      - mean_topk (mean of top 3)
      - novelty_score (0-10) via NOVELTY_SCORE_FUNC
    """
    if not novelty_json:
        return {"best_sim": 0.0, "mean_topk": 0.0, "score": 10, "num_results": 0}
    results = novelty_json.get("results", novelty_json.get("similar_papers", []) or [])
    if not results:
        return {"best_sim": 0.0, "mean_topk": 0.0, "score": 10, "num_results": 0}
    sims = [float(r.get("similarity", r.get("score", 0.0))) for r in results]
    best_sim = max(sims) if sims else 0.0
    # mean of top-k
    topk = sorted(sims, reverse=True)[:TOP_EVIDENCE]
    mean_topk = float(mean(topk)) if topk else 0.0
    score = NOVELTY_SCORE_FUNC(best_sim)
    return {"best_sim": best_sim, "mean_topk": mean_topk, "score": score, "num_results": len(results)}

def compute_plagiarism_summary(plag_json):
    """
    Summarize counts, top overlaps, and whether any exact match exists.
    """
    if not plag_json:
        return {"paraphrase_count": 0, "exact_count": 0, "top_overlaps": [], "has_exact": False}
    parap = plag_json.get("paraphrase_overlap", []) or []
    exact = plag_json.get("exact_overlap", []) or []
    top = top_k_overlaps(plag_json, k=TOP_EVIDENCE)
    has_exact = any(float(x.get("score", x.get("similarity", 0.0))) >= EXACT_MATCH_THRESHOLD for x in parap + exact)
    return {"paraphrase_count": len(parap), "exact_count": len(exact), "top_overlaps": top, "has_exact": has_exact}

def build_final_recommendation(scores, strengths, weaknesses, suggestions, plagiarism_summary, claim_dups):
    """
    Build a professional final recommendation string, with strong override rules:
     - If any exact-match plagiarism or claim duplicate -> Reject
     - Else follow score thresholds to recommend
    """
    # Hard override: exact reuse -> Reject
    if plagiarism_summary.get("has_exact") or (claim_dups and any(m.get("similarity", 0.0) >= EXACT_MATCH_THRESHOLD for m in claim_dups)):
        return dedent("""\
        The analysis found near-exact reuse of prior text/claims (exact-match similarity >= {:.3f}). This is a serious integrity issue.
        Recommendation: **Reject**. The manuscript must be substantially rewritten to remove copied content, clearly attribute prior work, and re-state original contributions before reconsideration.
        """).format(EXACT_MATCH_THRESHOLD)

    # otherwise score-based
    novelty = scores.get("Novelty", 5)
    plagiarism = scores.get("Plagiarism", 5)
    factual = scores.get("Factual Accuracy", 5)
    claims = scores.get("Claims (Citation Quality)", 5)

    parts = []
    if strengths:
        parts.append("Strengths: " + "; ".join(strengths) + ".")
    if weaknesses:
        parts.append("Weaknesses: " + "; ".join(weaknesses) + ".")
    if suggestions:
        parts.append("Suggestions: " + "; ".join(suggestions) + ".")

    # Determine decision
    if plagiarism <= 2 or novelty <= 2:
        decision = " **Decision: Reject** — Serious originality/plagiarism concerns. Please substantially rewrite and resubmit."
    elif novelty <= 4 or plagiarism <= 4:
        decision = " **Decision: Major Revisions** — The manuscript has important issues (novelty/originality) to address."
    elif factual <= 5:
        decision = " **Decision: Minor Revisions** — Fix factual issues and clarify methods/results."
    else:
        decision = " **Decision: Accept with Minor Revisions** — Overall solid, but polish presentation and address minor issues."

    return "\n".join(parts) + "\n\n" + decision

def synthesize_report(paper_dir: Path, output_file: Path, dry_run=False):
    # load files
    citation = load_json(paper_dir / "citation_report.json")
    novelty = load_json(paper_dir / "novelty.json")
    plagiarism = load_json(paper_dir / "plagiarism.json")
    factual = load_json(paper_dir / "factual.json")
    claim_file = next(paper_dir.glob("*claim_mapping*.json"), None)
    claims = load_json(claim_file) if claim_file else None

    # Compute novelty properly
    nov_stats = compute_novelty_score(novelty)
    best_sim = nov_stats["best_sim"]
    novelty_score = nov_stats["score"]

    # Plagiarism summary
    plag_sum = compute_plagiarism_summary(plagiarism)

    # Claim mapping duplicates
    claim_dups = top_k_duplicate_claims(claims, k=TOP_EVIDENCE) if claims else []

    # Build strengths, weaknesses, suggestions
    strengths = []
    weaknesses = []
    suggestions = []

    # Citation strengths/weaknesses
    cit_score = citation.get("analysis", {}).get("citation_quality_score") if citation else None
    if cit_score is not None:
        if cit_score >= 7:
            strengths.append("Good citation quality (diverse and recent references).")
        elif cit_score < 5:
            weaknesses.append("Citation quality is low or many missing DOIs/outdated references.")
            suggestions.append("Update references and include more recent and diverse citations.")
    else:
        strengths.append("Citation analysis unavailable.")

    # Novelty reasoning
    if nov_stats["num_results"] == 0:
        strengths.append("No similar papers found in the corpus (novel signal).")
    else:
        if best_sim >= EXACT_MATCH_THRESHOLD:
            weaknesses.append(f"Top similarity = {best_sim:.3f} (near-exact match with an existing paper).")
            suggestions.append("Rewrite sections identical to prior work and clearly cite them.")
        elif best_sim >= PLAGIARISM_SCORE_THRESH:
            weaknesses.append(f"High similarity found (top similarity = {best_sim:.3f}).")
            suggestions.append("Clarify novelty and explicitly contrast contributions vs. the closest prior art.")
        else:
            strengths.append(f"Top similarity is {best_sim:.3f} — suggests some overlap but not critical.")

    # Plagiarism reasoning
    if not plagiarism:
        strengths.append("No plagiarism analysis available.")
    else:
        if plag_sum["paraphrase_count"] + plag_sum["exact_count"] == 0:
            strengths.append("No plagiarism detected.")
        else:
            weaknesses.append(f"{plag_sum['paraphrase_count'] + plag_sum['exact_count']} plagiarism/paraphrase overlaps detected.")
            # add evidence bullet
            top_ov = plag_sum["top_overlaps"]
            if top_ov:
                suggestions.append("Key overlaps found — see evidence list in report; consider rewriting those parts.")
    # Factual reasoning
    factual_issues = 0
    if factual:
        hard = len(factual.get("issues", {}).get("hard_checks", []) or [])
        stat = len(factual.get("issues", {}).get("statistical_checks", []) or [])
        factual_issues = hard + stat
        if factual_issues > 0:
            weaknesses.append(f"Factual issues detected ({factual_issues}).")
            suggestions.append("Correct factual inconsistencies and double-check units/values.")
        else:
            strengths.append("Factual consistency verified.")

    # Scores (0-10)
    # Novelty uses computed novelty_score
    plagiarism_score = 10
    if plagiarism:
        total_overlaps = plag_sum["paraphrase_count"] + plag_sum["exact_count"]
        if total_overlaps == 0:
            plagiarism_score = 10
        elif total_overlaps < 5:
            plagiarism_score = 5
        else:
            plagiarism_score = 2

    factual_score = 10 if factual_issues == 0 else max(1, 8 - factual_issues)
    claims_score = int(round(cit_score)) if cit_score is not None else 5

    scores = {
        "Novelty": novelty_score,
        "Claims (Citation Quality)": claims_score,
        "Plagiarism": plagiarism_score,
        "Factual Accuracy": factual_score
    }

    # Build Claim label output (TRUE/FALSE) - map claim mapping "is_novel" -> supported?
    claim_label_lines = []
    if claims:
        for m in claims.get("mappings", []):
            # user mapping: "is_novel": False => claim duplicates existing -> NOT novel.
            # We want TRUE if claim is supported by evidence (i.e., appears in corpus / citation support)
            is_duplicate = not bool(m.get("is_novel", False))
            label = "FALSE" if is_duplicate else "TRUE"  # FALSE => not novel / potentially unsupported as original
            claim_label_lines.append(f"{label} → {m.get('claim')[:180]} (sim={m.get('similarity')})")
    else:
        claim_label_lines = ["No candidate claims were extracted from the paper. To enable claim labeling, ensure claim_mapping step is run and produces mappings."]

    # Evidence lists
    overlap_lines = []
    if plagiarism:
        top_ov = plag_sum["top_overlaps"]
        if not top_ov:
            overlap_lines.append("No overlaps reported by plagiarism step.")
        else:
            for ov in top_ov:
                if ov.get("link"):
                    overlap_lines.append(f"- [{ov['pdf_name']}]({ov['link']}) (score={ov['score']:.3f}) — {ov['snippet'][:200]}...")
                else:
                    overlap_lines.append(f"- {ov.get('pdf_name', ov.get('pdf_path'))} (score={ov['score']:.3f}) — {ov['snippet'][:200]}...")
    else:
        overlap_lines.append("No plagiarism JSON found.")

    # duplicate claims evidence
    dup_lines = []
    if claim_dups:
        for d in claim_dups:
            dup_lines.append(f"- sim={d.get('similarity'):.3f} — {d.get('claim')[:200]}...")
    else:
        if claims:
            # maybe no exact duplicates
            top_claims = sorted(claims.get("mappings", []), key=lambda x: x.get("similarity", 0.0), reverse=True)[:TOP_EVIDENCE]
            for tc in top_claims:
                dup_lines.append(
                    f"- sim={d.get('similarity'):.3f} — {d.get('matched_paper_title', 'Unknown')} "
                    f"({d.get('matched_paper_link', '')}) — {d.get('claim')[:200]}..."
                )
        else:
            dup_lines.append("No claim mapping available.")

    # Final recommendation (with override)
    final_reco = build_final_recommendation(scores, strengths, weaknesses, suggestions, plag_sum, claim_dups)

    # Build report text
    report = dedent(f"""
    **1. Summary of the Paper**
    This paper addresses a research problem of interest. Top similarity to corpus: {best_sim:.3f}. Citation quality score: {cit_score if cit_score is not None else 'N/A'}.

    **2. Strengths**
    - {"; ".join(strengths) if strengths else 'None identified.'}

    **3. Weaknesses**
    - {"; ".join(weaknesses) if weaknesses else 'None identified.'}

    **4. Suggestions for Improvement**
    - {"; ".join(suggestions) if suggestions else 'No major suggestions provided.'}

    **5. Section-wise Scores (0–10 each)**
    - Novelty: {scores['Novelty']}
    - Claims (Citation Quality): {scores['Claims (Citation Quality)']}
    - Plagiarism: {scores['Plagiarism']}
    - Factual Accuracy: {scores['Factual Accuracy']}

    **6. Claim Labels (TRUE/FALSE)**
    {"\n".join(claim_label_lines)}

    **7. Plagiarism / Overlap Evidence (top {TOP_EVIDENCE})**
    {"\n".join(overlap_lines)}

    **8. Duplicate Claim Evidence (top {TOP_EVIDENCE})**
    {"\n".join(dup_lines)}

    **9. Final Recommendation**
    {final_reco}

    ---
    Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """)

    if dry_run:
        print("\n=== DRY RUN REPORT ===\n")
        print(report)
        return report
    else:
        os.makedirs(output_file.parent, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Review saved to {output_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description="Synthesize review report from pipeline artifacts")
    parser.add_argument("--paper_dir", type=str, required=True, help="Directory with intermediate results (citation_report.json, novelty.json, plagiarism.json, factual.json, claim_mapping.json)")
    parser.add_argument("--output", type=str, default=None, help="Output review file (defaults to paper_dir/review.txt)")
    parser.add_argument("--dry-run", action="store_true", help="Print the report instead of saving")
    args = parser.parse_args()

    paper_dir = Path(args.paper_dir)
    output_file = Path(args.output) if args.output else paper_dir / "review.txt"
    synth = synthesize_report(paper_dir, output_file, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
