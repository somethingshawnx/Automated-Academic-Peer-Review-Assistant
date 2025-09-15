import os, json, argparse
from collections import defaultdict
from statistics import mean, pstdev
from PyPDF2 import PdfReader
from pint import UnitRegistry
import re

FAISS_DIR = "data/faiss_indexes"
ureg = UnitRegistry()
Q_ = ureg.Quantity

def extract_numeric_mentions(text):
    """Extract numeric values + units from text (very naive regex)."""
    mentions = []
    pattern = re.compile(r"([-+]?\d*\.?\d+)\s*([a-zA-Zµ%]*)")
    for match in pattern.finditer(text):
        val, unit = match.groups()
        try:
            value = float(val)
        except:
            continue
        mention = {
            "value": value,
            "unit": unit or None,
            "kind": "number",
            "si_unit": None,
            "value_si": None
        }
        mentions.append(mention)
    return mentions

def bind_metric_labels(mentions):
    """Bind mentions to SI units using pint."""
    for m in mentions:
        if m["unit"]:
            try:
                q = Q_(m["value"], m["unit"]).to_base_units()
                m["si_unit"] = str(q.units)
                m["value_si"] = q.magnitude
            except Exception:
                m["si_unit"] = None
                m["value_si"] = None
        else:
            m["si_unit"] = None
            m["value_si"] = m["value"]

def sanity_checks(mentions):
    """Check for impossible values (e.g., negative percentages)."""
    issues = []
    for m in mentions:
        if m["unit"] == "%" and (m["value"] < 0 or m["value"] > 100):
            issues.append(f"Invalid percentage: {m['value']}%")
    return issues

def internal_consistency_checks(mentions):
    """Naive check: flag if same unit appears with wildly different scales."""
    issues = []
    grouped = defaultdict(list)
    for m in mentions:
        if m["si_unit"] and m["value_si"] is not None:
            grouped[m["si_unit"]].append(m["value_si"])
    for unit, values in grouped.items():
        if len(values) > 1 and max(values) > 1000 * min(values):
            issues.append(f"Inconsistent scale for {unit}: min={min(values)}, max={max(values)}")
    return issues

def statistical_plausibility_checks(mentions, stats, z_thresh=3.0):
    """Check if values are statistical outliers compared to corpus stats."""
    issues = []
    for m in mentions:
        if m["si_unit"] and m["value_si"] is not None:
            key = f"{m['kind']}::{m['si_unit']}"
            if key in stats:
                mu = stats[key]["mean"]
                sigma = stats[key]["std"]
                if sigma > 0 and abs(m["value_si"] - mu) > z_thresh * sigma:
                    issues.append(f"Outlier {m['value_si']} {m['si_unit']} vs mean {mu}±{sigma}")
    return issues

# ---------------- Main factual check ----------------
def extract_text_from_pdf(pdf_path, max_chars=20000):
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text[:max_chars]

def read_text(path: str) -> str:
    return extract_text_from_pdf(path) if path.lower().endswith(".pdf") else open(path, "r", encoding="utf-8").read()

def build_corpus_stats_from_mapping(mapping_path: str) -> dict:
    if not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    agg = defaultdict(list)
    for entry in mapping.values():
        txt_path = entry.get("text_path")
        if not txt_path or not os.path.exists(txt_path):
            continue
        text = open(txt_path, "r", encoding="utf-8").read()
        mentions = extract_numeric_mentions(text)
        bind_metric_labels(mentions)
        for m in mentions:
            if m["value_si"] is not None:
                key = f"{m['kind']}::{m.get('si_unit')}"
                agg[key].append(float(m["value_si"]))

    stats = {}
    for k, vals in agg.items():
        if len(vals) >= 10:
            stats[k] = {
                "count": len(vals),
                "mean": float(mean(vals)),
                "std": float(pstdev(vals)),
                "min": float(min(vals)),
                "max": float(max(vals))
            }
    return stats

def factual_check(path: str, topic: str, z_thresh: float = 3.0):
    # Load text & mentions
    text = read_text(path)
    mentions = extract_numeric_mentions(text)
    bind_metric_labels(mentions)

    # Sanity + internal checks
    hard_issues = sanity_checks(mentions) + internal_consistency_checks(mentions)

    # Stats from FAISS mapping
    mapping_path = os.path.join(FAISS_DIR, f"{topic}_mapping.json")
    stats = build_corpus_stats_from_mapping(mapping_path)

    # Statistical plausibility
    stat_issues = statistical_plausibility_checks(mentions, stats, z_thresh)

    return {
        "file": path,
        "topic": topic,
        "num_mentions": len(mentions),
        "mentions": mentions[:2000],
        "issues": {
            "hard_checks": hard_issues,
            "statistical_checks": stat_issues
        },
        "corpus_stats_available_for": sorted(stats.keys())
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factual Verification with FAISS corpus")
    parser.add_argument("--path", type=str, required=True, help="Path to PDF/TXT file")
    parser.add_argument("--topic", type=str, required=True, help="Research topic (matches Step 3 index)")
    parser.add_argument("--z_thresh", type=float, default=3.0)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    results = factual_check(args.path, args.topic, args.z_thresh)
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))
